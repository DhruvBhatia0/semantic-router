"""LoRA fine-tune ModernBERT-base on morphllm/model-router-ambiguity-dataset.

Three-class classification (low / medium / high). bf16 on the GH200, class-
weighted cross-entropy to handle imbalance, W&B reporting via env vars.

Reference run command:

    cd src/training/custom && \\
      source .venv/bin/activate && \\
      eval "$(grep -E '^(export[[:space:]]+)?(WANDB_API_KEY|HF_TOKEN)=' ~/.zshrc)" && \\
      export WANDB_PROJECT=morph-router-ambiguity && \\
      export WANDB_DIR="$PWD/runs" && \\
      RUN="modernbert-r32-a64-ep10-effbs32-$(date +%Y%m%d-%H%M%S)" && \\
      export WANDB_NAME="$RUN" && \\
      mkdir -p "$WANDB_DIR" && \\
      OUT="$WANDB_DIR/$RUN" && \\
      LOG="$OUT.log" && \\
      nohup python -u -m morph_router.train \\
        --model modernbert-base \\
        --epochs 10 \\
        --batch-size 8 --grad-accum 4 \\
        --learning-rate 2e-4 \\
        --lora-rank 32 --lora-alpha 64 \\
        --max-length 8192 \\
        --logging-steps 10 \\
        --output-dir "$OUT" >"$LOG" 2>&1 &

Behaviour notes:
- One eval + one adapter checkpoint per epoch (`save_strategy="epoch"`).
- `save_total_limit=None` keeps every epoch's checkpoint under `$OUT/checkpoint-*`.
- After training, the best-by-macro_f1 checkpoint is reloaded and saved as the
  top-level adapter at `$OUT/`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Reuse upstream helpers; do not vendor them.
_UPSTREAM = Path(__file__).resolve().parents[2] / "model_classifier"
sys.path.insert(0, str(_UPSTREAM))
from common_lora_utils import (  # noqa: E402
    clear_gpu_memory,
    create_lora_config,
    get_max_length_for_model,
    resolve_model_path,
    set_gpu_device,
)

from morph_router.dataset import Splits, load_splits  # noqa: E402

logger = logging.getLogger("train")


# ---------- model ----------------------------------------------------------


def build_model(model_name: str, splits: Splits, lora_rank: int, lora_alpha: int):
    """Return (peft_model, tokenizer) on the active CUDA device."""
    base_path = resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        base_path,
        num_labels=splits.num_labels,
        id2label=splits.id2label,
        label2id=splits.label2id,
        dtype=torch.float32,  # FP32 weights; bf16 happens via TrainingArguments
    )

    lora_cfg = create_lora_config(model_name, rank=lora_rank, alpha=lora_alpha, dropout=0.1)
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(base, peft_cfg)
    model.print_trainable_parameters()
    # PEFT + gradient checkpointing: base weights are frozen, so we must
    # explicitly tell inputs to require grad or no gradient flows through.
    model.enable_input_require_grads()
    return model, tokenizer


# ---------- trainer with weighted CE --------------------------------------


class WeightedTrainer(Trainer):
    """HF Trainer with a fixed class-weight CE loss."""

    def __init__(self, class_weights: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self._class_weights.to(logits.device)
        loss = nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


# ---------- metrics --------------------------------------------------------


def make_compute_metrics(id2label):
    label_names = [id2label[i] for i in range(len(id2label))]

    def fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        out = {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted"),
        }
        per_class = f1_score(labels, preds, average=None, labels=list(range(len(label_names))))
        for name, val in zip(label_names, per_class):
            out[f"f1/{name}"] = float(val)
        return out

    return fn


# ---------- main -----------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not torch.cuda.is_available():
        sys.exit("CUDA not available; refusing to run on CPU.")

    device_str, gpu_id = set_gpu_device(gpu_id=args.gpu_id, auto_select=args.gpu_id is None)
    logger.info("device=%s", device_str)
    clear_gpu_memory()

    splits = load_splits(seed=args.seed, path=args.data_path)

    model, tokenizer = build_model(
        args.model, splits, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha
    )

    max_len = min(args.max_length, get_max_length_for_model(args.model))
    logger.info("tokenizing with max_length=%d", max_len)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    train_tok = splits.train.map(tok, batched=True, remove_columns=["text"])
    val_tok = splits.val.map(tok, batched=True, remove_columns=["text"])
    test_tok = splits.test.map(tok, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output_dir=%s", out_dir)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None,           # keep every epoch checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=["wandb"],
        run_name=os.environ.get("WANDB_NAME") or out_dir.name,
        seed=args.seed,
        dataloader_num_workers=2,
    )

    trainer = WeightedTrainer(
        class_weights=splits.class_weights,
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(splits.id2label),
    )

    started = time.time()
    trainer.train()
    logger.info("training took %.1fs", time.time() - started)

    logger.info("=== final test eval ===")
    test_metrics = trainer.evaluate(test_tok, metric_key_prefix="test")
    for k, v in test_metrics.items():
        logger.info("  %s = %s", k, v)

    # Saved artifacts: adapter, tokenizer, label map, full classification report
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "label_mapping.json").write_text(
        json.dumps(
            {"label2id": splits.label2id, "id2label": {int(k): v for k, v in splits.id2label.items()}},
            indent=2,
        )
    )
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    test_preds = trainer.predict(test_tok)
    y_pred = np.argmax(test_preds.predictions, axis=-1)
    y_true = test_preds.label_ids
    report = classification_report(
        y_true,
        y_pred,
        target_names=[splits.id2label[i] for i in range(splits.num_labels)],
        digits=4,
    )
    (out_dir / "classification_report.txt").write_text(report)
    print(report)
    logger.info("saved adapter + metrics to %s", out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="modernbert-base")
    p.add_argument("--epochs", type=int, default=3)              # upstream README + argparse default
    p.add_argument("--batch-size", type=int, default=8)          # per-step; smaller because max_length=8192
    p.add_argument("--grad-accum", type=int, default=4)          # effective batch = batch-size * grad-accum
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-rank", type=int, default=32)          # upstream script default
    p.add_argument("--lora-alpha", type=int, default=64)         # upstream script default
    p.add_argument("--max-length", type=int, default=8192,
                   help="cap (will also be clipped to model max).")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument(
        "--data-path",
        default=None,
        help="Optional local JSONL path. If unset, loads from the HF dataset repo.",
    )
    p.add_argument(
        "--output-dir",
        default=f"runs/morph-router-{time.strftime('%Y%m%d-%H%M%S')}",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
