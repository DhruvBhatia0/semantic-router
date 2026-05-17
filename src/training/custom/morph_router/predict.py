"""Run the trained LoRA adapter on prompts and return the predicted ambiguity class.

Three ways to call it:

    # 1. One-shot CLI
    python -m morph_router.predict --adapter runs/.../  \\
        --text "look give me back my homepage"

    # 2. Stdin pipe (one prompt per line)
    cat prompts.txt | python -m morph_router.predict --adapter runs/.../ -

    # 3. From Python
    from morph_router.predict import Classifier
    clf = Classifier("runs/.../")           # local folder
    # or
    clf = Classifier("morphllm/model_router_05-17-2026")  # HF repo id
    clf.predict(["short query", "another"])
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Sequence

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MAX_LEN = 4096
LABEL_ORDER = ["low", "medium", "high"]


class Classifier:
    """Thin wrapper around a PEFT LoRA adapter for 3-class ambiguity classification.

    Public API: __init__(path), predict(texts) -> list[dict].
    """

    def __init__(
        self,
        adapter: str,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LEN,
    ) -> None:
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        peft_cfg = PeftConfig.from_pretrained(adapter, token=os.environ.get("HF_TOKEN"))
        base_name = peft_cfg.base_model_name_or_path

        # Recover label mapping from the adapter folder if available, otherwise
        # fall back to the canonical order used during training.
        self.id2label = self._read_id2label(adapter) or {i: n for i, n in enumerate(LABEL_ORDER)}

        base = AutoModelForSequenceClassification.from_pretrained(
            base_name,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id={v: k for k, v in self.id2label.items()},
            dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
        )
        self.model = PeftModel.from_pretrained(base, adapter, token=os.environ.get("HF_TOKEN"))
        self.model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(adapter, token=os.environ.get("HF_TOKEN"))

    @staticmethod
    def _read_id2label(adapter: str) -> dict[int, str] | None:
        for name in ("label_mapping.json",):
            try:
                with open(Path(adapter) / name) as f:
                    data = json.load(f)
                return {int(k): v for k, v in data["id2label"].items()}
            except (FileNotFoundError, KeyError, NotADirectoryError):
                continue
        return None

    @torch.inference_mode()
    def predict(self, texts: Sequence[str], batch_size: int = 16) -> List[dict]:
        """Return `[{'text', 'label', 'label_id', 'confidence', 'probs'}, ...]`."""
        out: List[dict] = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i : i + batch_size])
            enc = self.tokenizer(
                chunk,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            preds = logits.argmax(-1).cpu().tolist()
            for text, p, pr in zip(chunk, preds, probs):
                out.append(
                    {
                        "text": text,
                        "label_id": int(p),
                        "label": self.id2label[int(p)],
                        "confidence": float(pr[p]),
                        "probs": {self.id2label[j]: float(v) for j, v in enumerate(pr)},
                    }
                )
        return out


# ---------- CLI ----------------------------------------------------------------


def _read_inputs(text_arg: str | None) -> List[str]:
    if text_arg is None:
        return [line.rstrip("\n") for line in sys.stdin if line.strip()]
    return [text_arg]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adapter",
        required=True,
        help="Local adapter folder or HF model repo id.",
    )
    p.add_argument(
        "--text",
        default=None,
        help="Prompt to classify. Omit to read prompts from stdin (one per line).",
    )
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--json", action="store_true", help="Emit JSON one result per line.")
    args = p.parse_args()

    clf = Classifier(args.adapter, max_length=args.max_length)
    inputs = _read_inputs(args.text)
    if not inputs:
        sys.exit("no input prompts")

    for result in clf.predict(inputs):
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            probs = "  ".join(f"{k}={v:.3f}" for k, v in result["probs"].items())
            preview = result["text"].replace("\n", " ")[:80]
            print(f"[{result['label']:<6} conf={result['confidence']:.3f}]  {probs}  | {preview}")


if __name__ == "__main__":
    main()
