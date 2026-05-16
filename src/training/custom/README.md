# Custom Training Env

This folder is just **an isolated Python env** for running upstream's training
scripts. It does not fork or rewrite any upstream code.

We need a managed env here because:

- `src/training/model_classifier/classifier_model_fine_tuning_lora/` ships no
  `requirements.txt` or `pyproject.toml`.
- On an aarch64 GH200 with CUDA 12.8, PyTorch must come from the SBSA wheel
  index (`https://download.pytorch.org/whl/cu128`), not default PyPI.

When we later need genuinely custom code (e.g. a 3-class trainer for the
morphllm dataset), it will live under sibling folders here so future
`git pull upstream main` never collides.

## One-time setup

```bash
cd src/training/custom
uv sync                              # creates .venv from pyproject.toml
eval "$(grep -E '^(export[[:space:]]+)?(WANDB_API_KEY|HF_TOKEN)=' ~/.zshrc)"
```

## Smoke test — run upstream's trainer

Activate this venv, then call upstream's script directly. Keep all outputs
inside `src/training/custom/runs/` so the upstream tree stays clean.

```bash
cd src/training/custom
source .venv/bin/activate
eval "$(grep -E '^(export[[:space:]]+)?(WANDB_API_KEY|HF_TOKEN)=' ~/.zshrc)"

export WANDB_PROJECT=vllm-sr-smoke
export WANDB_DIR="$PWD/runs"          # wandb/ folder lands here, not upstream
mkdir -p "$WANDB_DIR"

OUT="$PWD/runs/smoke-bert-r8-$(date +%H%M%S)"

python ../model_classifier/classifier_model_fine_tuning_lora/ft_linear_lora.py \
  --mode train \
  --model bert-base-uncased \
  --epochs 1 \
  --max-samples 200 \
  --lora-rank 8 --lora-alpha 16 \
  --batch-size 16 \
  --gpu-id 0 \
  --output-dir "$OUT"
```

What this verifies:

- Env: torch CUDA works on the GH200
- LoRA: PEFT applies adapters; trainable-param count is logged
- GPU: `set_gpu_device(gpu_id=0)` puts the model on `cuda:0`
- W&B: HF `Trainer` auto-reports because `wandb` is installed and `WANDB_API_KEY` is in the shell
