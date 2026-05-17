"""Download the 100k ambiguity JSONL from HF, cap each class, save one local file.

Source : morphllm/model-router-ambiguity-dataset :: ambiguity-100k-16-05-2026.jsonl
Output : data/ambiguity_capped_<per-class>.jsonl  (default 15k per class)

Usage:
    python -m morph_router.prepare_dataset --per-class 15000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "morphllm/model-router-ambiguity-dataset"
SRC_FILENAME = "ambiguity-100k-16-05-2026.jsonl"
VALID_LABELS = {"low", "medium", "high"}

logger = logging.getLogger("prepare_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=15000,
                   help="Max samples per label after capping (random subsample).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-name", default=None,
                   help="Output filename under data/. Defaults to ambiguity_capped_<per-class>.jsonl")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set. `eval \"$(grep ^export.*HF_TOKEN ~/.zshrc)\"`.")

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("downloading %s from %s ...", SRC_FILENAME, REPO_ID)
    src_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=SRC_FILENAME,
        token=token,
        local_dir=str(data_dir),
    )
    logger.info("downloaded to %s", src_path)

    by_class: dict[str, list[dict]] = {}
    other_counter: Counter[str] = Counter()
    bad_rows = 0
    with open(src_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_rows += 1
                continue
            label = row.get("ambiguity")
            if label in VALID_LABELS:
                by_class.setdefault(label, []).append(row)
            else:
                other_counter[str(label)] += 1

    raw_counts = {k: len(v) for k, v in by_class.items()}
    logger.info("raw 3-class counts : %s", raw_counts)
    if other_counter:
        logger.info("dropped non-{low,medium,high}: %s", dict(other_counter))
    if bad_rows:
        logger.warning("dropped %d malformed JSON lines", bad_rows)

    rng = random.Random(args.seed)
    kept: list[dict] = []
    for label in sorted(by_class):
        rows = by_class[label]
        if len(rows) > args.per_class:
            kept.extend(rng.sample(rows, args.per_class))
        else:
            kept.extend(rows)
    rng.shuffle(kept)

    out_name = args.out_name or f"ambiguity_capped_{args.per_class}.jsonl"
    out_path = data_dir / out_name
    with out_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    final_counts = dict(Counter(r["ambiguity"] for r in kept))
    logger.info("wrote %d rows to %s", len(kept), out_path)
    logger.info("final counts: %s", final_counts)


if __name__ == "__main__":
    main()
