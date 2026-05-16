"""One-shot: drop `ambiguity=='error'` rows and push the cleaned split back to HF.

After running this once, the upstream dataset becomes a clean 3-class set
(low / medium / high) and `dataset.py` no longer needs to filter at load time.

Usage:
    python -m morph_router.clean_dataset            # dry-run, prints what would change
    python -m morph_router.clean_dataset --push     # actually commits to HF
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter

from datasets import Dataset, DatasetDict, load_dataset

REPO_ID = "morphllm/model-router-ambiguity-dataset"
VALID_LABELS = {"low", "medium", "high"}

logger = logging.getLogger("clean_dataset")


def _load_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set; `eval \"$(grep ^export.*HF_TOKEN ~/.zshrc)\"` first")
    return token


def clean_split(ds: Dataset) -> Dataset:
    """Return a new Dataset with `ambiguity` ∈ VALID_LABELS only."""
    return ds.filter(lambda r: r["ambiguity"] in VALID_LABELS)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--push",
        action="store_true",
        help="Actually push the cleaned split to HF. Without this flag, runs as a dry-run.",
    )
    ap.add_argument(
        "--commit-message",
        default="drop ambiguity=='error' rows (3-class cleanup)",
    )
    args = ap.parse_args()

    token = _load_token()

    raw = load_dataset(REPO_ID, token=token)
    train = raw["train"]
    before = Counter(train["ambiguity"])
    logger.info("before: %d rows  %s", len(train), dict(before))

    cleaned = clean_split(train)
    after = Counter(cleaned["ambiguity"])
    logger.info("after : %d rows  %s", len(cleaned), dict(after))

    assert set(after.keys()) == VALID_LABELS, f"unexpected labels remain: {after}"
    assert len(cleaned) == sum(before[k] for k in VALID_LABELS)

    if not args.push:
        logger.info("dry-run: not pushing. Re-run with --push to commit.")
        return

    out = DatasetDict({"train": cleaned})
    logger.info("pushing to %s ...", REPO_ID)
    out.push_to_hub(REPO_ID, token=token, commit_message=args.commit_message)
    logger.info("done. verify at https://huggingface.co/datasets/%s", REPO_ID)


if __name__ == "__main__":
    main()
