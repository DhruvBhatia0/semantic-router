"""Load morphllm/model-router-ambiguity-dataset and produce train/val/test splits.

Public surface is intentionally small: one function returns a `Splits` record
with everything the trainer needs (datasets, label maps, class weights). All
HF / sklearn / numpy details stay inside.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

REPO_ID = "morphllm/model-router-ambiguity-dataset"
# Semantic order, not alphabetic, so id=0 is the "least ambiguous" class.
LABEL_ORDER: List[str] = ["low", "medium", "high"]


@dataclass
class Splits:
    train: Dataset
    val: Dataset
    test: Dataset
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    class_weights: np.ndarray  # shape (num_labels,), aligned with id2label

    @property
    def num_labels(self) -> int:
        return len(self.label2id)


def load_splits(
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> Splits:
    """Load the dataset and produce a stratified 80/10/10 split."""
    token = os.environ.get("HF_TOKEN")
    raw = load_dataset(REPO_ID, split="train", token=token)
    logger.info("loaded %d rows from %s", len(raw), REPO_ID)

    label2id = {name: i for i, name in enumerate(LABEL_ORDER)}
    id2label = {i: name for name, i in label2id.items()}

    raw = raw.filter(lambda r: r["ambiguity"] in label2id)
    raw = raw.map(
        lambda r: {"text": r["prompt"], "label": label2id[r["ambiguity"]]},
        remove_columns=raw.column_names,
    )

    indices = list(range(len(raw)))
    labels = raw["label"]
    train_idx, hold_idx = train_test_split(
        indices,
        test_size=val_frac + test_frac,
        random_state=seed,
        stratify=labels,
    )
    hold_labels = [labels[i] for i in hold_idx]
    val_share = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        hold_idx,
        test_size=1 - val_share,
        random_state=seed,
        stratify=hold_labels,
    )

    train_ds = raw.select(train_idx)
    val_ds = raw.select(val_idx)
    test_ds = raw.select(test_idx)

    train_labels = np.asarray(train_ds["label"])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(label2id)),
        y=train_labels,
    )

    logger.info(
        "splits: train=%d val=%d test=%d  class_weights=%s",
        len(train_ds), len(val_ds), len(test_ds),
        {id2label[i]: round(float(w), 3) for i, w in enumerate(class_weights)},
    )

    return Splits(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        label2id=label2id,
        id2label=id2label,
        class_weights=class_weights.astype(np.float32),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = load_splits()
    print("first train row:", s.train[0])
    print("label2id        :", s.label2id)
    print("class_weights   :", s.class_weights)
