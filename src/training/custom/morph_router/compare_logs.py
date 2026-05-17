"""Run prompts from a router request log through our LoRA and compare.

Input file layout (one record per row, list at the top):
    [
      {
        "prompt"     : '{"input":"...","mode":"balanced","provider":"raw"}',
        "completion" : '{"confidence":0.94,"difficulty":"hard"}',
        ...
      },
      ...
    ]

The existing router predicts `difficulty` (easy / medium / hard) with its own
confidence. We predict `ambiguity` (low / medium / high). These are different
concepts:
    - difficulty  : how hard is the task to solve?
    - ambiguity   : how well-specified is the request?

This script:
    1. Loads the log
    2. Runs every prompt through our adapter
    3. Writes a CSV with both side-by-side
    4. Prints a contingency table + the most informative disagreement examples

Usage:
    python -m morph_router.compare_logs                          # uses HF adapter
    python -m morph_router.compare_logs --adapter runs/.../      # uses local
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

from morph_router.predict import Classifier

logger = logging.getLogger("compare_logs")

DEFAULT_LOG = Path(__file__).resolve().parent / "data" / "request_logs_router_recent_100.json"
DEFAULT_ADAPTER = "morphllm/model_router_05-17-2026"


# ---------- io ----------------------------------------------------------------


def load_records(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    records: list[dict] = []
    for row in raw:
        prompt_blob = row.get("prompt")
        completion_blob = row.get("completion")
        if not prompt_blob:
            continue
        try:
            prompt = json.loads(prompt_blob)
        except json.JSONDecodeError:
            continue
        text = prompt.get("input", "")
        if not text:
            continue
        existing = {}
        if completion_blob:
            try:
                existing = json.loads(completion_blob)
            except json.JSONDecodeError:
                pass
        records.append(
            {
                "id": row.get("id"),
                "text": text,
                "input_tokens": row.get("input_tokens"),
                "latency_ms": row.get("latency_ms"),
                "existing_difficulty": existing.get("difficulty"),
                "existing_confidence": existing.get("confidence"),
            }
        )
    return records


def entropy(probs: dict[str, float]) -> float:
    return -sum(v * math.log(max(v, 1e-12)) for v in probs.values())


# ---------- presentation -----------------------------------------------------


def print_distributions(records: list[dict]) -> None:
    existing = Counter(r.get("existing_difficulty") or "<none>" for r in records)
    new = Counter(r["new_label"] for r in records)
    print("\n=== existing difficulty distribution ===")
    for k in ("easy", "medium", "hard", "<none>"):
        if k in existing:
            print(f"  {k:<8} {existing[k]:>3}")
    print("\n=== our ambiguity prediction distribution ===")
    for k in ("low", "medium", "high"):
        print(f"  {k:<8} {new[k]:>3}")


def print_crosstab(records: list[dict]) -> None:
    """rows = existing difficulty; cols = our ambiguity label."""
    rows = ["easy", "medium", "hard", "<none>"]
    cols = ["low", "medium", "high"]
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        e = r.get("existing_difficulty") or "<none>"
        table[e][r["new_label"]] += 1

    print("\n=== contingency: existing_difficulty x new_ambiguity ===")
    header = f"{'':>8}  " + "  ".join(f"{c:>6}" for c in cols) + f"  {'total':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if row not in table:
            continue
        counts = [table[row][c] for c in cols]
        total = sum(counts)
        print(f"{row:>8}  " + "  ".join(f"{v:>6}" for v in counts) + f"  {total:>6}")


def print_examples(records: list[dict], n: int = 8) -> None:
    """Show diverse, informative examples sorted by our model's entropy."""
    sorted_by_entropy = sorted(records, key=lambda r: r["new_entropy"], reverse=True)
    print(f"\n=== {n} highest-entropy (most uncertain) predictions ===")
    for r in sorted_by_entropy[:n]:
        _print_one(r)

    sorted_by_conf = sorted(records, key=lambda r: r["new_confidence"], reverse=True)
    print(f"\n=== {n} most-confident predictions ===")
    for r in sorted_by_conf[:n]:
        _print_one(r)


def _print_one(r: dict) -> None:
    p = r["new_probs"]
    text = r["text"].replace("\n", " ")
    if len(text) > 110:
        text = text[:110] + "..."
    print(
        f"  [our: {r['new_label']:<6} c={r['new_confidence']:.2f} H={r['new_entropy']:.2f}] "
        f"[existing: {r.get('existing_difficulty') or '-':<6} "
        f"c={r.get('existing_confidence') or 0:.2f}] "
        f"low={p['low']:.2f} med={p['medium']:.2f} high={p['high']:.2f} | {text}"
    )


def write_csv(records: list[dict], out_path: Path) -> None:
    fields = [
        "id",
        "input_tokens",
        "latency_ms",
        "existing_difficulty",
        "existing_confidence",
        "new_label",
        "new_confidence",
        "new_entropy",
        "p_low",
        "p_medium",
        "p_high",
        "text",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            p = r["new_probs"]
            w.writerow(
                {
                    "id": r["id"],
                    "input_tokens": r["input_tokens"],
                    "latency_ms": r["latency_ms"],
                    "existing_difficulty": r.get("existing_difficulty"),
                    "existing_confidence": r.get("existing_confidence"),
                    "new_label": r["new_label"],
                    "new_confidence": r["new_confidence"],
                    "new_entropy": r["new_entropy"],
                    "p_low": p["low"],
                    "p_medium": p["medium"],
                    "p_high": p["high"],
                    "text": r["text"],
                }
            )


# ---------- main -------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "data" / "compare_logs_results.csv"),
    )
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    log_path = Path(args.log)
    records = load_records(log_path)
    logger.info("loaded %d records from %s", len(records), log_path)

    clf = Classifier(args.adapter)
    preds = clf.predict([r["text"] for r in records], batch_size=args.batch_size)
    for r, p in zip(records, preds):
        r["new_label"] = p["label"]
        r["new_confidence"] = p["confidence"]
        r["new_probs"] = p["probs"]
        r["new_entropy"] = entropy(p["probs"])

    print_distributions(records)
    print_crosstab(records)
    print_examples(records, n=args.examples)

    out_path = Path(args.out)
    write_csv(records, out_path)
    print(f"\n[ok] wrote {len(records)} rows to {out_path}")


if __name__ == "__main__":
    main()
