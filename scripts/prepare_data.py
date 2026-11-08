"""End-to-end data preparation: CSV -> train/val/test.jsonl.

Contract: INTERFACES.md section 13 (and 4, 5, 7).

    python scripts/prepare_data.py --csv "data/raw/Processed Dataset.csv" --out-dir data/processed

Steps (all reused from src/reviewlens, nothing re-implemented here):
    preprocess.load_reviews   read CSV read-only, map columns in memory, drop empty/duplicates
    preprocess.clean_text     clean + normalise every review, drop reviews that become empty
    preprocess.split_data     stratified on sentiment, 80/10/10, seed 42
    labeling.label_review     silver insight label per review
    oversampling              negative reviews duplicated in the TRAIN split only
    write                     train.jsonl / val.jsonl / test.jsonl in --out-dir

The tokenizer is NOT fitted here: the vocabulary is built later from
train.jsonl only (tokenizer.py, Person 2). The original CSV is never written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Make `src/` importable when run as `python scripts/prepare_data.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewlens.config import Paths  # noqa: E402
from reviewlens.dataset import MAX_REVIEW_TOKENS  # noqa: E402
from reviewlens.labeling import label_review  # noqa: E402
from reviewlens.preprocess import clean_text, load_reviews, split_data  # noqa: E402

#: Fields of every jsonl row (INTERFACES.md 7). ReviewInsightDataset requires
#: review/positives/pain_points/suggestions; the rest is kept for analysis.
RECORD_FIELDS = (
    "review", "sentiment", "rating", "item_id", "positives", "pain_points", "suggestions",
)
SPLITS = ("train", "val", "test")
DEFAULT_NEG_OVERSAMPLE = 1.5


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
def _json_scalar(value: Any) -> Any:
    """NaN/NA -> None, numpy scalars -> Python, integral floats (IDs) -> int."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _item_id(value: Any) -> Any:
    value = _json_scalar(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _rating(value: Any) -> float | None:
    value = _json_scalar(value)
    return None if value is None else float(value)


def build_records(df: pd.DataFrame) -> list[dict]:
    """Rows with an already-cleaned ``review`` -> jsonl records with silver labels."""
    records: list[dict] = []
    for row in df.itertuples(index=False):
        sentiment = _json_scalar(row.sentiment)
        label = label_review(row.review, sentiment)
        records.append({
            "review": row.review,
            "sentiment": sentiment,
            "rating": _rating(row.rating),
            "item_id": _item_id(row.item_id),
            "positives": list(label.positives),
            "pain_points": list(label.pain_points),
            "suggestions": list(label.suggestions),
        })
    return records


def oversample_negatives(records: list[dict], factor: float, seed: int) -> list[dict]:
    """Return ``records`` where negative reviews appear ``factor`` times on average.

    ``factor=1`` returns the input unchanged. Otherwise every negative record is
    repeated ``floor(factor) - 1`` extra times, a seeded random subset of size
    ``round(frac(factor) * n_neg)`` once more, and the result is shuffled with
    the same seed (deterministic). Apply to the TRAIN split only.
    """
    if not (isinstance(factor, (int, float)) and math.isfinite(factor) and factor >= 1):
        raise ValueError(f"neg_oversample must be a finite number >= 1, got {factor!r}")
    if factor == 1:
        return list(records)

    rng = np.random.default_rng(seed)
    negatives = [r for r in records if r["sentiment"] == "negative"]
    extra = [dict(r) for _ in range(int(factor) - 1) for r in negatives]
    n_frac = int(round((factor - int(factor)) * len(negatives)))
    if n_frac:
        picked = rng.choice(len(negatives), size=n_frac, replace=False)
        extra += [dict(negatives[i]) for i in sorted(picked)]
    combined = list(records) + extra
    return [combined[i] for i in rng.permutation(len(combined))]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    """Write one strict-JSON object per line (UTF-8, no NaN), atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _distribution(records: Sequence[dict], key: str) -> dict[str, int]:
    return dict(Counter(str(r[key]) for r in records).most_common())


def prepare(
    csv_path: Path,
    out_dir: Path,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    neg_oversample: float = DEFAULT_NEG_OVERSAMPLE,
) -> dict:
    """Run the whole pipeline and return statistics (also printed by ``main``)."""
    csv_path, out_dir = Path(csv_path), Path(out_dir)
    if neg_oversample < 1:  # validate before doing any work
        raise ValueError(f"neg_oversample must be >= 1, got {neg_oversample}")

    df = load_reviews(csv_path)
    n_loaded = len(df)
    df = df.assign(review=df["review"].map(clean_text))
    df = df[df["review"] != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"no usable reviews in {csv_path} after cleaning")

    parts = dict(zip(SPLITS, split_data(df, val_frac=val_frac, test_frac=test_frac, seed=seed)))
    records = {name: build_records(part) for name, part in parts.items()}
    split_sizes = {name: len(recs) for name, recs in records.items()}
    records["train"] = oversample_negatives(records["train"], neg_oversample, seed)

    paths = Paths(raw_csv=csv_path, data_dir=out_dir)
    out_paths = {"train": paths.train_jsonl, "val": paths.val_jsonl, "test": paths.test_jsonl}
    for name in SPLITS:
        write_jsonl(out_paths[name], records[name])

    # --- statistics -----------------------------------------------------
    texts = {name: {r["review"] for r in records[name]} for name in SPLITS}
    all_recs = [r for name in SPLITS for r in records[name]]
    truncated = int((df["review"].str.split().str.len() > MAX_REVIEW_TOKENS).sum())
    phrase_counts = {
        section: dict(Counter(p for r in all_recs for p in r[section]).most_common())
        for section in ("positives", "pain_points", "suggestions")
    }
    return {
        "csv": str(csv_path),
        "out_dir": str(out_dir),
        "files": {name: str(p) for name, p in out_paths.items()},
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "neg_oversample": neg_oversample,
        "n_loaded": n_loaded,
        "n_empty_after_clean": n_loaded - len(df),
        "n_used": len(df),
        "split_sizes": split_sizes,                                   # before oversampling
        "written_sizes": {name: len(records[name]) for name in SPLITS},
        "sentiment": {name: _distribution(records[name], "sentiment") for name in SPLITS},
        "aspect_phrases": phrase_counts,                               # incl. oversampled copies
        "pct_review_truncated": 100.0 * truncated / len(df),           # > MAX_REVIEW_TOKENS words
        "shared_review_texts": {                                       # identical text, other item
            "train&val": len(texts["train"] & texts["val"]),
            "train&test": len(texts["train"] & texts["test"]),
            "val&test": len(texts["val"] & texts["test"]),
        },
    }


def _print_stats(stats: dict) -> None:
    print(f"CSV        : {stats['csv']}")
    print(f"Output dir : {stats['out_dir']}")
    print(f"Loaded {stats['n_loaded']} reviews, {stats['n_empty_after_clean']} empty after "
          f"clean_text -> {stats['n_used']} used")
    print(f"Split (seed={stats['seed']}, val={stats['val_frac']}, test={stats['test_frac']}): "
          f"{stats['split_sizes']}")
    print(f"Written (train oversampled x{stats['neg_oversample']} negatives): "
          f"{stats['written_sizes']}")
    for name, dist in stats["sentiment"].items():
        total = sum(dist.values())
        pretty = ", ".join(f"{k} {v} ({100 * v / total:.1f}%)" for k, v in dist.items())
        print(f"  sentiment {name:5s}: {pretty}")
    for section, counts in stats["aspect_phrases"].items():
        top = ", ".join(f"{k} ({v})" for k, v in list(counts.items())[:5])
        print(f"  top {section}: {top}")
    print(f"Reviews truncated to {MAX_REVIEW_TOKENS} words: {stats['pct_review_truncated']:.2f}%")
    print(f"Identical review texts shared across splits: {stats['shared_review_texts']}")
    print("Tokenizer NOT fitted here; build the vocabulary from train.jsonl only.")


def main(argv: Sequence[str] | None = None) -> int:
    defaults = Paths()
    parser = argparse.ArgumentParser(description="CSV -> train/val/test.jsonl (ReviewLens GPT)")
    parser.add_argument("--csv", type=Path, default=defaults.raw_csv,
                        help=f"original dataset CSV, read-only (default: {defaults.raw_csv})")
    parser.add_argument("--out-dir", type=Path, default=defaults.data_dir,
                        help=f"output folder for *.jsonl (default: {defaults.data_dir}); "
                             "on Kaggle use /kaggle/working/data/processed")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg-oversample", type=float, default=DEFAULT_NEG_OVERSAMPLE,
                        help="average copies of each negative review in TRAIN (1 = off)")
    args = parser.parse_args(argv)

    stats = prepare(args.csv, args.out_dir, val_frac=args.val_frac, test_frac=args.test_frac,
                    seed=args.seed, neg_oversample=args.neg_oversample)
    _print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
