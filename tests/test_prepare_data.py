"""Unit tests for scripts/prepare_data.py (INTERFACES.md section 13).

Only synthetic CSVs written to tmp_path are used; the real dataset is never read.
The tokenizer is not needed: compatibility with ReviewInsightDataset is checked
with a TEST-ONLY whitespace encoder (contract behaviour, IDs 0-9 = specials).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewlens.dataset import ReviewInsightDataset, collate_fn, format_target  # noqa: E402
from reviewlens.config import IGNORE_INDEX  # noqa: E402
from reviewlens.labeling import InsightLabel, label_review  # noqa: E402
from reviewlens.preprocess import clean_text  # noqa: E402

_spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts" / "prepare_data.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)

N_POS, N_NEG, N_NEU = 120, 50, 30          # 200 usable rows
REQUIRED = {"review", "positives", "pain_points", "suggestions"}   # ReviewInsightDataset

TEMPLATES = {
    "positive": ["Barang BAGUS bgt!! pengiriman cepat {i}", "mantaaap kualitas oke, harga murah {i}",
                 "Sesuai deskripsi, packing rapi 👍 {i}"],
    "negative": ["pengiriman lama sekali, gak sampai2 {i}", "barang rusak, tidak bisa nyala {i}",
                 "kemasan penyok tapi harga murah {i}"],
    "neutral": ["biasa saja {i}", "lumayan lah produk {i}"],
}


def make_df() -> pd.DataFrame:
    rows = []
    i = 0
    for sentiment, n in (("positive", N_POS), ("negative", N_NEG), ("neutral", N_NEU)):
        for k in range(n):
            tmpl = TEMPLATES[sentiment][k % len(TEMPLATES[sentiment])]
            rows.append({
                "itemId": 100000 + i,
                "name": "produk",
                "reviewContent": tmpl.format(i=f"ulasan{i}"),
                "rating": float(5 if sentiment == "positive" else 1 if sentiment == "negative" else 3),
                "sentiment_label": sentiment,
                "confidence": 0.9,
                "reviewTitle": "",
            })
            i += 1
    rows[3]["rating"] = float("nan")                          # NaN rating allowed
    rows += [
        {**rows[0]},                                          # duplicate (item_id, review)
        {**rows[1], "reviewContent": ""},                     # empty review
        {**rows[2], "itemId": 999001, "reviewContent": "!!! 😀😀 ???"},   # empty after clean_text
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def csv_path(tmp_path) -> Path:
    path = tmp_path / "input" / "Processed Dataset.csv"
    path.parent.mkdir()
    make_df().to_csv(path, index=False)
    return path


def run(csv_path: Path, out_dir: Path, **kw) -> dict:
    return prep.prepare(csv_path, out_dir, **kw)


def read_jsonl(path: Path) -> list[dict]:
    def no_nan(token):
        raise ValueError(f"non-standard JSON constant {token}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line, parse_constant=no_nan) for line in lines]


class WhitespaceEncoder:
    """Test-only stand-in for tokenizer.encode (INTERFACES.md 2.1 / 6)."""

    SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<review>",
                "<pos>", "<neg>", "<sug>", "<sep>", "<none>"]

    def __init__(self, texts):
        self.stoi = {t: i for i, t in enumerate(self.SPECIALS)}
        for t in texts:
            for w in t.split():
                self.stoi.setdefault(w, len(self.stoi))

    def encode(self, text):
        return [self.stoi.get(w, 1) for w in text.split()]


# ---------------------------------------------------------------------------
# Output files and JSONL validity
# ---------------------------------------------------------------------------
def test_writes_three_valid_jsonl_files(csv_path, tmp_path):
    out = tmp_path / "kaggle" / "working" / "data" / "processed"   # nested, not existing yet
    stats = run(csv_path, out)
    assert sorted(p.name for p in out.iterdir()) == ["test.jsonl", "train.jsonl", "val.jsonl"]
    for name in prep.SPLITS:
        path = out / f"{name}.jsonl"
        raw = path.read_bytes()
        assert raw.endswith(b"\n") and b"\r\n" not in raw
        recs = read_jsonl(path)                     # every line parses as strict JSON
        assert len(recs) == stats["written_sizes"][name] > 0


def test_no_tokenizer_fitting_and_no_temp_files(csv_path, tmp_path):
    out = tmp_path / "out"
    run(csv_path, out)
    names = {p.name for p in out.iterdir()}
    assert "vocab.json" not in names
    assert not any(n.endswith(".tmp") for n in names)


def test_original_csv_is_not_modified(csv_path, tmp_path):
    before = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    run(csv_path, tmp_path / "out")
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == before
    assert sorted(p.name for p in csv_path.parent.iterdir()) == ["Processed Dataset.csv"]


# ---------------------------------------------------------------------------
# Field completeness and content
# ---------------------------------------------------------------------------
def test_record_fields_and_types(csv_path, tmp_path):
    out = tmp_path / "out"
    run(csv_path, out)
    for name in prep.SPLITS:
        for rec in read_jsonl(out / f"{name}.jsonl"):
            assert tuple(rec) == prep.RECORD_FIELDS
            assert REQUIRED <= rec.keys()
            assert isinstance(rec["review"], str) and rec["review"].strip()
            assert rec["sentiment"] in {"positive", "negative", "neutral"}
            assert rec["rating"] is None or isinstance(rec["rating"], float)
            assert isinstance(rec["item_id"], int)
            for section in ("positives", "pain_points", "suggestions"):
                assert isinstance(rec[section], list)
                assert all(isinstance(p, str) and p.strip() for p in rec[section])
                assert len(rec[section]) <= 3


def test_reviews_are_cleaned_and_labels_match_labeling(csv_path, tmp_path):
    out = tmp_path / "out"
    run(csv_path, out)
    recs = [r for n in prep.SPLITS for r in read_jsonl(out / f"{n}.jsonl")]
    for rec in recs:
        assert clean_text(rec["review"]) == rec["review"]          # output of clean_text
        label = label_review(rec["review"], rec["sentiment"])
        assert rec["positives"] == label.positives
        assert rec["pain_points"] == label.pain_points
        assert rec["suggestions"] == label.suggestions
    assert any("tidak" in r["review"].split() for r in recs)       # "gak" normalised
    assert any(r["rating"] is None for r in recs)                  # NaN rating -> null


def test_empty_and_duplicate_rows_are_dropped(csv_path, tmp_path):
    stats = run(csv_path, tmp_path / "out")
    # 203 CSV rows: 1 duplicate + 1 empty dropped by load_reviews, 1 empty after clean_text.
    assert stats["n_loaded"] == 201
    assert stats["n_empty_after_clean"] == 1
    assert stats["n_used"] == 200


# ---------------------------------------------------------------------------
# Split ratio, stratification, determinism, leakage
# ---------------------------------------------------------------------------
def test_split_ratio_80_10_10(csv_path, tmp_path):
    stats = run(csv_path, tmp_path / "out", neg_oversample=1.0)
    assert stats["split_sizes"] == {"train": 160, "val": 20, "test": 20}
    assert stats["written_sizes"] == stats["split_sizes"]          # no oversampling


def test_split_is_stratified_on_sentiment(csv_path, tmp_path):
    out = tmp_path / "out"
    run(csv_path, out, neg_oversample=1.0)
    for name, n in (("train", 160), ("val", 20), ("test", 20)):
        counts = Counter(r["sentiment"] for r in read_jsonl(out / f"{name}.jsonl"))
        for sentiment, total in (("positive", N_POS), ("negative", N_NEG), ("neutral", N_NEU)):
            assert math.isclose(counts[sentiment] / n, total / 200, abs_tol=0.03), (name, counts)


def test_deterministic_with_same_seed(csv_path, tmp_path):
    run(csv_path, tmp_path / "a")
    run(csv_path, tmp_path / "b")
    for name in prep.SPLITS:
        assert (tmp_path / "a" / f"{name}.jsonl").read_bytes() == \
               (tmp_path / "b" / f"{name}.jsonl").read_bytes()


def test_different_seed_gives_different_split(csv_path, tmp_path):
    run(csv_path, tmp_path / "a", seed=42)
    run(csv_path, tmp_path / "b", seed=7)
    assert (tmp_path / "a" / "test.jsonl").read_bytes() != (tmp_path / "b" / "test.jsonl").read_bytes()


def test_no_row_leaks_across_splits(csv_path, tmp_path):
    out = tmp_path / "out"
    stats = run(csv_path, out)
    keys = {n: {(r["item_id"], r["review"]) for r in read_jsonl(out / f"{n}.jsonl")}
            for n in prep.SPLITS}
    assert not keys["train"] & keys["val"]
    assert not keys["train"] & keys["test"]
    assert not keys["val"] & keys["test"]
    assert sum(len(k) for k in keys.values()) == stats["n_used"]   # every row exactly once
    assert stats["shared_review_texts"] == {"train&val": 0, "train&test": 0, "val&test": 0}


# ---------------------------------------------------------------------------
# Oversampling (train only)
# ---------------------------------------------------------------------------
def test_oversampling_only_affects_train_negatives(csv_path, tmp_path):
    base, over = tmp_path / "base", tmp_path / "over"
    run(csv_path, base, neg_oversample=1.0)
    run(csv_path, over, neg_oversample=2.0)
    for name in ("val", "test"):
        assert (base / f"{name}.jsonl").read_bytes() == (over / f"{name}.jsonl").read_bytes()

    b = Counter(r["sentiment"] for r in read_jsonl(base / "train.jsonl"))
    o = Counter(r["sentiment"] for r in read_jsonl(over / "train.jsonl"))
    assert o["negative"] == 2 * b["negative"]
    assert o["positive"] == b["positive"] and o["neutral"] == b["neutral"]
    # Same unique rows; only negatives are repeated.
    unique = lambda p: {(r["item_id"], r["review"]) for r in read_jsonl(p)}  # noqa: E731
    assert unique(base / "train.jsonl") == unique(over / "train.jsonl")


def test_fractional_oversampling_default(csv_path, tmp_path):
    stats = run(csv_path, tmp_path / "out")                    # default 1.5
    assert prep.DEFAULT_NEG_OVERSAMPLE == 1.5
    n_neg = stats["sentiment"]["train"]["negative"]
    base_neg = round(N_NEG * 0.8)
    assert n_neg == base_neg + round(0.5 * base_neg)
    assert stats["written_sizes"]["train"] == 160 + round(0.5 * base_neg)


@pytest.mark.parametrize("factor", [0.5, 0, -1, float("nan"), float("inf")])
def test_oversample_rejects_invalid_factor(factor):
    with pytest.raises(ValueError):
        prep.oversample_negatives([{"sentiment": "negative"}], factor, seed=42)


def test_oversample_factor_one_is_identity():
    recs = [{"sentiment": "negative", "i": 1}, {"sentiment": "positive", "i": 2}]
    assert prep.oversample_negatives(recs, 1, seed=42) == recs


# ---------------------------------------------------------------------------
# Compatibility with ReviewInsightDataset / collate_fn
# ---------------------------------------------------------------------------
def test_outputs_load_into_review_insight_dataset(csv_path, tmp_path):
    out = tmp_path / "out"
    stats = run(csv_path, out)
    train = read_jsonl(out / "train.jsonl")
    # Vocabulary from TRAIN only (as tokenizer.py will do).
    enc = WhitespaceEncoder(
        [r["review"] for r in train]
        + [format_target(InsightLabel(r["positives"], r["pain_points"], r["suggestions"]))
           for r in train]
    )
    for name in prep.SPLITS:
        data = ReviewInsightDataset(out / f"{name}.jsonl", enc, block_size=128)
        assert len(data) == stats["written_sizes"][name]
        x, y = collate_fn([data[i] for i in range(min(16, len(data)))])
        assert x.shape == y.shape and x.shape[1] <= 128
        assert x.dtype == y.dtype == torch.long
        assert not torch.any(y == 0) and torch.all((y == IGNORE_INDEX) | (y > 0))


# ---------------------------------------------------------------------------
# Invalid input and CLI
# ---------------------------------------------------------------------------
def test_missing_csv(tmp_path):
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "nope.csv", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_required_column(tmp_path):
    path = tmp_path / "bad.csv"
    make_df().drop(columns=["sentiment_label"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="sentiment_label"):
        run(path, tmp_path / "out")


def test_all_reviews_empty_after_cleaning(tmp_path):
    path = tmp_path / "emoji.csv"
    pd.DataFrame({"itemId": [1, 2], "reviewContent": ["😀😀", "!!!"], "rating": [5, 4],
                  "sentiment_label": ["positive", "positive"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no usable reviews"):
        run(path, tmp_path / "out")


@pytest.mark.parametrize("kw", [{"val_frac": 0.6, "test_frac": 0.5}, {"neg_oversample": 0.5}])
def test_invalid_parameters(csv_path, tmp_path, kw):
    with pytest.raises(ValueError):
        run(csv_path, tmp_path / "out", **kw)


def test_cli_main(csv_path, tmp_path, capsys):
    out = tmp_path / "kaggle_working" / "data" / "processed"
    assert prep.main(["--csv", str(csv_path), "--out-dir", str(out)]) == 0
    assert {p.name for p in out.iterdir()} == {"train.jsonl", "val.jsonl", "test.jsonl"}
    printed = capsys.readouterr().out
    assert "Split (seed=42" in printed and "Tokenizer NOT fitted" in printed


def test_cli_defaults_follow_paths_contract():
    from reviewlens.config import Paths
    p = Paths()
    assert p.raw_csv == Path("data/raw/Processed Dataset.csv")
    assert p.data_dir == Path("data/processed")
