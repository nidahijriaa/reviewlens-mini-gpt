"""Unit tests for reviewlens.preprocess (INTERFACES.md sections 4 and 14).

Uses small synthetic CSV files; the real dataset is not required.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewlens.preprocess import (  # noqa: E402
    COLUMN_MAP,
    NEGATION_MAP,
    NEGATIONS,
    OPTIONAL_COLUMNS,
    SLANG_MAP,
    clean_text,
    load_reviews,
    normalize_token,
    split_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _base_row(**overrides) -> dict:
    row = {
        "name": "user",                # ignored column
        "reviewContent": "barang bagus",
        "reviewTitle": "judul",        # ignored column
        "sentiment_label": "positive",
        "confidence": 0.9,
        "rating": 5.0,
        "itemId": 100,
    }
    row.update(overrides)
    return row


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants_match_contract():
    assert COLUMN_MAP == {
        "reviewContent": "review",
        "sentiment_label": "sentiment",
        "rating": "rating",
        "itemId": "item_id",
    }
    assert OPTIONAL_COLUMNS == ("confidence",)
    assert NEGATIONS == ("tidak", "kurang", "belum", "bukan")
    assert set(NEGATION_MAP.values()) <= set(NEGATIONS)


def test_slang_map_has_no_negation_keys_and_is_idempotent():
    assert not set(SLANG_MAP) & set(NEGATION_MAP)
    # Values must be fixed points, otherwise clean_text would not be idempotent.
    for value in list(SLANG_MAP.values()) + list(NEGATION_MAP.values()):
        assert normalize_token(value) == value, value


# ---------------------------------------------------------------------------
# 1. Column mapping
# ---------------------------------------------------------------------------


def test_load_reviews_maps_columns(tmp_path):
    csv = _write_csv(tmp_path / "r.csv", [_base_row(), _base_row(reviewContent="kurang oke", itemId=101)])
    df = load_reviews(csv)
    assert list(df.columns) == ["review", "sentiment", "rating", "item_id", "confidence"]
    assert df.loc[0, "review"] == "barang bagus"
    assert df.loc[0, "sentiment"] == "positive"
    assert df.loc[0, "rating"] == 5.0
    assert df.loc[0, "item_id"] == 100
    assert df.loc[0, "confidence"] == pytest.approx(0.9)


def test_load_reviews_confidence_is_optional(tmp_path):
    row = _base_row()
    del row["confidence"]
    df = load_reviews(_write_csv(tmp_path / "r.csv", [row]))
    assert list(df.columns) == ["review", "sentiment", "rating", "item_id"]


def test_load_reviews_handles_utf8_bom(tmp_path):
    path = tmp_path / "bom.csv"
    pd.DataFrame([_base_row()]).to_csv(path, index=False, encoding="utf-8-sig")
    df = load_reviews(path)
    assert df.loc[0, "review"] == "barang bagus"


def test_load_reviews_does_not_modify_source_file(tmp_path):
    csv = _write_csv(tmp_path / "r.csv", [_base_row(), _base_row(), _base_row(reviewContent="")])
    before = _sha256(csv)
    mtime = csv.stat().st_mtime_ns
    load_reviews(csv)
    assert _sha256(csv) == before
    assert csv.stat().st_mtime_ns == mtime


# ---------------------------------------------------------------------------
# 2. Missing required column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", list(COLUMN_MAP))
def test_load_reviews_missing_required_column_raises(tmp_path, missing):
    row = _base_row()
    del row[missing]
    csv = _write_csv(tmp_path / "r.csv", [row])
    with pytest.raises(ValueError, match=missing):
        load_reviews(csv)


def test_load_reviews_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reviews(tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# 3. Empty reviews removed
# ---------------------------------------------------------------------------


def test_load_reviews_drops_empty_reviews(tmp_path):
    rows = [
        _base_row(reviewContent="bagus", itemId=1),
        _base_row(reviewContent=None, itemId=2),
        _base_row(reviewContent="", itemId=3),
        _base_row(reviewContent="   ", itemId=4),
        _base_row(reviewContent="jelek", itemId=5),
    ]
    df = load_reviews(_write_csv(tmp_path / "r.csv", rows))
    assert df["review"].tolist() == ["bagus", "jelek"]
    assert df.index.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# 4. Duplicates removed
# ---------------------------------------------------------------------------


def test_load_reviews_drops_duplicate_item_review_pairs(tmp_path):
    rows = [
        _base_row(reviewContent="bagus", itemId=1),
        _base_row(reviewContent="bagus", itemId=1),   # duplicate -> dropped
        _base_row(reviewContent="bagus", itemId=2),   # same text, other item -> kept
        _base_row(reviewContent="mantap", itemId=1),  # other text, same item -> kept
    ]
    df = load_reviews(_write_csv(tmp_path / "r.csv", rows))
    assert len(df) == 3
    assert not df.duplicated(subset=["item_id", "review"]).any()
    assert sorted(zip(df["item_id"], df["review"])) == [(1, "bagus"), (1, "mantap"), (2, "bagus")]


# ---------------------------------------------------------------------------
# 5. NaN rating allowed
# ---------------------------------------------------------------------------


def test_load_reviews_keeps_nan_rating(tmp_path):
    rows = [
        _base_row(reviewContent="a", rating=None, itemId=1),
        _base_row(reviewContent="b", rating=4.0, itemId=2),
    ]
    df = load_reviews(_write_csv(tmp_path / "r.csv", rows))
    assert len(df) == 2
    assert pd.isna(df.loc[0, "rating"])
    assert df.loc[1, "rating"] == 4.0
    assert pd.api.types.is_float_dtype(df["rating"])


# ---------------------------------------------------------------------------
# 6. Negation normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant", ["gak", "ga", "nggak", "ngga", "gk", "tdk", "enggak", "kagak"]
)
def test_negation_variants_become_tidak(variant):
    assert normalize_token(variant) == "tidak"
    assert clean_text(f"{variant} cepat") == "tidak cepat"


def test_negation_contract_examples():
    # Exact examples from INTERFACES.md section 14.
    assert clean_text("gak cepat") == "tidak cepat"
    assert clean_text("ga") == "tidak"
    assert clean_text("GAK cepat!!!") == "tidak cepat"
    assert clean_text("blm sampai, krg rapi, bkn ori") == "belum sampai kurang rapi bukan ori"


def test_negation_only_whole_tokens():
    # "ga" inside other words must not be replaced.
    assert clean_text("juga harga bagus") == "juga harga bagus"


# ---------------------------------------------------------------------------
# 7. Slang normalisation
# ---------------------------------------------------------------------------


def test_slang_normalisation():
    assert clean_text("brg bgt sesui") == "barang banget sesuai"
    assert clean_text("pengirimn cpt, yg jual ramah") == "pengiriman cepat yang jual ramah"
    assert normalize_token("bagus") == "bagus"  # unknown word unchanged


# ---------------------------------------------------------------------------
# 8. Repeated characters
# ---------------------------------------------------------------------------


def test_repeated_characters_collapsed_to_two():
    assert clean_text("mantaaap") == "mantaap"
    assert clean_text("mantaaapppp") == "mantaapp"
    assert clean_text("bagusss bangettt") == "baguss bangett"
    assert clean_text("keren") == "keren"            # untouched
    assert clean_text("baguss") == "baguss"          # exactly 2 stays 2


def test_repeated_characters_then_normalised():
    # Documents current behaviour: repetition collapse runs before
    # normalisation, so a stretched slang word only maps if the collapsed
    # form is itself in the dictionaries ("sesuiii" -> "sesuii", unmapped).
    assert clean_text("sesuiii") == "sesuii"
    assert clean_text("tdkkk") == "tdkk"


def test_digits_are_not_collapsed():
    assert clean_text("harga 1000 ribu") == "harga 1000 ribu"


# ---------------------------------------------------------------------------
# clean_text: URL, emoji, punctuation, whitespace, empty input
# ---------------------------------------------------------------------------


def test_removes_url_emoji_punctuation_and_tidies_spaces():
    raw = "  Barang BAGUS!!! 😍👍 cek https://lazada.co.id/x?y=1 dan www.toko.com ,ok.  "
    assert clean_text(raw) == "barang bagus cek dan oke"


def test_punctuation_between_words_becomes_space():
    assert clean_text("bagus,cepat.murah") == "bagus cepat murah"
    assert clean_text("snake_case") == "snake case"


@pytest.mark.parametrize("value", [None, float("nan"), np.nan, pd.NA, "", "   ", "!!! 😍", "https://a.b"])
def test_clean_text_empty_inputs(value):
    assert clean_text(value) == ""


# ---------------------------------------------------------------------------
# 9. Stratified, deterministic split
# ---------------------------------------------------------------------------


def _synthetic_df(n_pos=500, n_neg=300, n_neu=200) -> pd.DataFrame:
    sentiments = ["positive"] * n_pos + ["negative"] * n_neg + ["neutral"] * n_neu
    return pd.DataFrame(
        {
            "review": [f"ulasan {i}" for i in range(len(sentiments))],
            "sentiment": sentiments,
            "rating": np.where(np.arange(len(sentiments)) % 7 == 0, np.nan, 5.0),
            "item_id": np.arange(len(sentiments)) % 50,
        }
    )


def test_split_sizes_and_stratification():
    df = _synthetic_df()
    train, val, test = split_data(df, val_frac=0.1, test_frac=0.1, seed=42)
    assert (len(train), len(val), len(test)) == (800, 100, 100)
    expected = {"positive": (400, 50, 50), "negative": (240, 30, 30), "neutral": (160, 20, 20)}
    for label, (n_tr, n_va, n_te) in expected.items():
        assert (train["sentiment"] == label).sum() == n_tr
        assert (val["sentiment"] == label).sum() == n_va
        assert (test["sentiment"] == label).sum() == n_te


def test_split_no_overlap_and_complete():
    df = _synthetic_df()
    train, val, test = split_data(df)
    tr, va, te = set(train["review"]), set(val["review"]), set(test["review"])
    assert not tr & va and not tr & te and not va & te
    assert tr | va | te == set(df["review"])
    assert len(train) + len(val) + len(test) == len(df)


def test_split_is_deterministic():
    df = _synthetic_df()
    a = split_data(df, seed=42)
    b = split_data(df, seed=42)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)
    c = split_data(df, seed=7)
    assert a[1]["review"].tolist() != c[1]["review"].tolist()


def test_split_preserves_columns_and_does_not_mutate_input():
    df = _synthetic_df()
    copy = df.copy()
    train, val, test = split_data(df)
    pd.testing.assert_frame_equal(df, copy)
    for part in (train, val, test):
        assert list(part.columns) == list(df.columns)
        assert part.index.tolist() == list(range(len(part)))


def test_split_keeps_rows_with_missing_sentiment():
    df = _synthetic_df(n_pos=20, n_neg=20, n_neu=0)
    df.loc[[0, 1, 2], "sentiment"] = None
    train, val, test = split_data(df)
    assert len(train) + len(val) + len(test) == len(df)


@pytest.mark.parametrize("val_frac,test_frac", [(-0.1, 0.1), (0.5, 0.5), (1.0, 0.0)])
def test_split_rejects_invalid_fractions(val_frac, test_frac):
    with pytest.raises(ValueError):
        split_data(_synthetic_df(), val_frac=val_frac, test_frac=test_frac)


# ---------------------------------------------------------------------------
# 10. Training / inference cleaning consistency
# ---------------------------------------------------------------------------

RAW_REVIEWS = [
    "Barangnya BAGUSSS bgt, pengiriman gak lama!!! 😍",
    "brg tdk sesui deskripsi... kecewa https://bit.ly/abc",
    "Mantaaapppp, seller ramah 👍👍",
    "nggak ngga enggak kagak gk ga",
]


def test_training_and_inference_cleaning_identical(tmp_path):
    rows = [_base_row(reviewContent=r, itemId=i) for i, r in enumerate(RAW_REVIEWS)]
    df = load_reviews(_write_csv(tmp_path / "r.csv", rows))
    # Training path: CSV -> load_reviews -> clean_text on the column.
    train_clean = df["review"].map(clean_text).tolist()
    # Inference path: raw user string -> clean_text.
    infer_clean = [clean_text(r) for r in RAW_REVIEWS]
    assert train_clean == infer_clean
    assert infer_clean[3] == "tidak tidak tidak tidak tidak tidak"


@pytest.mark.parametrize("raw", RAW_REVIEWS)
def test_clean_text_is_idempotent(raw):
    once = clean_text(raw)
    assert clean_text(once) == once
