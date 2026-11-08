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
    split_data_with_stats,
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
# 9b. Leakage-free split: group by clean_text(review), drop conflict groups
# ---------------------------------------------------------------------------


def _leaky_df() -> pd.DataFrame:
    """Synthetic data shaped like the real problem.

    - 600 unique reviews (positive/negative/neutral);
    - generic texts repeated across many items, with variants that only become
      identical after clean_text ("Mantap!!", "MANTAAAP", "mantap");
    - a few identical texts with different sentiment labels (conflicts).
    """
    rows = []
    sentiments = ["positive"] * 360 + ["negative"] * 150 + ["neutral"] * 90
    for i, s in enumerate(sentiments):
        rows.append((f"ulasan unik nomor {i}", s))
    # Note: "MANTAAAP" would clean to "mantaap" (3+ letters -> 2), a different group.
    for variant in ["mantap", "Mantap!!", "MANTAP", "mantap 👍"] * 30:     # 120 rows, 1 group
        rows.append((variant, "positive"))
    for variant in ["barang bagus", "brg bagus", "Barang BAGUS."] * 10:   # 30 rows, 1 group
        rows.append((variant, "positive"))
    for variant in ["gak nyala", "ga nyala!", "tdk nyala"] * 8:            # 24 rows, 1 group
        rows.append((variant, "negative"))
    # Conflict groups: same cleaned text, more than one sentiment.
    rows += [("lumayan", "positive"), ("Lumayan.", "neutral"), ("lumayan", "positive")]
    rows += [("biasa aja", "neutral"), ("biasa saja", "negative")]          # aja -> saja
    rows += [("ok", "positive"), ("oke", "positive"), ("OKE!", "neutral")]  # ok -> oke
    df = pd.DataFrame(rows, columns=["review", "sentiment"])
    df["item_id"] = np.arange(len(df))          # unique row id for the checks below
    df["rating"] = 5.0
    return df


CONFLICT_KEYS = {"lumayan", "biasa saja", "oke"}


def _keys(part: pd.DataFrame) -> set[str]:
    return set(part["review"].map(clean_text))


def test_no_clean_text_in_more_than_one_split():
    train, val, test, _ = split_data_with_stats(_leaky_df())
    k_tr, k_va, k_te = _keys(train), _keys(val), _keys(test)
    assert not k_tr & k_va and not k_tr & k_te and not k_va & k_te


def test_variants_identical_after_cleaning_form_one_group():
    train, val, test, stats = split_data_with_stats(_leaky_df())
    for key, n_rows in (("mantap", 120), ("barang bagus", 30), ("tidak nyala", 24)):
        holders = [p for p in (train, val, test) if key in _keys(p)]
        assert len(holders) == 1, key
        assert (holders[0]["review"].map(clean_text) == key).sum() == n_rows
    # 600 unique + 3 repeated groups + 3 conflict groups
    assert stats["n_groups"] == 606


def test_conflict_groups_are_dropped_entirely():
    df = _leaky_df()
    train, val, test, stats = split_data_with_stats(df)
    kept = pd.concat([train, val, test])
    assert not _keys(kept) & CONFLICT_KEYS
    conflict_rows = df["review"].map(clean_text).isin(CONFLICT_KEYS).sum()
    assert conflict_rows == 8
    assert stats["n_conflict_groups"] == 3
    assert stats["n_rows_dropped"] == 8
    assert stats["n_rows_kept"] == len(df) - 8 == len(kept)
    assert set(stats["conflict_examples"]) == CONFLICT_KEYS


def test_same_sentiment_duplicates_are_all_kept():
    train, val, test, _ = split_data_with_stats(_leaky_df())
    kept = pd.concat([train, val, test])
    assert (kept["review"].map(clean_text) == "mantap").sum() == 120
    # original (uncleaned) variants are preserved as-is
    assert set(kept["review"]) >= {"mantap", "Mantap!!", "MANTAP", "mantap 👍"}


def test_labels_and_values_unchanged():
    df = _leaky_df()
    original = df.set_index("item_id")
    train, val, test, _ = split_data_with_stats(df)
    kept = pd.concat([train, val, test]).set_index("item_id")
    pd.testing.assert_frame_equal(kept.sort_index(), original.loc[kept.index].sort_index())


def test_non_conflict_rows_appear_exactly_once():
    df = _leaky_df()
    train, val, test, stats = split_data_with_stats(df)
    ids = pd.concat([train, val, test])["item_id"].tolist()
    assert len(ids) == len(set(ids))
    expected = set(df.loc[~df["review"].map(clean_text).isin(CONFLICT_KEYS), "item_id"])
    assert set(ids) == expected
    assert stats["split_rows"] == {"train": len(train), "val": len(val), "test": len(test)}
    assert sum(stats["split_groups"].values()) == stats["n_groups"] - stats["n_conflict_groups"]


def test_group_split_is_deterministic():
    df = _leaky_df()
    a = split_data_with_stats(df, seed=42)
    b = split_data_with_stats(df, seed=42)
    for x, y in zip(a[:3], b[:3]):
        pd.testing.assert_frame_equal(x, y)
    assert a[3] == b[3]
    # Row order of the input does not change which split a row ends up in.
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    c = split_data_with_stats(shuffled, seed=42)
    for x, y in zip(a[:3], c[:3]):
        assert set(x["item_id"]) == set(y["item_id"])


def test_proportions_close_to_80_10_10_in_rows():
    _, _, _, stats = split_data_with_stats(_leaky_df())
    frac = stats["split_frac"]
    assert abs(frac["train"] - 0.8) < 0.02
    assert abs(frac["val"] - 0.1) < 0.02
    assert abs(frac["test"] - 0.1) < 0.02


def test_large_group_handled_as_one_unit():
    # One group holds 40% of the positive stratum.
    rows = [("mantap", "positive")] * 400 + [(f"unik {i}", "positive") for i in range(600)]
    rows += [(f"jelek {i}", "negative") for i in range(200)]
    df = pd.DataFrame(rows, columns=["review", "sentiment"])
    train, val, test, stats = split_data_with_stats(df)
    holders = [n for n, p in zip(("train", "val", "test"), (train, val, test))
               if "mantap" in set(p["review"])]
    assert holders == ["train"]                      # largest group -> furthest below target
    assert (train["review"] == "mantap").sum() == 400
    assert stats["split_rows"] == {"train": 960, "val": 120, "test": 120}
    assert (val["sentiment"] == "positive").sum() == 100
    assert (test["sentiment"] == "negative").sum() == 20


def test_group_larger_than_val_and_test_targets():
    # Every row identical: the only group cannot be split, so it all goes to train.
    df = pd.DataFrame({"review": ["Mantap!"] * 50, "sentiment": ["positive"] * 50})
    train, val, test, stats = split_data_with_stats(df)
    assert (len(train), len(val), len(test)) == (50, 0, 0)
    assert list(val.columns) == list(test.columns) == ["review", "sentiment"]
    assert stats["n_conflict_groups"] == 0


def test_split_data_matches_split_data_with_stats():
    df = _leaky_df()
    plain = split_data(df)
    full = split_data_with_stats(df)
    for x, y in zip(plain, full[:3]):
        pd.testing.assert_frame_equal(x, y)


def test_split_requires_review_column():
    with pytest.raises(ValueError, match="review"):
        split_data(pd.DataFrame({"sentiment": ["positive"]}))


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
