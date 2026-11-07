"""Unit tests for reviewlens.labeling (INTERFACES.md sections 5 and 14)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewlens import labeling, preprocess  # noqa: E402
from reviewlens.labeling import (  # noqa: E402
    ASPECTS,
    Aspect,
    InsightLabel,
    detect_aspects,
    label_review,
)
from reviewlens.preprocess import NEGATIONS, clean_text  # noqa: E402

ASPECT_NAMES = [
    "kualitas_produk", "fungsi_kerusakan", "pengiriman", "kemasan",
    "kesesuaian_deskripsi", "harga", "pelayanan_penjual", "keaslian", "garansi",
]
BY_NAME = {a.name: a for a in ASPECTS}
GENERAL = (
    labeling._GENERAL_POSITIVE,
    labeling._GENERAL_PAIN_POINT,
    labeling._GENERAL_SUGGESTION,
)


def label(raw: str, sentiment: str | None = None) -> InsightLabel:
    """Labeling always receives clean_text output (as in prepare_data)."""
    return label_review(clean_text(raw), sentiment)


def detect(raw: str) -> dict[str, str]:
    return detect_aspects(clean_text(raw))


# ---------------------------------------------------------------------------
# Lexicon structure
# ---------------------------------------------------------------------------


def test_nine_aspects_in_contract_order():
    assert [a.name for a in ASPECTS] == ASPECT_NAMES
    assert all(isinstance(a, Aspect) for a in ASPECTS)


def test_all_lexicon_phrases_are_unique():
    phrases = [p for a in ASPECTS for p in (a.pos_phrase, a.neg_phrase, a.suggestion)]
    phrases += list(GENERAL)
    assert len(phrases) == len(set(phrases))
    assert all(p and p == p.strip() for p in phrases)


def test_each_phrase_maps_to_exactly_one_aspect_and_polarity():
    index: dict[str, set[tuple[str, str]]] = {}
    for a in ASPECTS:
        for polarity, phrase in (("pos", a.pos_phrase), ("neg", a.neg_phrase),
                                 ("sug", a.suggestion)):
            index.setdefault(phrase, set()).add((a.name, polarity))
    assert all(len(owners) == 1 for owners in index.values())


def test_lexicon_tokens_survive_clean_text():
    # A lexicon word that clean_text rewrites could never match.
    for a in ASPECTS:
        for word in a.keywords + a.pos_cues + a.neg_cues:
            assert clean_text(word) == word, (a.name, word)
            assert " " not in word


def test_lexicon_has_no_negation_words_and_no_conflicting_cues():
    for a in ASPECTS:
        words = set(a.keywords) | set(a.pos_cues) | set(a.neg_cues)
        assert not words & set(NEGATIONS), a.name
        pos = {labeling._key(c) for c in a.pos_cues}
        neg = {labeling._key(c) for c in a.neg_cues}
        assert not pos & neg, a.name


def test_negations_come_from_preprocess():
    assert labeling.NEGATIONS is preprocess.NEGATIONS
    source = inspect.getsource(labeling)
    assert "NEGATIONS =" not in source
    assert "NEGATIONS: " not in source


# ---------------------------------------------------------------------------
# All 9 aspects: keyword + positive cue / keyword + negative cue
# ---------------------------------------------------------------------------

POSITIVE_CASES = {
    "kualitas_produk": "kualitas barang bagus",
    "fungsi_kerusakan": "tv berfungsi normal",
    "pengiriman": "pengiriman cepat",
    "kemasan": "packing rapi",
    "kesesuaian_deskripsi": "sesuai deskripsi",
    "harga": "harga murah",
    "pelayanan_penjual": "seller ramah",
    "keaslian": "flashdisk original",
    "garansi": "garansi resmi",
}
NEGATIVE_CASES = {
    "kualitas_produk": "kualitas jelek",
    "fungsi_kerusakan": "tv mati total",
    "pengiriman": "pengiriman lama",
    "kemasan": "kardus penyok",
    "kesesuaian_deskripsi": "warna berbeda dengan foto",
    "harga": "harganya mahal",
    "pelayanan_penjual": "penjual judes",
    "keaslian": "barang kw",
    "garansi": "klaim garansi ditolak",
}


@pytest.mark.parametrize("name", ASPECT_NAMES)
def test_positive_cue_per_aspect(name):
    raw = POSITIVE_CASES[name]
    assert detect(raw).get(name) == "pos", detect(raw)
    lab = label(raw, None)
    assert BY_NAME[name].pos_phrase in lab.positives
    assert BY_NAME[name].neg_phrase not in lab.pain_points


@pytest.mark.parametrize("name", ASPECT_NAMES)
def test_negative_cue_per_aspect(name):
    raw = NEGATIVE_CASES[name]
    assert detect(raw).get(name) == "neg", detect(raw)
    lab = label(raw, None)
    assert BY_NAME[name].neg_phrase in lab.pain_points
    assert BY_NAME[name].suggestion in lab.suggestions


def test_cue_overrides_weak_sentiment():
    # Explicit cue wins over the (possibly wrong) weak sentiment label.
    assert label("pengiriman cepat", "negative").positives == ["pengiriman cepat"]
    assert label("pengiriman lama", "positive").pain_points == ["pengiriman lambat"]


def test_suffix_nya_and_stretched_words_match():
    assert detect("pengirimannya cepat")["pengiriman"] == "pos"
    assert detect("barangnya mantaaap")["kualitas_produk"] == "pos"


def test_mixed_review_from_contract_example():
    lab = label("barang bagus tapi pengiriman lama sekali", "neutral")
    assert lab.positives == ["kualitas produk bagus"]
    assert lab.pain_points == ["pengiriman lambat"]
    assert lab.suggestions == ["percepat proses pengiriman"]


def test_cue_goes_to_nearest_aspect_only():
    # "lambat" belongs to the seller (respon), not to delivery.
    assert detect("respon seller lambat") == {"pelayanan_penjual": "neg"}
    # "rusak" right after "kemasan" is packaging damage, not product damage.
    assert detect("kemasan rusak") == {"kemasan": "neg"}


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------


def test_tidak_cepat_is_negative():
    assert detect_aspects("tidak cepat") == {"pengiriman": "neg"}
    assert detect_aspects("pengiriman tidak cepat") == {"pengiriman": "neg"}
    lab = label_review("pengiriman tidak cepat", "positive")
    assert lab.pain_points == ["pengiriman lambat"]
    assert lab.suggestions == ["percepat proses pengiriman"]


@pytest.mark.parametrize("variant", ["gak", "ga", "nggak", "ngga", "gk", "tdk", "enggak", "kagak"])
def test_gak_cepat_after_normalisation(variant):
    review = clean_text(f"{variant} cepat")
    assert review == "tidak cepat"
    assert label_review(review, None).pain_points == ["pengiriman lambat"]


def test_negation_window_is_two_words():
    assert detect_aspects("pengiriman tidak terlalu cepat")["pengiriman"] == "neg"
    # 3 words before the cue -> outside the window, cue stays positive.
    assert detect_aspects("tidak pengiriman sangat sangat cepat")["pengiriman"] == "pos"


def test_negation_does_not_cross_clause_boundary():
    assert detect_aspects("tidak mahal tapi pengiriman cepat") == {
        "harga": "pos", "pengiriman": "pos",
    }


@pytest.mark.parametrize("word", ["kurang", "belum", "bukan"])
def test_other_canonical_negations(word):
    assert detect_aspects(f"kualitas {word} bagus")["kualitas_produk"] == "neg"


def test_bukan_ori_after_normalisation():
    assert detect("bkn ori") == {"keaslian": "neg"}


# ---------------------------------------------------------------------------
# "tidak rusak" must not become a damage complaint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sentiment", ["positive", "negative", "neutral", None])
@pytest.mark.parametrize(
    "raw",
    ["barang tidak rusak", "alhamdulillah tidak rusak", "tidak ada kerusakan",
     "kemasan aman tidak rusak", "gak rusak sama sekali"],
)
def test_tidak_rusak_is_not_damage(raw, sentiment):
    lab = label(raw, sentiment)
    assert "produk rusak atau cacat" not in lab.pain_points
    assert "kemasan rusak atau kurang aman" not in lab.pain_points
    assert detect(raw).get("fungsi_kerusakan") != "neg"


def test_tidak_rusak_is_positive_function():
    assert detect("alhamdulillah tidak rusak") == {"fungsi_kerusakan": "pos"}
    assert detect("kemasan aman tidak rusak") == {"kemasan": "pos"}


# ---------------------------------------------------------------------------
# Sentiment fallback
# ---------------------------------------------------------------------------


def test_aspect_without_cue_uses_sentiment():
    assert label("pengiriman via jne", "positive").positives == ["pengiriman cepat"]
    neg = label("pengiriman via jne", "negative")
    assert neg.pain_points == ["pengiriman lambat"]
    assert neg.suggestions == ["percepat proses pengiriman"]
    assert label("pengiriman via jne", "Negative ").pain_points == ["pengiriman lambat"]


@pytest.mark.parametrize("sentiment", ["neutral", None, "", "unknown"])
def test_neutral_or_none_without_cue_is_ignored(sentiment):
    lab = label("pengiriman via jne", sentiment)
    assert lab == InsightLabel([], [], [])
    assert detect_aspects("pengiriman via jne") == {}


def test_neutral_keeps_cue_based_aspects_only():
    lab = label("barang sudah diterima pengiriman cepat", "neutral")
    assert lab.positives == ["pengiriman cepat"]  # "barang" has no cue -> dropped
    assert lab.pain_points == []


# ---------------------------------------------------------------------------
# No aspect
# ---------------------------------------------------------------------------


def test_positive_without_aspect():
    lab = label("terima kasih mantap", "positive")
    assert lab == InsightLabel(["produk memuaskan secara umum"], [], [])


def test_negative_without_aspect():
    lab = label("kecewa sekali", "negative")
    assert lab == InsightLabel(
        [], ["pelanggan kurang puas secara umum"],
        ["tingkatkan kualitas layanan secara umum"],
    )


@pytest.mark.parametrize("sentiment", ["neutral", None])
def test_neutral_without_aspect_is_empty(sentiment):
    assert label("biasa saja", sentiment) == InsightLabel([], [], [])


@pytest.mark.parametrize("review", ["", "   "])
def test_empty_review(review):
    assert detect_aspects(review) == {}
    assert label_review(review, "positive").positives == ["produk memuaskan secara umum"]


# ---------------------------------------------------------------------------
# Limits, pairing, uniqueness, determinism
# ---------------------------------------------------------------------------

MANY_NEG = ("kualitas jelek tapi tv mati tapi pengiriman lama tapi kardus penyok "
            "tapi harga mahal tapi penjual judes")
MANY_POS = ("kualitas bagus dan tv berfungsi normal dan pengiriman cepat dan "
            "packing rapi dan harga murah")


def test_max_three_items_per_section():
    neg = label(MANY_NEG, "negative")
    pos = label(MANY_POS, "positive")
    assert len(detect(MANY_NEG)) == 6 and len(detect(MANY_POS)) == 5
    assert len(neg.pain_points) == 3 and len(neg.suggestions) == 3
    assert len(pos.positives) == 3


def test_cue_based_aspects_win_the_three_slots():
    # "barang" has no cue (sentiment fallback); the 3 explicit cues must be kept.
    lab = label_review("barang ori garansi resmi respon seller cepat", "positive")
    assert lab.positives == [
        "produk asli dan original", "garansi resmi dan jelas", "pelayanan penjual ramah",
    ]


def test_items_follow_order_of_mention():
    assert label(MANY_NEG, "negative").pain_points == [
        "kualitas produk buruk", "produk rusak atau cacat", "pengiriman lambat",
    ]
    assert list(detect("pengiriman cepat barang bagus")) == ["pengiriman", "kualitas_produk"]


@pytest.mark.parametrize(
    "raw,sentiment",
    [(MANY_NEG, "negative"), (MANY_POS, "positive"),
     ("barang bagus tapi pengiriman lama", "neutral"),
     ("kecewa", "negative"), ("garansi ditolak seller cuek", None)],
)
def test_pain_points_and_suggestions_are_paired(raw, sentiment):
    lab = label(raw, sentiment)
    assert len(lab.pain_points) == len(lab.suggestions)
    neg_to_sug = {a.neg_phrase: a.suggestion for a in ASPECTS}
    neg_to_sug[labeling._GENERAL_PAIN_POINT] = labeling._GENERAL_SUGGESTION
    for pain, sug in zip(lab.pain_points, lab.suggestions):
        assert neg_to_sug[pain] == sug


@pytest.mark.parametrize("raw", [MANY_NEG, MANY_POS, "barang bagus barang bagus produk mantap"])
def test_no_duplicate_phrases(raw):
    for sentiment in ("positive", "negative"):
        lab = label(raw, sentiment)
        all_items = lab.positives + lab.pain_points + lab.suggestions
        assert len(all_items) == len(set(all_items))


def test_deterministic():
    reviews = [MANY_NEG, MANY_POS, "barang bagus tapi pengiriman lama", "tidak rusak", "kecewa"]
    first = [label(r, "negative") for r in reviews]
    for _ in range(3):
        assert [label(r, "negative") for r in reviews] == first
    assert [label(r, "negative") for r in reversed(reviews)] == list(reversed(first))
