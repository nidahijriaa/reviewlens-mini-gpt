"""Unit tests for reviewlens.dataset (INTERFACES.md sections 2, 7 and 14).

tokenizer.py is developed on another branch (feat/tokenizer) and is not
available here yet. ``FakeWordTokenizer`` below is a TEST-ONLY stand-in that
mimics just the contract behaviour dataset.py relies on (INTERFACES.md 2.1, 6):
whitespace split, special tokens with fixed IDs 0-9, unknown words -> <unk>.
It is not a replacement implementation and is never imported by src/.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewlens import dataset as ds  # noqa: E402
from reviewlens.config import IGNORE_INDEX, ModelConfig  # noqa: E402
from reviewlens.dataset import (  # noqa: E402
    MAX_NEW_TOKENS,
    MAX_REVIEW_TOKENS,
    MAX_SEQ_TOKENS,
    MAX_TARGET_TOKENS,
    PREFIX_TOKENS,
    ReviewInsightDataset,
    collate_fn,
    encode_example,
    format_prompt,
    format_target,
)
from reviewlens.labeling import InsightLabel, label_review  # noqa: E402
from reviewlens.model import MiniGPT  # noqa: E402
from reviewlens.preprocess import clean_text, split_data  # noqa: E402

BLOCK = 128

# ---------------------------------------------------------------------------
# Test-only tokenizer (contract behaviour only)
# ---------------------------------------------------------------------------
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<review>",
            "<pos>", "<neg>", "<sug>", "<sep>", "<none>"]
PAD, UNK, BOS, EOS, REVIEW, POS, NEG, SUG, SEP, NONE = range(10)


class FakeWordTokenizer:
    def __init__(self, words=()):
        self.stoi = {tok: i for i, tok in enumerate(SPECIALS)}
        for w in words:
            self.stoi.setdefault(w, len(self.stoi))
        self.itos = {i: t for t, i in self.stoi.items()}

    @classmethod
    def from_texts(cls, texts):
        return cls(w for t in texts for w in t.split())

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(w, UNK) for w in text.split()]

    def decode(self, ids) -> str:
        return " ".join(self.itos[int(i)] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


def lbl(pos=(), neg=(), sug=()) -> InsightLabel:
    return InsightLabel(list(pos), list(neg), list(sug))


def make_tok(*texts) -> FakeWordTokenizer:
    return FakeWordTokenizer.from_texts(texts)


def write_jsonl(path: Path, records) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")
    return path


def record(review, pos=(), neg=(), sug=(), sentiment="positive"):
    return {"review": review, "sentiment": sentiment, "rating": 5.0, "item_id": 1,
            "positives": list(pos), "pain_points": list(neg), "suggestions": list(sug)}


# ---------------------------------------------------------------------------
# Length constants (INTERFACES.md 2.2)
# ---------------------------------------------------------------------------
def test_length_constants_match_contract():
    assert ModelConfig(vocab_size=10).block_size == BLOCK
    assert (PREFIX_TOKENS, MAX_TARGET_TOKENS) == (2, 48)
    assert MAX_SEQ_TOKENS == BLOCK + 1 == 129
    assert MAX_REVIEW_TOKENS == BLOCK + 1 - 2 - 48 == 79
    assert MAX_NEW_TOKENS == 47
    # Longest inference input: prompt (2 + 79 + 1) + MAX_NEW_TOKENS - 1 <= block_size
    assert PREFIX_TOKENS + MAX_REVIEW_TOKENS + 1 + MAX_NEW_TOKENS - 1 <= BLOCK


# ---------------------------------------------------------------------------
# format_prompt / format_target
# ---------------------------------------------------------------------------
def test_format_prompt():
    assert format_prompt("barang bagus") == "<bos> <review> barang bagus <pos>"


def test_format_prompt_truncates_review_to_79_words():
    words = [f"w{i}" for i in range(300)]
    out = format_prompt(" ".join(words)).split()
    assert out[:2] == ["<bos>", "<review>"] and out[-1] == "<pos>"
    assert out[2:-1] == words[:MAX_REVIEW_TOKENS]


@pytest.mark.parametrize("bad", ["", "   ", "barang <pos> bagus"])
def test_format_prompt_rejects_invalid(bad):
    with pytest.raises(ValueError):
        format_prompt(bad)


def test_format_prompt_rejects_non_string():
    with pytest.raises(TypeError):
        format_prompt(None)  # type: ignore[arg-type]


def test_format_target_canonical():
    label = lbl(["kualitas produk bagus", "harga murah"], ["pengiriman lambat"],
                ["percepat proses pengiriman"])
    assert format_target(label) == (
        "<pos> kualitas produk bagus <sep> harga murah "
        "<neg> pengiriman lambat <sug> percepat proses pengiriman <eos>"
    )


def test_format_target_empty_sections_use_none():
    assert format_target(lbl(["harga murah"])) == (
        "<pos> harga murah <neg> <none> <sug> <none> <eos>"
    )
    assert format_target(lbl()) == "<pos> <none> <neg> <none> <sug> <none> <eos>"


@pytest.mark.parametrize("label", [lbl([""]), lbl(neg=["  "]), lbl(sug=["a <eos>"]), lbl([3])])
def test_format_target_rejects_invalid_phrases(label):
    with pytest.raises(ValueError):
        format_target(label)


def test_format_target_rejects_non_list_section():
    with pytest.raises(TypeError):
        format_target(InsightLabel("harga murah", [], []))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# encode_example
# ---------------------------------------------------------------------------
def test_encode_example_matches_canonical_string():
    label = lbl(["harga murah"], ["pengiriman lambat"], ["percepat proses pengiriman"])
    review = "barang murah tapi pengiriman lama"
    tok = make_tok(review, format_target(label))
    ids = encode_example(review, label, tok, BLOCK)
    assert ids == tok.encode(f"<bos> <review> {review} {format_target(label)}")
    assert ids[0] == BOS and ids[1] == REVIEW and ids[-1] == EOS


def test_long_review_truncated_target_intact():
    review = " ".join(f"kata{i}" for i in range(300))
    label = lbl(["harga murah"], ["pengiriman lambat"], ["percepat proses pengiriman"])
    tok = make_tok(review, format_target(label))
    ids = encode_example(review, label, tok, BLOCK)

    target_ids = tok.encode(format_target(label))
    assert len(ids) <= BLOCK + 1
    assert ids[-len(target_ids):] == target_ids               # target untouched
    assert ids[2:-len(target_ids)] == tok.encode(review)[:MAX_REVIEW_TOKENS]


def test_long_target_drops_items_in_order_and_keeps_markers():
    long = lambda tag: " ".join(f"{tag}{i}" for i in range(6))  # noqa: E731  6 words/phrase
    label = lbl([long("p"), long("q"), long("r")],
                [long("n"), long("m"), long("o")],
                [long("s"), long("t"), long("u")])
    tok = make_tok(format_target(label))
    ids = encode_example("barang bagus", label, tok, BLOCK)
    text = tok.decode(ids)
    target = text[text.index("<pos>"):]

    assert len(tok.encode(target)) <= MAX_TARGET_TOKENS
    for marker in ("<pos>", "<neg>", "<sug>", "<eos>"):
        assert target.split().count(marker) == 1
    assert target.endswith("<eos>")
    # 3*3 phrases of 6 words + 6 <sep> + 4 markers = 64 > 48 -> 3 items dropped,
    # one per pass: last suggestion, last pain point, last positive.
    assert long("u") not in target and long("o") not in target and long("r") not in target
    assert long("t") in target and long("m") in target and long("q") in target
    # The original label object is not mutated.
    assert len(label.suggestions) == 3


def test_encode_example_small_block_size():
    review = " ".join(f"k{i}" for i in range(50))
    label = lbl(["harga murah"])
    tok = make_tok(review, format_target(label))
    ids = encode_example(review, label, tok, block_size=16)
    assert len(ids) == 17
    assert ids[-1] == EOS


@pytest.mark.parametrize("block_size", [0, -5, 4])
def test_encode_example_rejects_bad_block_size(block_size):
    tok = make_tok("barang bagus")
    with pytest.raises(ValueError):
        encode_example("barang bagus", lbl(), tok, block_size)


def test_encode_example_rejects_empty_review():
    with pytest.raises(ValueError):
        encode_example("  ", lbl(), make_tok(), BLOCK)


# ---------------------------------------------------------------------------
# ReviewInsightDataset
# ---------------------------------------------------------------------------
def test_dataset_items(tmp_path):
    recs = [record("barang bagus", ["kualitas produk bagus"]),
            record("pengiriman lama sekali", neg=["pengiriman lambat"],
                   sug=["percepat proses pengiriman"], sentiment="negative")]
    path = write_jsonl(tmp_path / "train.jsonl", recs)
    tok = make_tok(*(r["review"] for r in recs), *(format_target(lbl(r["positives"],
                   r["pain_points"], r["suggestions"])) for r in recs))
    data = ReviewInsightDataset(path, tok, BLOCK)

    assert len(data) == 2
    for i, rec in enumerate(recs):
        item = data[i]
        assert isinstance(item, torch.Tensor) and item.dtype == torch.long and item.dim() == 1
        assert item.numel() <= BLOCK + 1
        label = lbl(rec["positives"], rec["pain_points"], rec["suggestions"])
        assert item.tolist() == encode_example(rec["review"], label, tok, BLOCK)


def test_dataset_skips_blank_lines(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n" + json.dumps(record("barang bagus")) + "\n\n", encoding="utf-8")
    assert len(ReviewInsightDataset(path, make_tok("barang bagus"), BLOCK)) == 1


@pytest.mark.parametrize("content", ["", "\n\n  \n"])
def test_dataset_rejects_empty_file(tmp_path, content):
    path = tmp_path / "empty.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="no examples"):
        ReviewInsightDataset(path, make_tok(), BLOCK)


def test_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReviewInsightDataset(tmp_path / "nope.jsonl", make_tok(), BLOCK)


@pytest.mark.parametrize(
    "line, match",
    [
        ("{not json", "invalid JSON"),
        ("[1, 2]", "JSON object"),
        (json.dumps({"review": "barang bagus", "positives": []}), "missing keys"),
        (json.dumps(record("")), "empty"),
        (json.dumps(record("barang bagus", pos=[""])), "empty or non-string"),
    ],
)
def test_dataset_rejects_invalid_records(tmp_path, line, match):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record("ok")) + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        ReviewInsightDataset(path, make_tok("ok"), BLOCK)


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------
def seqs():
    return [
        torch.tensor([BOS, REVIEW, 20, 21, POS, NONE, NEG, NONE, SUG, NONE, EOS]),  # len 11
        torch.tensor([BOS, REVIEW, 22, POS, NONE, NEG, NONE, SUG, NONE, EOS]),      # len 10
        torch.tensor([BOS, REVIEW, 23, 24, 25, 26, POS, NONE, NEG, NONE, SUG, NONE, EOS]),
    ]


def test_collate_shapes_and_dtypes():
    x, y = collate_fn(seqs())
    assert x.shape == y.shape == (3, 12)
    assert x.dtype == y.dtype == torch.long


def test_collate_shift():
    batch = seqs()
    x, y = collate_fn(batch)
    for i, s in enumerate(batch):
        n = s.numel() - 1
        assert torch.equal(x[i, :n], s[:-1])
        assert torch.equal(y[i, :n], s[1:])


def test_collate_pads_x_with_pad_and_y_with_ignore_index():
    batch = seqs()
    x, y = collate_fn(batch)
    for i, s in enumerate(batch):
        n = s.numel() - 1
        assert torch.all(x[i, n:] == PAD)
        assert torch.all(y[i, n:] == IGNORE_INDEX)
    # PAD (0) is never used as a target: y has only real IDs or -100.
    assert not torch.any(y == PAD)
    assert int((y == IGNORE_INDEX).sum()) == (12 - 10) + (12 - 9)


def test_collate_keeps_real_eos_as_target():
    batch = seqs()
    x, y = collate_fn(batch)
    for i, s in enumerate(batch):
        n = s.numel() - 1
        assert y[i, n - 1] == EOS          # last real target is <eos>, not ignored
    assert int((y == EOS).sum()) == len(batch)


def test_collate_equal_lengths_no_padding():
    s = torch.tensor([BOS, REVIEW, 20, POS, NONE, NEG, NONE, SUG, NONE, EOS])
    x, y = collate_fn([s, s.clone()])
    assert x.shape == (2, 9)
    assert not torch.any(y == IGNORE_INDEX)


@pytest.mark.parametrize(
    "batch",
    [
        [],
        [torch.tensor([BOS])],                                   # too short
        [torch.tensor([[BOS, EOS]])],                            # 2-D
        [torch.tensor([BOS, EOS], dtype=torch.int32)],           # wrong dtype
        [[BOS, EOS]],                                            # not a tensor
    ],
)
def test_collate_rejects_invalid(batch):
    with pytest.raises(ValueError):
        collate_fn(batch)


# ---------------------------------------------------------------------------
# Compatibility with MiniGPT
# ---------------------------------------------------------------------------
def _tiny_dataset(tmp_path):
    recs = [
        record("barang bagus sekali", ["kualitas produk bagus"]),
        record("pengiriman lama", neg=["pengiriman lambat"], sug=["percepat proses pengiriman"],
               sentiment="negative"),
        record("harga murah kualitas oke banget mantap", ["harga murah", "kualitas produk bagus"]),
    ]
    path = write_jsonl(tmp_path / "train.jsonl", recs)
    tok = make_tok(*(r["review"] for r in recs),
                   *(format_target(lbl(r["positives"], r["pain_points"], r["suggestions"]))
                     for r in recs))
    return ReviewInsightDataset(path, tok, 32), tok


def test_batches_work_with_minigpt_and_ignore_padding(tmp_path):
    data, tok = _tiny_dataset(tmp_path)
    loader = DataLoader(data, batch_size=3, shuffle=False, collate_fn=collate_fn)
    x, y = next(iter(loader))
    assert x.shape[1] <= 32

    torch.manual_seed(0)
    model = MiniGPT(ModelConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=1,
                                n_head=2, d_model=16, d_ff=32, dropout=0.0)).eval()
    logits, loss = model(x, y)
    assert torch.isfinite(loss)

    keep = y != IGNORE_INDEX
    manual = F.cross_entropy(logits[keep], y[keep])
    torch.testing.assert_close(loss, manual)
    # Number of counted targets = sum of (len - 1) over real sequences, incl. <eos>.
    assert int(keep.sum()) == sum(s.numel() - 1 for s in data.examples)


def test_eos_target_contributes_to_loss(tmp_path):
    data, tok = _tiny_dataset(tmp_path)
    x, y = collate_fn(list(data.examples))
    torch.manual_seed(0)
    model = MiniGPT(ModelConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=1,
                                n_head=2, d_model=16, d_ff=32, dropout=0.0)).eval()
    _, loss = model(x, y)
    y_no_eos = y.masked_fill(y == EOS, IGNORE_INDEX)
    _, loss_no_eos = model(x, y_no_eos)
    assert not torch.isclose(loss, loss_no_eos)


# ---------------------------------------------------------------------------
# Train / val / test splits (preprocess.split_data, 80/10/10, seed 42)
# ---------------------------------------------------------------------------
def _reviews_df(n=300) -> pd.DataFrame:
    sentiments = ["positive"] * 200 + ["negative"] * 70 + ["neutral"] * 30
    adjectives = ["bagus", "jelek", "lambat", "cepat", "murah", "mahal"]
    rows = []
    for i in range(n):
        review = clean_text(f"Barang {adjectives[i % 6]}!! pengiriman ke-{i} OK")
        rows.append({"review": review, "sentiment": sentiments[i], "rating": 5.0,
                     "item_id": 1000 + i})
    return pd.DataFrame(rows)


def _to_records(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.itertuples(index=False):
        label = label_review(r.review, r.sentiment)
        out.append(record(r.review, label.positives, label.pain_points, label.suggestions,
                          sentiment=r.sentiment))
    return out


def test_split_80_10_10_deterministic_and_stratified():
    df = _reviews_df()
    train, val, test = split_data(df)                       # defaults: 0.1 / 0.1 / seed 42
    assert (len(train), len(val), len(test)) == (240, 30, 30)
    again = split_data(df, val_frac=0.1, test_frac=0.1, seed=42)
    for a, b in zip((train, val, test), again):
        pd.testing.assert_frame_equal(a, b)
    for part, n in ((train, 240), (val, 30), (test, 30)):
        frac = part["sentiment"].value_counts(normalize=True)
        assert math.isclose(frac["positive"], 200 / 300, abs_tol=0.02)
        assert len(part) == n


def test_splits_have_no_leakage_and_build_datasets(tmp_path):
    train, val, test = split_data(_reviews_df())
    paths = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        paths[name] = write_jsonl(tmp_path / f"{name}.jsonl", _to_records(part))

    # No row appears in two splits, and together they cover everything.
    keys = {n: set(zip(p["item_id"], p["review"])) for n, p in
            (("train", train), ("val", val), ("test", test))}
    assert not keys["train"] & keys["val"]
    assert not keys["train"] & keys["test"]
    assert not keys["val"] & keys["test"]
    assert len(keys["train"] | keys["val"] | keys["test"]) == 300

    # Vocabulary built from the TRAIN split only (INTERFACES.md 6) ...
    train_recs = _to_records(train)
    tok = make_tok(*(r["review"] for r in train_recs),
                   *(format_target(lbl(r["positives"], r["pain_points"], r["suggestions"]))
                     for r in train_recs))
    # ... so words that occur only in val/test are <unk> (no vocab leakage).
    val_only = {w for r in val["review"] for w in r.split()} - \
               {w for r in train["review"] for w in r.split()}
    assert val_only and all(tok.encode(w) == [UNK] for w in val_only)

    sizes = {}
    for name, path in paths.items():
        data = ReviewInsightDataset(path, tok, BLOCK)
        sizes[name] = len(data)
        x, y = collate_fn([data[i] for i in range(min(8, len(data)))])
        assert x.shape == y.shape and x.shape[1] <= BLOCK
    assert sizes == {"train": 240, "val": 30, "test": 30}


# ---------------------------------------------------------------------------
# Consistency with the real tokenizer (runs once tokenizer.py is merged)
# ---------------------------------------------------------------------------
def test_special_tokens_agree_with_tokenizer():
    tokenizer_mod = pytest.importorskip("reviewlens.tokenizer")
    tok = tokenizer_mod.WordTokenizer.build(["barang bagus"], min_freq=1)
    assert tok.encode(" ".join(SPECIALS)) == list(range(10))
    assert tok.encode("<pad>") == [ds._PAD_ID]
    ids = encode_example("barang bagus", lbl(["harga murah"]), tok, BLOCK)
    assert ids[:2] == [BOS, REVIEW] and ids[-1] == EOS
