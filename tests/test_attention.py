"""Unit tests for reviewlens.attention (INTERFACES.md sections 8 and 14)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

# Make `src/` importable without installing the package.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from reviewlens.attention import MultiHeadSelfAttention  # noqa: E402

D_MODEL, N_HEAD, BLOCK, DROPOUT = 256, 4, 128, 0.1


def make_attn(dropout: float = DROPOUT, seed: int = 0) -> MultiHeadSelfAttention:
    torch.manual_seed(seed)
    return MultiHeadSelfAttention(D_MODEL, N_HEAD, BLOCK, dropout)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------
def test_shape():
    attn = make_attn().eval()
    x = torch.randn(2, 16, D_MODEL)
    assert attn(x).shape == (2, 16, D_MODEL)


@pytest.mark.parametrize("batch_size", [1, 2, 5, 64])
def test_various_batch_sizes(batch_size):
    attn = make_attn().eval()
    x = torch.randn(batch_size, 10, D_MODEL)
    assert attn(x).shape == (batch_size, 10, D_MODEL)


@pytest.mark.parametrize("seq_len", [1, 2, 7, 64, BLOCK])
def test_various_sequence_lengths(seq_len):
    attn = make_attn().eval()
    x = torch.randn(3, seq_len, D_MODEL)
    assert attn(x).shape == (3, seq_len, D_MODEL)


def test_rejects_sequence_longer_than_block_size():
    attn = make_attn()
    with pytest.raises(ValueError, match="block_size"):
        attn(torch.randn(1, BLOCK + 1, D_MODEL))


def test_rejects_wrong_embedding_dim():
    attn = make_attn()
    with pytest.raises(ValueError, match="d_model"):
        attn(torch.randn(1, 4, D_MODEL + 1))


def test_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(d_model=250, n_head=4, block_size=8, dropout=0.0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"d_model": 0}, "d_model must be > 0"),
        ({"n_head": 0}, "n_head must be > 0"),
        ({"block_size": 0}, "block_size must be > 0"),
        ({"dropout": 1.0}, "dropout"),
        ({"dropout": -0.1}, "dropout"),
    ],
)
def test_rejects_invalid_constructor_args(kwargs, match):
    args = {"d_model": D_MODEL, "n_head": N_HEAD, "block_size": BLOCK, "dropout": DROPOUT}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        MultiHeadSelfAttention(**args)


# ---------------------------------------------------------------------------
# Causal masking
# ---------------------------------------------------------------------------
def test_mask_is_lower_triangular_buffer():
    attn = make_attn()
    names = dict(attn.named_buffers())
    assert "mask" in names
    assert "mask" not in dict(attn.named_parameters())
    assert attn.mask.shape == (1, 1, BLOCK, BLOCK)
    assert torch.equal(attn.mask[0, 0], torch.tril(torch.ones(BLOCK, BLOCK)))


@pytest.mark.parametrize("t", [0, 1, 5, 18])
def test_causal_mask(t):
    """Changing tokens at positions > t must not change outputs at positions <= t."""
    attn = make_attn(dropout=DROPOUT).eval()  # dropout > 0 but disabled by eval()
    T = 20
    torch.manual_seed(1)
    x = torch.randn(2, T, D_MODEL)
    x_mod = x.clone()
    x_mod[:, t + 1 :] = torch.randn(2, T - t - 1, D_MODEL) * 5.0  # perturb the future

    with torch.no_grad():
        out = attn(x)
        out_mod = attn(x_mod)

    # Past/present unchanged ...
    torch.testing.assert_close(out[:, : t + 1], out_mod[:, : t + 1], rtol=0, atol=1e-6)
    # ... while the future really did change (so the test is not vacuous).
    assert not torch.allclose(out[:, t + 1 :], out_mod[:, t + 1 :])


def test_causal_mask_gradient():
    """d out[:, t] / d x[:, j] must be exactly zero for every j > t."""
    attn = make_attn(dropout=0.0)
    T, t = 12, 4
    x = torch.randn(1, T, D_MODEL, requires_grad=True)
    attn(x)[:, t].sum().backward()
    grad = x.grad[0]
    assert torch.count_nonzero(grad[t + 1 :]) == 0
    assert torch.count_nonzero(grad[: t + 1]) > 0


def test_first_position_ignores_everything_else():
    """Position 0 can only see itself, so its output is independent of the rest."""
    attn = make_attn().eval()
    x = torch.randn(1, 10, D_MODEL)
    with torch.no_grad():
        full = attn(x)[:, 0]
        alone = attn(x[:, :1])[:, 0]
    torch.testing.assert_close(full, alone, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# Numerical sanity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seq_len", [1, 32, BLOCK])
@pytest.mark.parametrize("train_mode", [False, True])
def test_no_nan(seq_len, train_mode):
    attn = make_attn()
    attn.train(train_mode)
    x = torch.randn(4, seq_len, D_MODEL) * 10.0  # large activations
    out = attn(x)
    assert torch.isfinite(out).all()


def test_dropout_only_active_in_train_mode():
    attn = make_attn(dropout=0.5)
    x = torch.randn(2, 16, D_MODEL)
    attn.eval()
    with torch.no_grad():
        assert torch.equal(attn(x), attn(x))
    attn.train()
    with torch.no_grad():
        assert not torch.equal(attn(x), attn(x))


def test_parameter_count():
    # qkv: 256*768 + 768 = 197,376 ; proj: 256*256 + 256 = 65,792 (PROJECT_CONTEXT 8.2)
    attn = make_attn()
    n = sum(p.numel() for p in attn.parameters())
    assert n == 197_376 + 65_792 == 263_168


def test_no_forbidden_attention_apis():
    """Attention must be written by hand (PROJECT_CONTEXT section 10)."""
    src = (SRC / "reviewlens" / "attention.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in src.splitlines()
        if not line.lstrip().startswith("#")
    )
    # Strip the module docstring, which mentions the forbidden names on purpose.
    code = re.sub(r'^"""[\s\S]*?"""', "", code, count=1)
    for forbidden in (
        "MultiheadAttention",
        "nn.Transformer",
        "TransformerEncoder",
        "TransformerDecoder",
        "scaled_dot_product_attention",
    ):
        assert forbidden not in code, forbidden
