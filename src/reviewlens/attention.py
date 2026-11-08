"""Manual causal Multi-Head Self-Attention for the Mini-GPT decoder.

Contract: INTERFACES.md section 8.

Everything is computed explicitly -- no ``nn.MultiheadAttention``,
``nn.Transformer*`` or ``F.scaled_dot_product_attention``:

    Q, K, V   = split(Linear(d_model, 3*d_model)(x))            # (B, T, d_model) each
    Q, K, V   -> (B, n_head, T, d_head)
    scores    = Q @ K^T / sqrt(d_head)                           # (B, n_head, T, T)
    scores    = scores.masked_fill(future positions, -inf)       # causal mask (torch.tril)
    weights   = dropout(softmax(scores, dim=-1))
    out       = weights @ V                                      # (B, n_head, T, d_head)
    out       -> concat heads (B, T, d_model) -> Linear(d_model, d_model) -> dropout

This module knows nothing about tokens or data; it only maps tensors
``(B, T, d_model) -> (B, T, d_model)`` with ``T <= block_size``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["MultiHeadSelfAttention"]


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention implemented by hand.

    Args:
        d_model: model / embedding width; must be divisible by ``n_head``.
        n_head: number of attention heads (``d_head = d_model // n_head``).
        block_size: maximum sequence length; size of the causal mask buffer.
        dropout: dropout probability for attention weights and output projection.
    """

    def __init__(self, d_model: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        # Validation order: d_model, n_head, block_size, divisibility, dropout.
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if n_head <= 0:
            raise ValueError(f"n_head must be > 0, got {n_head}")
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if d_model % n_head != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_head ({n_head})")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.block_size = block_size

        # One fused projection producing Q, K and V: d_model -> 3 * d_model.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Output projection after the heads are concatenated.
        self.proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)    # on attention weights
        self.resid_dropout = nn.Dropout(dropout)   # on the projected output

        # Causal mask: 1 on/below the diagonal (allowed), 0 above it (future).
        # Shape (1, 1, block_size, block_size) so it broadcasts over (B, n_head).
        # A buffer (not a Parameter): moves with .to(device), is never trained.
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model) with T <= block_size  ->  (B, T, d_model)."""
        if x.dim() != 3:
            raise ValueError(f"expected input of shape (B, T, d_model), got {tuple(x.shape)}")
        B, T, C = x.shape
        if C != self.d_model:
            raise ValueError(f"last dim of input is {C}, expected d_model={self.d_model}")
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")

        # 1. Linear projection to Q, K, V -> each (B, T, d_model).
        q, k, v = self.qkv(x).split(self.d_model, dim=2)

        # 2. Split into heads: (B, T, d_model) -> (B, T, n_head, d_head) -> (B, n_head, T, d_head).
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # 3-4. Scaled dot-product scores: (B, nh, T, d_head) @ (B, nh, d_head, T) -> (B, nh, T, T).
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 5-6. Causal mask: position i may only attend to j <= i; future -> -inf.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # 7. Softmax over the key dimension (each row sums to 1; -inf -> weight 0).
        att = F.softmax(att, dim=-1)

        # 8. Dropout on the attention weights.
        att = self.attn_dropout(att)

        # 9. Weighted sum of values: (B, nh, T, T) @ (B, nh, T, d_head) -> (B, nh, T, d_head).
        y = att @ v

        # 10. Concatenate heads: (B, nh, T, d_head) -> (B, T, nh, d_head) -> (B, T, d_model).
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 11-12. Output projection + residual dropout.
        return self.resid_dropout(self.proj(y))
