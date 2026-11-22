"""Self-contained PyTorch models for Proofline quickstart demos."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 2048) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.fc2(F.silu(self.fc1(h)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 8) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
        x = x + self.proj(attn.transpose(1, 2).contiguous().view(b, n, d))
        h2 = self.norm2(x)
        return x + self.fc2(F.gelu(self.fc1(h2)))


class FusedConvBlock(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return h + F.relu(h) * self.scale


def mlp_block() -> nn.Module:
    return ResidualMLPBlock()


def transformer_block() -> nn.Module:
    return TransformerBlock()


def conv_block() -> nn.Module:
    return FusedConvBlock()
