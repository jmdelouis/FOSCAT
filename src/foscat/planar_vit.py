# healpix_unet_torch.py
# (Planar Vision Transformer baseline for lat–lon grids)
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Building blocks
# ---------------------------


class _MLP(nn.Module):
    """ViT MLP: Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class _ViTBlock(nn.Module):
    """
    Transformer block (Pre-LN):
      x = x + Drop(MHA(LN(x)))
      x = x + Drop(MLP(LN(x)))
    """

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.1
    ):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio, drop)
        self.drop_path = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head self-attention
        x = x + self.drop_path(
            self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        )
        # Feed-forward
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------
# Planar ViT (lat–lon images)
# ---------------------------


class PlanarViT(nn.Module):
    """
    Vision Transformer for 2D lat–lon grids (planar baseline).

    Input : (B, C=T_in, H, W)
    Output: (B, out_ch, H, W)   # dense per-pixel prediction

    Pipeline
    --------
    1) Patch embedding via Conv2d(kernel_size=patch, stride=patch) -> embed_dim
    2) Optional CLS token (disabled by default for dense output)
    3) Learned positional embeddings (or none)
    4) Stack of Transformer blocks
    5) Linear head per token, then nearest upsample back to (H, W)

    Notes
    -----
    - Keep H, W divisible by `patch`.
    - For residual-of-persistence training (recommended for monthly SST):
        pred = x[:, -1:, ...] + model(x)
      and train the loss on `pred` vs target.
    """

    def __init__(
        self,
        in_ch: int,  # e.g., T_in months
        H: int,
        W: int,
        *,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch: int = 4,
        out_ch: int = 1,
        dropout: float = 0.1,
        cls_token: bool = False,    # keep False for dense prediction
        pos_embed: str = "learned", # or "none"
    ):
        super().__init__()
        assert H % patch == 0 and W % patch == 0, "H and W must be divisible by patch"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.H, self.W = H, W
        self.patch = patch
        self.embed_dim = embed_dim
        self.cls_token_enabled = bool(cls_token)
        self.use_pos_embed = (pos_embed == "learned")

        # 1) Patch embedding (Conv2d with stride=patch) → tokens
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

        # 2) Token bookkeeping & positional embeddings
        Hp, Wp = H // patch, W // patch
        self.num_tokens = Hp * Wp + (1 if self.cls_token_enabled else 0)

        if self.cls_token_enabled:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        # 3) Transformer encoder
        self.blocks = nn.ModuleList([
            _ViTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=dropout)
            for _ in range(depth)
        ])

        # 4) Patch-wise head (token -> out_ch)
        self.head = nn.Linear(embed_dim, out_ch)

        # Store for unpatching
        self.Hp, self.Wp = Hp, Wp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)  with H,W fixed to construction-time H,W
        returns: (B, out_ch, H, W)
        """
        B, C, H, W = x.shape
        if (H != self.H) or (W != self.W):
            raise ValueError(f"Input H,W must be ({self.H},{self.W}), got ({H},{W}).")

        # Patch embedding → (B, E, Hp, Wp) → (B, Np, E)
        z = self.patch_embed(x)                 # (B, E, Hp, Wp)
        z = z.flatten(2).transpose(1, 2)        # (B, Np, E)

        # Optional CLS
        if self.cls_token_enabled:
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,E)
            z = torch.cat([cls, z], dim=1)          # (B,1+Np,E)

        # Positional embedding
        if self.pos_embed is not None:
            z = z + self.pos_embed[:, :z.shape[1], :]

        # Transformer
        for blk in self.blocks:
            z = blk(z)                             # (B, N, E)

        # Drop CLS for dense output
        if self.cls_token_enabled:
            tokens = z[:, 1:, :]                   # (B, Np, E)
        else:
            tokens = z

        # Token head → (B, Np, out_ch) → (B, out_ch, Hp, Wp) → upsample to (H, W)
        y_tok = self.head(tokens).transpose(1, 2)  # (B, out_ch, Np)
        y = y_tok.reshape(B, -1, self.Hp, self.Wp) # (B, out_ch, Hp, Wp)
        y = F.interpolate(y, scale_factor=self.patch, mode="nearest")
        return y


# ---------------------------
# Utilities
# ---------------------------

def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------
# Smoke test
# ---------------------------

if __name__ == "__main__":
    # Example: T_in=6, grid 128x256, predict 1 channel
    B, C, H, W = 2, 6, 128, 256
    x = torch.randn(B, C, H, W)

    model = PlanarViT(
        in_ch=C, H=H, W=W,
        embed_dim=384, depth=8, num_heads=12,
        mlp_ratio=4.0, patch=4, out_ch=1, dropout=0.1,
        cls_token=False, pos_embed="learned"
    )
    y = model(x)
    tot, trn = count_parameters(model)
    print("Output:", tuple(y.shape))
    print("Params:", f"total={tot:,}", f"trainable={trn:,}")
