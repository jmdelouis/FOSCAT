"""
HealpixViT — Vision Transformer on HEALPix with Foscat
======================================================

This module provides a **Vision Transformer (ViT)** adapted to spherical data laid out on a
**HEALPix nested grid**. It integrates **Foscat**'s `SphericalStencil` operators to perform
spherical convolutions for patch embedding, hierarchical **Down/Up** between HEALPix levels,
and an optional per-pixel spherical head after the Transformer encoder.

Why this design?
----------------
- HEALPix provides a hierarchical, equal-area tessellation of the sphere. In **nested** ordering,
  each pixel at level \(L\) has 4 children at level \(L+1\). This makes **tokenization** natural:
  we can repeatedly call `Down()` to move to a coarser grid that serves as the **token grid**.
- A Transformer encoder then operates on the **sequence of tokens**. For dense outputs, we map the
  token features back to the finest grid with `Up()` and refine with a spherical convolution head.
- Because we reuse the same Foscat operators as in a HEALPix U-Net, we preserve consistency with
  existing spherical CNN pipelines while gaining the long-range modeling capacity of Transformers.

Typical use cases
-----------------
- **Global regression/classification** (e.g., predicting a climate index from full-sky fields).
- **Dense regression/segmentation** (e.g., SST anomaly prediction, cloud/ice masks) directly on
  HEALPix maps, including **multi-resolution fusion** thanks to nested Down/Up.

Notes on `cell_ids`
-------------------
- This implementation supports passing **runtime `cell_ids`** to `forward(...)` to match your
  data pipeline (e.g., when per-sample IDs are managed externally). If omitted, it uses the
  `cell_ids` provided at construction.
- All IDs are assumed to be **nested** and **int64**, with range `[0, 12*nside^2 - 1]` at each level.
  Sanity checks are included to prevent HEALPix `pix2loc` errors.
"""

from __future__ import annotations
from typing import List, Optional, Literal, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import foscat.scat_cov as sc
import foscat.SphericalStencil as ho

# -----------------------------------------------------------------------------
# Helper: safe type alias
# -----------------------------------------------------------------------------
ArrayLikeI64 = Union[np.ndarray, torch.Tensor]


class HealpixViT(nn.Module):
    """Vision Transformer on the HEALPix sphere using Foscat-oriented ops.

    Parameters
    ----------
    in_nside : int
        Input HEALPix nside at the **finest** level (nested ordering). The number of pixels is
        `Npix = 12 * in_nside**2`.
    n_chan_in : int
        Number of input channels at the finest grid.
    embed_dim : int
        Transformer embedding dimension (also the channel count after patch embedding).
    depth : int
        Number of Transformer encoder layers.
    num_heads : int
        Number of attention heads per layer.
    cell_ids : np.ndarray
        Finest-level **nested** cell indices (shape `[Npix]`, dtype `int64`). These define the
        pixel layout of your input features.
    mlp_ratio : float, default=4.0
        Expansion ratio for the MLP inside each Transformer block.
    token_down : int, default=2
        Number of `Down()` steps to reach the token grid. The token nside is
        `token_nside = in_nside // (2**token_down)`.
    task : {"regression","segmentation","global"}, default="regression"
        - "global": return a vector (pooled tokens → `out_channels`).
        - "regression"/"segmentation": return per-pixel predictions on the finest grid.
    out_channels : int, default=1
        Output channels for dense tasks (ignored for `task="global"`).
    final_activation : {"none","sigmoid","softmax"} | None
        Optional activation for the output. If `None`, sensible defaults are chosen per task.
    KERNELSZ : int, default=3
        Spatial kernel size for spherical convolutions (Foscat oriented conv).
    gauge_type : {"cosmo","phi"}, default="cosmo"
        Orientation/gauge definition in `SphericalStencil`.
    G : int, default=1
        Number of gauges (internal orientation multiplicity). `embed_dim` must be divisible by `G`.
    prefer_foscat_gpu : bool, default=True
        Try Foscat on CUDA if available; fall back to CPU otherwise.
    cls_token : bool, default=False
        Include a `[CLS]` token for global tasks.
    pos_embed : {"learned","none"}, default="learned"
        Positional encoding type for tokens.
    head_type : {"mean","cls"}, default="mean"
        Pooling strategy for global tasks (mean over tokens or CLS vector).
    dtype : {"float32","float64"}, default="float32"
        Numpy dtype used for internal Foscat buffers. Model parameters remain `float32`.

    Input/Output shapes
    -------------------
    Input:  `(B, C_in, Npix)` with `Npix = 12 * in_nside**2`.
    Output: - global task: `(B, out_channels)`
            - dense task:  `(B, out_channels, Npix)`
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        in_nside: int,
        n_chan_in: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        cell_ids: np.ndarray,
        mlp_ratio: float = 4.0,
        token_down: int = 2,
        task: Literal["regression","segmentation","global"] = "regression",
        out_channels: int = 1,
        final_activation: Optional[Literal["none","sigmoid","softmax"]] = None,
        KERNELSZ: int = 3,
        gauge_type: Optional[Literal["cosmo","phi"]] = "cosmo",
        G: int = 1,
        prefer_foscat_gpu: bool = True,
        cls_token: bool = False,
        pos_embed: Literal["learned","none"] = "learned",
        head_type: Literal["mean","cls"] = "mean",
        dtype: Literal["float32","float64"] = "float32",
    ) -> None:
        super().__init__()

        # ------------------- store config & dtypes -------------------
        self.in_nside = int(in_nside)
        self.n_chan_in = int(n_chan_in)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.token_down = int(token_down)
        self.task = task
        self.out_channels = int(out_channels)
        self.KERNELSZ = int(KERNELSZ)
        self.gauge_type = gauge_type
        self.G = int(G)
        self.prefer_foscat_gpu = bool(prefer_foscat_gpu)
        self.cls_token_enabled = bool(cls_token)
        self.pos_embed_type = pos_embed
        self.head_type = head_type

        if dtype == "float32":
            self.np_dtype = np.float32
            self.torch_dtype = torch.float32
        else:
            self.np_dtype = np.float64
            self.torch_dtype = torch.float32  # keep params in fp32

        # ------------------- validate inputs -------------------
        if cell_ids is None:
            raise ValueError("cell_ids (finest) must be provided.")
        self.cell_ids_fine = np.asarray(cell_ids)
        self._check_ids(self.cell_ids_fine, self.in_nside, name="cell_ids_fine")

        if self.G < 1:
            raise ValueError("G must be >= 1")
        if self.embed_dim % self.G != 0:
            raise ValueError(f"embed_dim={self.embed_dim} must be divisible by G={self.G}")
        if self.task not in {"regression", "segmentation", "global"}:
            raise ValueError("task must be 'regression', 'segmentation', or 'global'")

        # Default final activation per task if not specified
        if final_activation is None:
            if self.task == "regression":
                self.final_activation = "none"
            elif self.task == "segmentation":
                self.final_activation = "sigmoid" if out_channels == 1 else "softmax"
            else:
                self.final_activation = "none"
        else:
            self.final_activation = final_activation

        # ------------------- foscat functional wrapper -------------------
        self.f = sc.funct(KERNELSZ=self.KERNELSZ)

        # ------------------- build hierarchy (fine → coarse) -------------------
        # We progressively `Down()` to precompute the token grid ids and operators.
        self.hconv_levels: List[ho.SphericalStencil] = []  # op at successive levels
        self.level_cell_ids: List[np.ndarray] = [self.cell_ids_fine]
        current_nside = self.in_nside

        # dummy buffer to probe Down; lives in Foscat backend dtype
        dummy = self.f.backend.bk_cast(
            np.zeros((1, 1, self.cell_ids_fine.shape[0]), dtype=self.np_dtype)
        )

        for _ in range(self.token_down):
            hc = ho.SphericalStencil(
                current_nside,
                self.KERNELSZ,
                n_gauges=self.G,
                gauge_type=self.gauge_type,
                cell_ids=self.level_cell_ids[-1],
                dtype=self.torch_dtype,
            )
            self.hconv_levels.append(hc)

            # Down to get next cell ids
            dummy, next_ids = hc.Down(
                dummy,
                cell_ids=self.level_cell_ids[-1],
                nside=current_nside,
                max_poll=True,
            )
            next_ids = self.f.backend.to_numpy(next_ids)
            current_nside //= 2
            self._check_ids(next_ids, current_nside, name="token_level_cell_ids")
            self.level_cell_ids.append(next_ids)

        # token grid (where the Transformer runs)
        self.token_nside = current_nside if self.token_down > 0 else self.in_nside
        if self.token_nside < 1:
            raise ValueError(
                f"token_down={self.token_down} too large for in_nside={self.in_nside}"
            )
        self.token_cell_ids = self.level_cell_ids[-1]

        # Operators at token and fine levels (used for Up and head)
        self.hconv_token = ho.SphericalStencil(
            self.token_nside,
            self.KERNELSZ,
            n_gauges=self.G,
            gauge_type=self.gauge_type,
            cell_ids=self.token_cell_ids,
            dtype=self.torch_dtype,
        )
        self.hconv_head = ho.SphericalStencil(
            self.in_nside,
            self.KERNELSZ,
            n_gauges=self.G,
            gauge_type=self.gauge_type,
            cell_ids=self.cell_ids_fine,
            dtype=self.torch_dtype,
        )

        # ------------------- patch embedding (finest grid) -------------------
        embed_g = self.embed_dim // self.G
        # weight shapes follow Foscat conv expectations: (Cin, Cout_per_gauge, KERNELSZ*KERNELSZ)
        self.patch_w1 = nn.Parameter(
            torch.empty(self.n_chan_in, embed_g, self.KERNELSZ * self.KERNELSZ)
        )
        nn.init.kaiming_uniform_(self.patch_w1.view(self.n_chan_in * embed_g, -1), a=np.sqrt(5))
        self.patch_bn1 = nn.GroupNorm(
            num_groups=min(8, embed_g if embed_g > 1 else 1), num_channels=self.embed_dim
        )

        self.patch_w2 = nn.Parameter(
            torch.empty(self.embed_dim, embed_g, self.KERNELSZ * self.KERNELSZ)
        )
        nn.init.kaiming_uniform_(self.patch_w2.view(self.embed_dim * embed_g, -1), a=np.sqrt(5))
        self.patch_bn2 = nn.GroupNorm(
            num_groups=min(8, embed_g if embed_g > 1 else 1), num_channels=self.embed_dim
        )

        # ------------------- positional encoding -------------------
        self.n_tokens = int(self.token_cell_ids.shape[0])
        if self.cls_token_enabled:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            n_pe = self.n_tokens + 1
        else:
            self.cls_token = None
            n_pe = self.n_tokens

        if self.pos_embed_type == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, n_pe, self.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        # ------------------- transformer encoder -------------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.embed_dim * self.mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.depth)

        # ------------------- output heads -------------------
        if self.task == "global":
            # Global head: a single Linear on pooled token features
            self.global_head = nn.Linear(self.embed_dim, self.out_channels)
        else:
            # Dense head: project token embeddings to channels, Up to fine grid, optional conv
            if self.out_channels % self.G != 0:
                raise ValueError(
                    f"out_channels={self.out_channels} must be divisible by G={self.G}"
                )
            out_g = self.out_channels // self.G
            self.token_proj = nn.Linear(self.embed_dim, self.G * out_g)
            self.head_w = nn.Parameter(
                torch.empty(self.out_channels, out_g, self.KERNELSZ * self.KERNELSZ)
            )
            nn.init.kaiming_uniform_(
                self.head_w.view(self.out_channels * out_g, -1), a=np.sqrt(5)
            )
            self.head_bn = (
                nn.GroupNorm(
                    num_groups=min(8, self.out_channels if self.out_channels > 1 else 1),
                    num_channels=self.out_channels,
                )
                if self.task == "segmentation"
                else None
            )

        # ------------------- device probing (CUDA → CPU fallback) -------------------
        pref = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.runtime_device = self._probe_and_set_runtime_device(pref)

    # ------------------------------------------------------------------
    # Internal sanity checks
    # ------------------------------------------------------------------
    @staticmethod
    def _check_ids(ids: ArrayLikeI64, nside: int, name: str = "cell_ids") -> None:
        """Sanity check to avoid HEALPix `pix2loc` errors.
        Ensures dtype=int64, range in [0, 12*nside^2 - 1].
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().numpy()
        ids = np.asarray(ids)
        if ids.dtype != np.int64:
            raise TypeError(f"{name} must be int64, got {ids.dtype}.")
        npix = 12 * nside * nside
        imin, imax = int(ids.min()), int(ids.max())
        if imin < 0 or imax >= npix:
            raise ValueError(
                f"{name} out of range for nside={nside}: min={imin}, max={imax}, allowed=[0,{npix-1}]"
            )

    # ------------------------------------------------------------------
    # Device utilities
    # ------------------------------------------------------------------
    def _move_hc(self, hc: ho.SphericalStencil, device: torch.device) -> None:
        """Move internal tensors of SphericalStencil to the given device.
        This mirrors the plumbing in U-Net-like codebases using Foscat.
        """
        for name, val in list(vars(hc).items()):
            try:
                if torch.is_tensor(val):
                    setattr(hc, name, val.to(device))
                elif isinstance(val, (list, tuple)) and val and torch.is_tensor(val[0]):
                    setattr(hc, name, type(val)([v.to(device) for v in val]))
            except Exception:
                # Some attributes may be non-tensors; ignore.
                pass

    @torch.no_grad()
    def _probe_and_set_runtime_device(self, preferred: torch.device) -> torch.device:
        """Try to run on CUDA with Foscat; otherwise, gracefully fall back to CPU.
        Performs a tiny dry-run spherical convolution to ensure compatibility.
        """
        if preferred.type == "cuda" and self.prefer_foscat_gpu:
            try:
                super().to(preferred)
                for hc in self.hconv_levels + [self.hconv_token, self.hconv_head]:
                    self._move_hc(hc, preferred)
                # Dry run: minimal conv on finest grid
                npix0 = int(self.cell_ids_fine.shape[0])
                x_try = torch.zeros(1, self.n_chan_in, npix0, device=preferred)
                hc0 = self.hconv_levels[0] if len(self.hconv_levels) > 0 else self.hconv_head
                y_try = hc0.Convol_torch(x_try, self.patch_w1)
                _ = (y_try if torch.is_tensor(y_try) else torch.as_tensor(y_try, device=preferred)).sum().item()
                self._foscat_device = preferred
                return preferred
            except Exception as e:
                # Record and fall back
                self._gpu_probe_error = repr(e)
        cpu = torch.device("cpu")
        super().to(cpu)
        for hc in self.hconv_levels + [self.hconv_token, self.hconv_head]:
            self._move_hc(hc, cpu)
        self._foscat_device = cpu
        return cpu

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _patch_embed(self, x: torch.Tensor, cell_ids: Optional[np.ndarray]) -> torch.Tensor:
        """Spherical patch embedding at the **finest** grid.
        Applies two Foscat oriented convolutions with GN+GELU.
        Input `(B, C_in, Nfine)` → Output `(B, embed_dim, Nfine)`.
        """
        hc0 = self.hconv_levels[0] if len(self.hconv_levels) > 0 else self.hconv_head
        if cell_ids is None:
            # Use constructor-time ids
            y = hc0.Convol_torch(x, self.patch_w1)
            y = self._as_tensor_batch(y)
            y = self.patch_bn1(y)
            y = F.gelu(y)
            y = hc0.Convol_torch(y, self.patch_w2)
            y = self._as_tensor_batch(y)
            y = self.patch_bn2(y)
            y = F.gelu(y)
            return y
        else:
            # Use runtime ids provided by caller
            y = hc0.Convol_torch(x, self.patch_w1, cell_ids=cell_ids)
            y = self._as_tensor_batch(y)
            y = self.patch_bn1(y)
            y = F.gelu(y)
            y = hc0.Convol_torch(y, self.patch_w2, cell_ids=cell_ids)
            y = self._as_tensor_batch(y)
            y = self.patch_bn2(y)
            y = F.gelu(y)
            return y

    def _down_to_tokens(
        self, x: torch.Tensor, cell_ids: Optional[np.ndarray]
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Apply `token_down` Down() steps to reach the **token grid**.
        Returns `(x_tokens, token_cell_ids)` where `x_tokens` has shape `(B, C, N_tokens)`.
        If `cell_ids` is provided, uses them as the starting fine-grid ids; otherwise uses
        the constructor-time `self.cell_ids_fine`.
        """
        l_data = x
        l_cell_ids = self.cell_ids_fine if cell_ids is None else np.asarray(cell_ids)
        current_nside = self.in_nside

        for hc in self.hconv_levels:
            l_data, l_cell_ids = hc.Down(
                l_data, cell_ids=l_cell_ids, nside=current_nside, max_poll=True
            )
            l_data = self._as_tensor_batch(l_data)
            current_nside //= 2
        return l_data, l_cell_ids

    def _tokens_to_sequence(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """Rearrange `(B, C, Ntok)` → `(B, Ntok(+CLS), C)` and add positional embeddings."""
        B, C, Nt = x_tokens.shape
        seq = x_tokens.permute(0, 2, 1)  # (B, Nt, C)
        if self.cls_token_enabled:
            cls = self.cls_token.expand(B, -1, -1)
            seq = torch.cat([cls, seq], dim=1)  # (B, 1+Nt, C)
        if self.pos_embed is not None:
            seq = seq + self.pos_embed[:, : seq.shape[1], :]
        return seq

    def _sequence_to_tokens(
        self, seq: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Strip CLS if present and return `(tokens_only, cls_vector)`."""
        if self.cls_token_enabled:
            cls_vec = seq[:, 0, :]
            tokens = seq[:, 1:, :]
            return tokens, cls_vec
        return seq, None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, cell_ids: Optional[ArrayLikeI64] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(B, C_in, Npix)` at the finest grid.
        cell_ids : Optional[np.ndarray or torch.Tensor]
            Optional **nested** pixel indices for the input; if provided, they are used throughout
            the pipeline (patch embedding, Down, Up, head conv). If `None`, the constructor-time
            `cell_ids` are used.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.dim() != 3:
            raise ValueError("Input must be (B, C, Npix)")
        if x.shape[1] != self.n_chan_in:
            raise ValueError(f"Expected {self.n_chan_in} channels, got {x.shape[1]}")

        # Normalize/validate runtime ids once
        runtime_ids = None
        if cell_ids is not None:
            if isinstance(cell_ids, torch.Tensor):
                cell_ids = cell_ids.detach().cpu().numpy()
            cell_ids = np.asarray(cell_ids)
            # If given per-batch ids (B, Npix), take first row (assume same layout for the batch)
            if cell_ids.ndim == 2:
                cell_ids = cell_ids[0]
            self._check_ids(cell_ids, self.in_nside, name="forward:cell_ids")
            runtime_ids = cell_ids

        x = x.to(self.runtime_device)

        # 1) Patch embedding (finest grid)
        x = self._patch_embed(x, runtime_ids)  # (B, embed_dim, Nfine)

        # 2) Down to token grid
        x_tok, token_ids = self._down_to_tokens(x, runtime_ids)  # (B, embed_dim, Ntok)

        # 3) Transformer encoder on token sequence
        seq = self._tokens_to_sequence(x_tok)  # (B, Ntok(+1), embed_dim)
        seq = self.encoder(seq)               # (B, Ntok(+1), embed_dim)
        tokens, cls_vec = self._sequence_to_tokens(seq)

        if self.task == "global":
            # Global vector from mean/CLS pooling
            if self.head_type == "cls" and self.cls_token_enabled and cls_vec is not None:
                out = self.global_head(cls_vec)  # (B, out_channels)
            else:
                out = self.global_head(tokens.mean(dim=1))
            return out

        # 4) Project tokens to channels at token grid
        tok_proj = self.token_proj(tokens)      # (B, Ntok, out_channels)
        tok_proj = tok_proj.permute(0, 2, 1)  # (B, out_channels, Ntok)
        # Sanity: token feature count must match token_ids length
        if isinstance(token_ids, torch.Tensor):
            _tok_ids = token_ids.detach().cpu().numpy()
        else:
            _tok_ids = np.asarray(token_ids)
        assert tok_proj.shape[-1] == _tok_ids.shape[0], (
            f"Ntok mismatch: {tok_proj.shape[-1]} != {_tok_ids.shape[0]}"
        )

        # 5) Up from token grid to finest grid
        # Use constructor-time fine ids by default; override if runtime ids provided.
        fine_ids = self.cell_ids_fine if runtime_ids is None else runtime_ids        # 5) Multi-step Up from token grid to finest grid (one HEALPix level at a time)
        # Use constructor-time fine ids by default; override if runtime ids provided.
        fine_ids_runtime = self.cell_ids_fine if runtime_ids is None else runtime_ids

        # Build the ID chain from fine → ... → token for THIS forward, using runtime ids
        _ids = fine_ids_runtime
        nside_tmp = self.in_nside
        ids_chain = [np.asarray(_ids)]
        _dummy = self.f.backend.bk_cast(np.zeros((1, 1, ids_chain[0].shape[0]), dtype=self.np_dtype))
        for hc in self.hconv_levels:
            _dummy, _next = hc.Down(_dummy, cell_ids=ids_chain[-1], nside=nside_tmp, max_poll=True)
            ids_chain.append(self.f.backend.to_numpy(_next))
            nside_tmp //= 2

        # Sanity: token_ids from the actual Down path must match the last element of the chain
        if isinstance(token_ids, torch.Tensor):
            _tok_ids = token_ids.detach().cpu().numpy()
        else:
            _tok_ids = np.asarray(token_ids)
        assert tok_proj.shape[-1] == _tok_ids.shape[0], f"Ntok mismatch: {tok_proj.shape[-1]} != {_tok_ids.shape[0]}"
        assert np.array_equal(_tok_ids, ids_chain[-1]), "token_ids mismatch with runtime Down() chain"

        # Precompute nsides represented by hconv_levels (fine→coarse, excluding token level)
        nsides_levels = [self.in_nside // (2 ** k) for k in range(self.token_down)]  # e.g., [8, 4] for token_down=2

        # Now Up step-by-step: token (coarse) → ... → fine
        y_up = tok_proj
        for i in range(len(ids_chain) - 1, 0, -1):
            coarse_ids = ids_chain[i]
            fine_ids_step = ids_chain[i - 1]
            source_nside = self.in_nside // (2 ** i)       # e.g., 2, then 4
            fine_nside   = self.in_nside // (2 ** (i - 1)) # e.g., 4, then 8
            # pick the operator of the target (fine) level
            if fine_nside == self.in_nside:
                op_fine = self.hconv_head
            else:
                idx = nsides_levels.index(fine_nside)
                op_fine = self.hconv_levels[idx]
            y_up = op_fine.Up(y_up, cell_ids=coarse_ids, o_cell_ids=fine_ids_step, nside=source_nside)
            if not torch.is_tensor(y_up):
                y_up = torch.as_tensor(y_up, device=self.runtime_device)
            y_up = self._as_tensor_batch(y_up)  # (B, out_channels, N at this fine level)


        # 6) Optional spherical head conv for refinement
        y = self.hconv_head.Convol_torch(y_up, self.head_w, cell_ids=fine_ids)
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, device=self.runtime_device)
        y = self._as_tensor_batch(y)

        if self.task == "segmentation" and self.head_bn is not None:
            y = self.head_bn(y)

        if self.final_activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.final_activation == "softmax":
            y = torch.softmax(y, dim=1)
        return y

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def _as_tensor_batch(self, x):
        """Normalize outputs of Foscat ops into a contiguous batch tensor.
        Foscat may return a tensor or a single-element list of tensors.
        This function ensures we always get a tensor of the expected shape.
        """
        if isinstance(x, list):
            if len(x) == 1:
                t = x[0]
                return t.unsqueeze(0) if t.dim() == 2 else t
            raise ValueError("Variable-length list not supported here; pass a tensor.")
        return x

    @torch.no_grad()
    def predict(
        self, x: Union[torch.Tensor, np.ndarray], batch_size: int = 8
    ) -> torch.Tensor:
        """Convenience method for batched inference.

        Parameters
        ----------
        x : Tensor or ndarray
            Input `(B, C_in, Npix)`.
        batch_size : int
            Mini-batch size used during prediction.
        """
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        outs = []
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size].to(self.runtime_device)
            outs.append(self.forward(xb))
        return torch.cat(outs, dim=0)


# -----------------------------------------------------------------------------
# Minimal smoke test (requires foscat installed)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # A tiny grid to validate shapes and device plumbing
    in_nside = 4
    npix = 12 * in_nside * in_nside
    cell_ids = np.arange(npix, dtype=np.int64)  # nested, fine-level ids

    B, Cin = 2, 3
    x = torch.randn(B, Cin, npix)

    model = HealpixViT(
        in_nside=in_nside,
        n_chan_in=Cin,
        embed_dim=64,
        depth=2,
        num_heads=4,
        cell_ids=cell_ids,
        token_down=2,         # token_nside = in_nside // 4 = 1 here
        task="regression",
        out_channels=1,
        KERNELSZ=3,
        G=1,
        cls_token=False,
    )

    with torch.no_grad():
        y = model(x)  # You can also pass `cell_ids=cell_ids` if your pipeline manages them at runtime
        print("Output:", y.shape)  # (B, out_channels, npix)
