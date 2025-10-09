# healpix_vit_varlevels.py
# HEALPix ViT with level-wise (variable) channel widths and U-Net-style spherical decoder
from __future__ import annotations
from typing import List, Optional, Literal, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import foscat.scat_cov as sc
import foscat.SphericalStencil as ho


class HealpixViTVarLevels(nn.Module):
    """
    HEALPix Vision Transformer (Foscat-based) with *variable channel widths per level*
    and a U-Net-like spherical decoder.

    Key idea
    --------
    - Encoder uses a list of channel dimensions `level_dims = [C_fine, C_l1, ..., C_token]`
      that evolve *with depth* (e.g., 128 -> 192 -> 256).
    - At each encoder level (before a Down()), we apply a spherical convolution that
      maps C_i -> C_{i+1}. Down() then reduces the HEALPix resolution by one level.
    - Transformer runs at the token grid with embedding dim = C_token.
    - Decoder upsamples *one level at a time*; after each Up() it concatenates the
      upsampled token features (C_{i+1}) with the corresponding skip (C_i) and applies
      a spherical convolution to fuse (C_{i+1} + C_i) -> C_i.
    - Final head maps C_fine -> out_channels at the finest grid.

    Shapes (dense tasks)
    --------------------
    Input : (B, Cin, Nfine)
      → patch-embed (Cin -> C_fine) at finest grid
      → for i in [0..L-1]: EncConv(C_i->C_{i+1}) → Down()  (store skip_i=C_i at grid i)
      → tokens at grid L with dim C_token
      → Transformer on tokens (C_token)
      → for i in [L-1..0]: Up() to grid i  → concat(skip_i, up) [C_i + C_{i+1}] → DecConv → C_i
      → Head: C_fine -> out_channels at finest grid

    Requirements
    ------------
    - level_dims length must be token_down+1, with:
        len(level_dims) = token_down + 1
        level_dims[0]   = channels at finest grid after patch embedding
        level_dims[-1]  = Transformer embedding dimension
    - Each value in level_dims must be divisible by G (number of gauges).
    - out_channels must be divisible by G.

    Parameters (main)
    -----------------
    in_nside        : input HEALPix nside (nested)
    n_chan_in       : input channels at finest grid (Cin)
    level_dims      : list of ints, channel width per level from fine to token
    depth           : number of Transformer encoder layers
    num_heads       : self-attention heads
    cell_ids        : finest-level nested indices (Nfine = 12*nside^2)
    task            : "regression" | "segmentation" | "global"
    out_channels    : output channels for dense tasks
    KERNELSZ        : spherical kernel size for Foscat convolutions
    gauge_type      : "cosmo" | "phi"
    G               : number of gauges
    """

    def __init__(
        self,
        *,
        in_nside: int,
        n_chan_in: int,
        level_dims: List[int],       # e.g., [128, 192, 256]  (fine -> token)
        depth: int,
        num_heads: int,
        cell_ids: np.ndarray,
        task: Literal["regression", "segmentation", "global"] = "regression",
        out_channels: int = 1,
        mlp_ratio: float = 4.0,
        KERNELSZ: int = 3,
        gauge_type: Literal["cosmo", "phi"] = "cosmo",
        G: int = 1,
        prefer_foscat_gpu: bool = True,
        cls_token: bool = False,
        pos_embed: Literal["learned", "none"] = "learned",
        head_type: Literal["mean", "cls"] = "mean",
        dropout: float = 0.0,
        dtype: Literal["float32", "float64"] = "float32",
    ) -> None:
        super().__init__()

        # ---- config ----
        self.in_nside = int(in_nside)
        self.n_chan_in = int(n_chan_in)
        self.level_dims = list(level_dims)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.task = task
        self.out_channels = int(out_channels)
        self.mlp_ratio = float(mlp_ratio)
        self.KERNELSZ = int(KERNELSZ)
        self.gauge_type = gauge_type
        self.G = int(G)
        self.prefer_foscat_gpu = bool(prefer_foscat_gpu)
        self.cls_token_enabled = bool(cls_token)
        self.pos_embed_type = pos_embed
        self.head_type = head_type
        self.dropout = float(dropout)
        self.dtype = dtype

        if len(self.level_dims) < 1:
            raise ValueError("level_dims must have at least one element (fine level).")
        self.token_down = len(self.level_dims) - 1
        self.embed_dim = int(self.level_dims[-1])  # Transformer dim

        for d in self.level_dims:
            if d % self.G != 0:
                raise ValueError(f"Each level dim must be divisible by G={self.G}; got {d}.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        if dtype == "float32":
            self.np_dtype = np.float32
            self.torch_dtype = torch.float32
        else:
            self.np_dtype = np.float64
            self.torch_dtype = torch.float32  # keep model in fp32

        if cell_ids is None:
            raise ValueError("cell_ids (finest) must be provided (nested ordering).")
        self.cell_ids_fine = np.asarray(cell_ids)

        # Default activation
        if self.task == "segmentation":
            self.final_activation = "sigmoid" if self.out_channels == 1 else "softmax"
        else:
            self.final_activation = "none"

        # Foscat wrapper
        self.f = sc.funct(KERNELSZ=self.KERNELSZ)

        # ---- Build operators per level (fine -> ... -> token) and compute ids ----
        self.hconv_levels: List[ho.SphericalStencil] = []
        self.level_cell_ids: List[np.ndarray] = [self.cell_ids_fine]
        current_nside = self.in_nside

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

            dummy, next_ids = hc.Down(
                dummy, cell_ids=self.level_cell_ids[-1], nside=current_nside, max_poll=True
            )
            self.level_cell_ids.append(self.f.backend.to_numpy(next_ids))
            current_nside //= 2

        self.token_nside = current_nside if self.token_down > 0 else self.in_nside
        self.token_cell_ids = self.level_cell_ids[-1]

        # Token and fine-level operators (for convenience)
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

        # ---------------- Patch embedding (Cin -> C_fine) ----------------
        fine_dim = self.level_dims[0]
        fine_g = fine_dim // self.G
        self.patch_w = nn.Parameter(
            torch.empty(self.n_chan_in, fine_g, self.KERNELSZ * self.KERNELSZ)
        )
        nn.init.kaiming_uniform_(self.patch_w.view(self.n_chan_in * fine_g, -1), a=np.sqrt(5))
        self.patch_bn = nn.GroupNorm(num_groups=min(8, fine_dim if fine_dim > 1 else 1),
                                     num_channels=fine_dim)

        # ---------------- Encoder convs per level (C_i -> C_{i+1}) ----------------
        self.enc_w: nn.ParameterList = nn.ParameterList()
        self.enc_bn: nn.ModuleList = nn.ModuleList()
        for i in range(self.token_down):
            Cin = self.level_dims[i]
            Cout = self.level_dims[i+1]
            Cout_g = Cout // self.G
            w = nn.Parameter(torch.empty(Cin, Cout_g, self.KERNELSZ * self.KERNELSZ))
            nn.init.kaiming_uniform_(w.view(Cin * Cout_g, -1), a=np.sqrt(5))
            self.enc_w.append(w)
            self.enc_bn.append(nn.GroupNorm(num_groups=min(8, Cout if Cout > 1 else 1),
                                            num_channels=Cout))

        # ---------------- Transformer at token grid ----------------
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

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.embed_dim * self.mlp_ratio),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.depth)

        # Projection at token grid (keep C_token)
        self.token_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # ---------------- Decoder convs per level ( (C_{i+1}+C_i) -> C_i ) ----------------
        self.dec_w: nn.ParameterList = nn.ParameterList()
        self.dec_bn: nn.ModuleList = nn.ModuleList()
        for i in range(self.token_down - 1, -1, -1):
            # decoder proceeds from token level back to fine; we create weights in the same order
            Cin_fuse = self.level_dims[i+1] + self.level_dims[i]  # up + skip
            Cout = self.level_dims[i]
            Cout_g = Cout // self.G
            w = nn.Parameter(torch.empty(Cin_fuse, Cout_g, self.KERNELSZ * self.KERNELSZ))
            nn.init.kaiming_uniform_(w.view(Cin_fuse * Cout_g, -1), a=np.sqrt(5))
            self.dec_w.append(w)  # index 0 corresponds to up from token to level L-1
            self.dec_bn.append(nn.GroupNorm(num_groups=min(8, Cout if Cout > 1 else 1),
                                            num_channels=Cout))

        # ---------------- Final head (C_fine -> out_channels) ----------------
        if self.task == "global":
            self.global_head = nn.Linear(self.embed_dim, self.out_channels)
        else:
            if self.out_channels % self.G != 0:
                raise ValueError(f"out_channels={self.out_channels} must be divisible by G={self.G}")
            out_g = self.out_channels // self.G
            self.head_w = nn.Parameter(torch.empty(self.out_channels, out_g, self.KERNELSZ * self.KERNELSZ))
            nn.init.kaiming_uniform_(self.head_w.view(self.out_channels * out_g, -1), a=np.sqrt(5))
            self.head_bn = (nn.GroupNorm(num_groups=min(8, self.out_channels if self.out_channels > 1 else 1),
                                         num_channels=self.out_channels)
                            if self.task == "segmentation" else None)

        # ---------------- Device probe ----------------
        pref = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.runtime_device = self._probe_and_set_runtime_device(pref)

    # ---------------- device helpers ----------------
    def _move_hc(self, hc: ho.SphericalStencil, device: torch.device) -> None:
        for name, val in list(vars(hc).items()):
            try:
                if torch.is_tensor(val):
                    setattr(hc, name, val.to(device))
                elif isinstance(val, (list, tuple)) and val and torch.is_tensor(val[0]):
                    setattr(hc, name, type(val)([v.to(device) for v in val]))
            except Exception:
                pass

    @torch.no_grad()
    def _probe_and_set_runtime_device(self, preferred: torch.device) -> torch.device:
        if preferred.type == "cuda" and self.prefer_foscat_gpu:
            try:
                super().to(preferred)
                for hc in self.hconv_levels + [self.hconv_token, self.hconv_head]:
                    self._move_hc(hc, preferred)
                # dry run
                npix0 = int(self.cell_ids_fine.shape[0])
                x_try = torch.zeros(1, self.n_chan_in, npix0, device=preferred)
                hc0 = self.hconv_levels[0] if len(self.hconv_levels) > 0 else self.hconv_head
                y_try = hc0.Convol_torch(x_try, self.patch_w, cell_ids=self.cell_ids_fine)
                _ = y_try.sum().item()
                self._foscat_device = preferred
                return preferred
            except Exception as e:
                self._gpu_probe_error = repr(e)
        cpu = torch.device("cpu")
        super().to(cpu)
        for hc in self.hconv_levels + [self.hconv_token, self.hconv_head]:
            self._move_hc(hc, cpu)
        self._foscat_device = cpu
        return cpu

    # ---------------- helpers ----------------
    def _as_tensor_batch(self, x):
        if isinstance(x, list):
            if len(x) == 1:
                t = x[0]
                return t.unsqueeze(0) if t.dim() == 2 else t
            raise ValueError("Variable-length list not supported here; pass a tensor.")
        return x

    # ---------------- forward ----------------
    def forward(
        self,
        x: torch.Tensor,
        runtime_ids: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        x: (B, Cin, Nfine), nested ordering
        runtime_ids: optional fine-level ids to decode onto (defaults to training ids)
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.dim() != 3:
            raise ValueError("Input must be (B, Cin, Npix)")
        if x.shape[1] != self.n_chan_in:
            raise ValueError(f"Expected {self.n_chan_in} channels, got {x.shape[1]}")

        x = x.to(self.runtime_device)

        # -------- Patch embedding Cin -> C_fine --------
        hc_fine0 = self.hconv_levels[0] if len(self.hconv_levels) > 0 else self.hconv_head
        z = hc_fine0.Convol_torch(x, self.patch_w, cell_ids=self.cell_ids_fine)  # (B, C_fine, Nfine)
        if not torch.is_tensor(z):
            z = torch.as_tensor(z, device=self.runtime_device)
        z = self._as_tensor_batch(z)
        z = self.patch_bn(z)
        z = F.gelu(z)

        # -------- Encoder path: for each level i: EncConv(C_i->C_{i+1}) then Down() --------
        skips: List[torch.Tensor] = []
        ids_list: List[np.ndarray] = []

        l_data = z
        l_cell_ids = self.cell_ids_fine if runtime_ids is None else np.asarray(runtime_ids)
        current_nside = self.in_nside

        for i, hc in enumerate(self.hconv_levels):
            # save skip BEFORE going down (channels = C_i, grid = current level)
            skips.append(self._as_tensor_batch(l_data))
            ids_list.append(np.asarray(l_cell_ids))

            # conv to next channels C_{i+1} at same grid
            w_enc = self.enc_w[i]
            l_data = hc.Convol_torch(l_data, w_enc, cell_ids=l_cell_ids)  # (B, C_{i+1}, N_current)
            if not torch.is_tensor(l_data):
                l_data = torch.as_tensor(l_data, device=self.runtime_device)
            l_data = self._as_tensor_batch(l_data)
            l_data = self.enc_bn[i](l_data)
            l_data = F.gelu(l_data)

            # Down one level
            l_data, l_cell_ids = hc.Down(l_data, cell_ids=l_cell_ids, nside=current_nside, max_poll=True)
            l_data = self._as_tensor_batch(l_data)
            current_nside //= 2

        # We are now at token grid with channels = C_token
        x_tok = l_data              # (B, C_token, Ntok)
        token_ids = l_cell_ids      # ids at token level
        assert x_tok.shape[1] == self.embed_dim, "Token channels mismatch with embed_dim."

        # -------- Transformer on tokens --------
        seq = x_tok.permute(0, 2, 1)  # (B, Ntok, E)
        if self.cls_token_enabled:
            cls = self.cls_token.expand(seq.size(0), -1, -1)
            seq = torch.cat([cls, seq], dim=1)
        if self.pos_embed is not None:
            seq = seq + self.pos_embed[:, :seq.shape[1], :]

        seq = self.encoder(seq)       # (B, Ntok(+1), E)
        if self.cls_token_enabled:
            tokens = seq[:, 1:, :]    # drop cls for dense
        else:
            tokens = seq

        tok_feat = self.token_proj(tokens).permute(0, 2, 1)  # (B, C_token, Ntok)

        if self.task == "global":
            if self.head_type == "cls" and self.cls_token_enabled:
                cls_vec = seq[:, 0, :]
                return nn.Linear(self.embed_dim, self.out_channels).to(seq.device)(cls_vec)
            else:
                return nn.Linear(self.embed_dim, self.out_channels).to(seq.device)(tokens.mean(dim=1))

        # -------- Build runtime id chain (fine -> ... -> token) --------
        fine_ids_runtime = self.cell_ids_fine if runtime_ids is None else np.asarray(runtime_ids)
        ids_chain = [np.asarray(fine_ids_runtime)]
        nside_tmp = self.in_nside
        _dummy = self.f.backend.bk_cast(np.zeros((1, 1, ids_chain[0].shape[0]), dtype=self.np_dtype))
        for hc in self.hconv_levels:
            _dummy, _next = hc.Down(_dummy, cell_ids=ids_chain[-1], nside=nside_tmp, max_poll=True)
            ids_chain.append(self.f.backend.to_numpy(_next))
            nside_tmp //= 2

        tok_ids_np = token_ids if isinstance(token_ids, np.ndarray) else np.asarray(token_ids)
        assert tok_feat.shape[-1] == tok_ids_np.shape[0], "Token count mismatch."
        assert np.array_equal(tok_ids_np, ids_chain[-1]), "Token ids mismatch with runtime chain."

        # list of nsides at each encoder level (fine -> ... -> pre-token)
        nsides_levels = [self.in_nside // (2 ** k) for k in range(self.token_down)]

        # -------- Decoder: Up step-by-step with fusion conv --------
        y = tok_feat  # (B, C_token, Ntok)
        dec_idx = 0   # index in self.dec_w / self.dec_bn (built from token->fine order)
        for i in range(len(ids_chain)-1, 0, -1):
            coarse_ids = ids_chain[i]      # current y grid
            fine_ids   = ids_chain[i-1]    # target grid
            source_ns  = self.in_nside // (2 ** i)
            fine_ns    = self.in_nside // (2 ** (i-1))

            # choose operator for the target fine level
            if fine_ns == self.in_nside:
                op_fine = self.hconv_head
            else:
                idx = nsides_levels.index(fine_ns)
                op_fine = self.hconv_levels[idx]

            # Up one level
            y_up = op_fine.Up(y, cell_ids=coarse_ids, o_cell_ids=fine_ids, nside=source_ns)
            if not torch.is_tensor(y_up):
                y_up = torch.as_tensor(y_up, device=self.runtime_device)
            y_up = self._as_tensor_batch(y_up)  # (B, C_{i}, N_fine)

            # Skip at this level (channels = C_{i-1})
            skip_i = self._as_tensor_batch(skips[i-1]).to(y_up.device)
            assert np.array_equal(np.asarray(ids_list[i-1]), np.asarray(fine_ids)), "Skip ids misaligned."

            # Concat and fuse: (C_{i} + C_{i-1}) -> C_{i-1}
            y_cat = torch.cat([y_up, skip_i], dim=1)
            y = op_fine.Convol_torch(y_cat, self.dec_w[dec_idx], cell_ids=fine_ids)
            if not torch.is_tensor(y):
                y = torch.as_tensor(y, device=self.runtime_device)
            y = self._as_tensor_batch(y)
            y = self.dec_bn[dec_idx](y)
            y = F.gelu(y)
            if self.dropout > 0:
                y = F.dropout(y, p=self.dropout, training=self.training)
            dec_idx += 1

        # y is now (B, C_fine, Nfine)
        # -------- Final head to out_channels --------
        y = self.hconv_head.Convol_torch(y, self.head_w, cell_ids=fine_ids_runtime)
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

    @torch.no_grad()
    def predict(self, x: Union[torch.Tensor, np.ndarray], batch_size: int = 8) -> torch.Tensor:
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        outs = []
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size].to(self.runtime_device)
            outs.append(self.forward(xb))
        return torch.cat(outs, dim=0)


# -------------------------- Smoke test --------------------------
if __name__ == "__main__":
    # nside=4 → Npix=192, 2 down levels → token_nside=1
    in_nside = 4
    npix = 12 * in_nside * in_nside
    cell_ids = np.arange(npix, dtype=np.int64)

    B, Cin = 2, 3
    x = torch.randn(B, Cin, npix)

    # Channel widths per level (fine -> token), divisible by G=1 here
    level_dims = [64, 96, 128]

    model = HealpixViTVarLevels(
        in_nside=in_nside,
        n_chan_in=Cin,
        level_dims=level_dims,  # len=3 => token_down=2
        depth=2,
        num_heads=4,
        cell_ids=cell_ids,
        task="regression",
        out_channels=1,
        KERNELSZ=3,
        G=1,
        cls_token=False,
        dropout=0.1,
    ).eval()

    with torch.no_grad():
        y = model(x)
        print("Output:", y.shape)  # (B, Cout, Nfine)
