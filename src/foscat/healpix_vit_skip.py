# healpix_vit_skip.py
# HEALPix ViT U-Net with temporal encoders and Transformer-based skip fusion.
# - Multi-level HEALPix pyramid using Foscat.SphericalStencil
# - Per-level temporal encoding (sequence over T_in months) at encoder
# - Decoder uses cross-attention to fuse upsampled features with encoder skips
# - Double spherical convolution + GroupNorm + GELU at each encoder/decoder level

from __future__ import annotations
from typing import List, Optional, Literal
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import foscat.scat_cov as sc
import foscat.SphericalStencil as ho


class MLP(nn.Module):
    def __init__(self, d: int, hidden_mult: int = 4, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden_mult * d),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_mult * d, d),
            nn.Dropout(drop),
        )
    def forward(self, x):
        return self.net(x)


class HealpixViTSkip(nn.Module):
    def __init__(
        self,
        *,
        in_nside: int,
        n_chan_in: int,
        level_dims: List[int],
        depth_token: int,
        num_heads_token: int,
        cell_ids: np.ndarray,
        task: Literal["regression","segmentation","global"] = "regression",
        out_channels: int = 1,
        mlp_ratio_token: float = 4.0,
        KERNELSZ: int = 3,
        gauge_type: Literal["cosmo","phi"] = "cosmo",
        G: int = 1,
        prefer_foscat_gpu: bool = True,
        dropout: float = 0.1,
        dtype: Literal["float32","float64"] = "float32",
        pos_embed_per_level: bool = True,
    ) -> None:
        super().__init__()

        self.in_nside = int(in_nside)
        self.n_chan_in = int(n_chan_in)
        self.level_dims = list(level_dims)
        self.token_down = len(self.level_dims) - 1
        assert self.token_down >= 0
        self.C_fine = int(self.level_dims[0])
        self.embed_dim = int(self.level_dims[-1])
        self.depth_token = int(depth_token)
        self.num_heads_token = int(num_heads_token)
        self.mlp_ratio_token = float(mlp_ratio_token)
        self.task = task
        self.out_channels = int(out_channels)
        self.KERNELSZ = int(KERNELSZ)
        self.gauge_type = gauge_type
        self.G = int(G)
        self.prefer_foscat_gpu = bool(prefer_foscat_gpu)
        self.dropout = float(dropout)
        self.dtype = dtype
        self.pos_embed_per_level = bool(pos_embed_per_level)

        for d in self.level_dims:
            if d % self.G != 0:
                raise ValueError(f"All level_dims must be divisible by G={self.G}, got {d}.")
        if self.embed_dim % self.num_heads_token != 0:
            raise ValueError("embed_dim must be divisible by num_heads_token.")

        if dtype == "float32":
            self.np_dtype = np.float32
            self.torch_dtype = torch.float32
        else:
            self.np_dtype = np.float64
            self.torch_dtype = torch.float32

        if cell_ids is None:
            raise ValueError("cell_ids (finest) must be provided.")
        self.cell_ids_fine = np.asarray(cell_ids)

        if self.task == "segmentation":
            self.final_activation = "sigmoid" if self.out_channels == 1 else "softmax"
        else:
            self.final_activation = "none"

        self.f = sc.funct(KERNELSZ=self.KERNELSZ)

        # Build stencils
        self.hconv_levels: List[ho.SphericalStencil] = []
        self.level_cell_ids: List[np.ndarray] = [self.cell_ids_fine]
        current_nside = self.in_nside
        dummy = self.f.backend.bk_cast(np.zeros((1, 1, self.cell_ids_fine.shape[0]), dtype=self.np_dtype))
        for _ in range(self.token_down):
            hc = ho.SphericalStencil(current_nside, self.KERNELSZ, n_gauges=self.G,
                                     gauge_type=self.gauge_type, cell_ids=self.level_cell_ids[-1],
                                     dtype=self.torch_dtype)
            self.hconv_levels.append(hc)
            dummy, next_ids = hc.Down(dummy, cell_ids=self.level_cell_ids[-1], nside=current_nside, max_poll=True)
            self.level_cell_ids.append(self.f.backend.to_numpy(next_ids))
            current_nside //= 2

        self.token_nside = current_nside if self.token_down > 0 else self.in_nside
        self.token_cell_ids = self.level_cell_ids[-1]

        self.hconv_token = ho.SphericalStencil(self.token_nside, self.KERNELSZ, n_gauges=self.G,
                                               gauge_type=self.gauge_type, cell_ids=self.token_cell_ids, dtype=self.torch_dtype)
        self.hconv_head  = ho.SphericalStencil(self.in_nside, self.KERNELSZ, n_gauges=self.G,
                                               gauge_type=self.gauge_type, cell_ids=self.cell_ids_fine, dtype=self.torch_dtype)

        self.nsides_levels = [self.in_nside // (2**i) for i in range(self.token_down+1)]
        self.ntokens_levels = [12 * n * n for n in self.nsides_levels]

        # Patch embed (double conv)
        fine_g = self.C_fine // self.G
        self.pe_w1 = nn.Parameter(torch.empty(self.n_chan_in, fine_g, self.KERNELSZ*self.KERNELSZ))
        nn.init.kaiming_uniform_(self.pe_w1.view(self.n_chan_in * fine_g, -1), a=np.sqrt(5))
        self.pe_w2 = nn.Parameter(torch.empty(self.C_fine, fine_g, self.KERNELSZ*self.KERNELSZ))
        nn.init.kaiming_uniform_(self.pe_w2.view(self.C_fine * fine_g, -1), a=np.sqrt(5))
        self.pe_bn1 = nn.GroupNorm(num_groups=min(8, self.C_fine if self.C_fine>1 else 1), num_channels=self.C_fine)
        self.pe_bn2 = nn.GroupNorm(num_groups=min(8, self.C_fine if self.C_fine>1 else 1), num_channels=self.C_fine)

        # Encoder double convs
        self.enc_w1 = nn.ParameterList()
        self.enc_w2 = nn.ParameterList()
        self.enc_bn1 = nn.ModuleList()
        self.enc_bn2 = nn.ModuleList()
        for i in range(self.token_down):
            Cin = self.level_dims[i]
            Cout = self.level_dims[i+1]
            Cout_g = Cout // self.G
            w1 = nn.Parameter(torch.empty(Cin, Cout_g, self.KERNELSZ*self.KERNELSZ))
            nn.init.kaiming_uniform_(w1.view(Cin * Cout_g, -1), a=np.sqrt(5))
            w2 = nn.Parameter(torch.empty(Cout, Cout_g, self.KERNELSZ*self.KERNELSZ))
            nn.init.kaiming_uniform_(w2.view(Cout * Cout_g, -1), a=np.sqrt(5))
            self.enc_w1.append(w1); self.enc_w2.append(w2)
            self.enc_bn1.append(nn.GroupNorm(num_groups=min(8, Cout if Cout>1 else 1), num_channels=Cout))
            self.enc_bn2.append(nn.GroupNorm(num_groups=min(8, Cout if Cout>1 else 1), num_channels=Cout))

        # Temporal encoders per level (fine..pre-token)
        self.temporal_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.level_dims[i],
                    nhead=max(1, min(8, self.level_dims[i] // 64)),
                    dim_feedforward=2*self.level_dims[i],
                    dropout=self.dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=2,
            )
            for i in range(self.token_down)
        ])

        # Token-level Transformer
        self.n_tokens = int(self.token_cell_ids.shape[0])
        self.pos_token = nn.Parameter(torch.zeros(1, self.n_tokens, self.embed_dim))
        nn.init.trunc_normal_(self.pos_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads_token,
            dim_feedforward=int(self.embed_dim * self.mlp_ratio_token),
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder_token = nn.TransformerEncoder(enc_layer, num_layers=self.depth_token)

        # Decoder fusion modules per level (cross-attention)
        self.dec_q = nn.ModuleList()
        self.dec_k = nn.ModuleList()
        self.dec_v = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        self.dec_mlp = nn.ModuleList()
        self.level_pos = nn.ParameterList() if self.pos_embed_per_level else None
        for i in range(self.token_down, 0, -1):
            Cfine = self.level_dims[i-1]
            d_fuse = Cfine
            self.dec_q.append(nn.Linear(Cfine, d_fuse))
            self.dec_k.append(nn.Linear(Cfine, d_fuse))
            self.dec_v.append(nn.Linear(Cfine, d_fuse))
            self.dec_attn.append(nn.MultiheadAttention(embed_dim=d_fuse, num_heads=max(1, min(8, d_fuse // 64)), batch_first=True))
            self.dec_mlp.append(MLP(d_fuse, hidden_mult=4, drop=self.dropout))
            if self.pos_embed_per_level:
                n_tok_i = self.ntokens_levels[i-1]
                p = nn.Parameter(torch.zeros(1, n_tok_i, d_fuse))
                nn.init.trunc_normal_(p, std=0.02)
                self.level_pos.append(p)

        # Decoder refinement double convs
        self.dec_refine_w1 = nn.ParameterList()
        self.dec_refine_w2 = nn.ParameterList()
        self.dec_refine_bn1 = nn.ModuleList()
        self.dec_refine_bn2 = nn.ModuleList()
        for i in range(self.token_down, 0, -1):
            Cfine = self.level_dims[i-1]
            Cfine_g = Cfine // self.G
            w1 = nn.Parameter(torch.empty(Cfine, Cfine_g, self.KERNELSZ*self.KERNELSZ))
            nn.init.kaiming_uniform_(w1.view(Cfine * Cfine_g, -1), a=np.sqrt(5))
            w2 = nn.Parameter(torch.empty(Cfine, Cfine_g, self.KERNELSZ*self.KERNELSZ))
            nn.init.kaiming_uniform_(w2.view(Cfine * Cfine_g, -1), a=np.sqrt(5))
            self.dec_refine_w1.append(w1); self.dec_refine_w2.append(w2)
            self.dec_refine_bn1.append(nn.GroupNorm(num_groups=min(8, Cfine if Cfine>1 else 1), num_channels=Cfine))
            self.dec_refine_bn2.append(nn.GroupNorm(num_groups=min(8, Cfine if Cfine>1 else 1), num_channels=Cfine))

        # Head
        if self.task == "global":
            self.global_head = nn.Linear(self.embed_dim, self.out_channels)
        else:
            if self.out_channels % self.G != 0:
                raise ValueError(f"out_channels={self.out_channels} must be divisible by G={self.G}")
            out_g = self.out_channels // self.G
            self.head_w = nn.Parameter(torch.empty(self.C_fine, out_g, self.KERNELSZ*self.KERNELSZ))
            nn.init.kaiming_uniform_(self.head_w.view(self.C_fine * out_g, -1), a=np.sqrt(5))
            self.head_bn = nn.GroupNorm(num_groups=min(8, self.out_channels if self.out_channels>1 else 1),
                                        num_channels=self.out_channels) if self.task=="segmentation" else None

        pref = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.runtime_device = self._probe_and_set_runtime_device(pref)

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
        if preferred.type == "cuda":
            try:
                super().to(preferred)
                for hc in self.hconv_levels + [self.hconv_token, self.hconv_head]:
                    self._move_hc(hc, preferred)
                npix0 = int(self.cell_ids_fine.shape[0])
                x_try = torch.zeros(1, self.n_chan_in, npix0, device=preferred)
                hc0 = self.hconv_levels[0] if len(self.hconv_levels)>0 else self.hconv_head
                y_try = hc0.Convol_torch(x_try, self.pe_w1, cell_ids=self.cell_ids_fine)
                _ = (y_try if torch.is_tensor(y_try) else torch.as_tensor(y_try, device=preferred)).sum().item()
                self._foscat_device = preferred
                return preferred
            except Exception:
                pass
        cpu = torch.device("cpu")
        super().to(cpu)
        for hc in self.hconv_levels + [self.hconv_token, self.hconv_head]:
            self._move_hc(hc, cpu)
        self._foscat_device = cpu
        return cpu

    def _as_tensor_batch(self, x):
        if isinstance(x, list):
            if len(x) == 1:
                t = x[0]
                return t.unsqueeze(0) if t.dim() == 2 else t
            raise ValueError("Variable-length list not supported here; pass a tensor.")
        return x

    def _to_numpy_ids(self, ids):
        if torch.is_tensor(ids):
            return ids.detach().cpu().numpy()
        return np.asarray(ids)

    def _patch_embed_fine(self, x_t: torch.Tensor) -> torch.Tensor:
        hc0 = self.hconv_levels[0] if len(self.hconv_levels)>0 else self.hconv_head
        z = hc0.Convol_torch(x_t, self.pe_w1, cell_ids=self.cell_ids_fine)
        z = self._as_tensor_batch(z if torch.is_tensor(z) else torch.as_tensor(z, device=self.runtime_device))
        z = self.pe_bn1(z); z = F.gelu(z)
        z = hc0.Convol_torch(z, self.pe_w2, cell_ids=self.cell_ids_fine)
        z = self._as_tensor_batch(z if torch.is_tensor(z) else torch.as_tensor(z, device=self.runtime_device))
        z = self.pe_bn2(z); z = F.gelu(z)
        return z

    def forward(self, x: torch.Tensor, runtime_ids: Optional[np.ndarray] = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Expected input shape (B, T_in, C_in, Npix)")
        B, T_in, C_in, Nf = x.shape
        if C_in != self.n_chan_in:
            raise ValueError(f"Expected n_chan_in={self.n_chan_in}, got {C_in}")
        x = x.to(self.runtime_device)

        fine_ids_runtime = self.cell_ids_fine if runtime_ids is None else self._to_numpy_ids(runtime_ids)
        ids_chain = [np.asarray(fine_ids_runtime)]
        nside_tmp = self.in_nside
        _dummy = self.f.backend.bk_cast(np.zeros((1, 1, ids_chain[0].shape[0]), dtype=self.np_dtype))
        for hc in self.hconv_levels:
            _dummy, _next = hc.Down(_dummy, cell_ids=ids_chain[-1], nside=nside_tmp, max_poll=True)
            ids_chain.append(self.f.backend.to_numpy(_next))
            nside_tmp //= 2

        # Encoder histories per level
        l_hist: List[torch.Tensor] = []
        l_ids:  List[np.ndarray] = []

        feats_fine = []
        for t in range(T_in):
            zt = self._patch_embed_fine(x[:, t, :, :])
            feats_fine.append(zt.unsqueeze(1))
        feats_fine = torch.cat(feats_fine, dim=1)  # (B, T_in, C_fine, N_fine)
        l_hist.append(feats_fine)
        l_ids.append(self.cell_ids_fine)

        current_nside = self.in_nside
        l_data_hist = feats_fine
        for i, hc in enumerate(self.hconv_levels):
            Cin = self.level_dims[i]
            Cout = self.level_dims[i+1]
            w1, w2 = self.enc_w1[i], self.enc_w2[i]
            feats_next = []
            for t in range(T_in):
                zt = l_data_hist[:, t, :, :]
                zt = hc.Convol_torch(zt, w1, cell_ids=l_ids[-1])
                zt = self._as_tensor_batch(zt if torch.is_tensor(zt) else torch.as_tensor(zt, device=self.runtime_device))
                zt = self.enc_bn1[i](zt); zt = F.gelu(zt)
                zt = hc.Convol_torch(zt, w2, cell_ids=l_ids[-1])
                zt = self._as_tensor_batch(zt if torch.is_tensor(zt) else torch.as_tensor(zt, device=self.runtime_device))
                zt = self.enc_bn2[i](zt); zt = F.gelu(zt)
                feats_next.append(zt.unsqueeze(1))
            feats_next = torch.cat(feats_next, dim=1)  # (B, T_in, Cout, N_i)

            feats_down = []
            next_ids_list = None
            for t in range(T_in):
                zt, next_ids = hc.Down(feats_next[:, t, :, :], cell_ids=l_ids[-1], nside=current_nside, max_poll=True)
                zt = self._as_tensor_batch(zt)
                feats_down.append(zt.unsqueeze(1))
                next_ids_list = next_ids
            feats_down = torch.cat(feats_down, dim=1)  # (B, T_in, Cout, N_{i+1})

            l_hist.append(feats_down)
            l_ids.append(self.f.backend.to_numpy(next_ids_list))
            l_data_hist = feats_down
            current_nside //= 2

        # Temporal encoder on skips (levels 0..token_down-1)
        skips: List[torch.Tensor] = []
        for i in range(self.token_down):
            Bx, Tx, Cx, Nx = l_hist[i].shape
            z = l_hist[i].permute(0, 3, 1, 2).reshape(Bx*Nx, Tx, Cx)
            z = self.temporal_encoders[i](z)
            z = z.mean(dim=1)
            H_i = z.view(Bx, Nx, Cx).permute(0, 2, 1).contiguous()
            skips.append(H_i)

        # Token-level transformer (spatial)
        x_tok_hist = l_hist[-1]                       # (B, T_in, E, Ntok)
        x_tok = x_tok_hist.mean(dim=1)                # (B, E, Ntok)  (could add temporal encoder here as well)
        seq = x_tok.permute(0, 2, 1) + self.pos_token[:, :x_tok.shape[2], :]
        seq = self.encoder_token(seq)
        y = seq.permute(0, 2, 1)                      # (B, E, Ntok)

        if self.task == "global":
            g = seq.mean(dim=1)
            return self.global_head(g)

        # Decoder: Up + cross-attn fusion + double conv refinement
        dec_idx = 0
        for i in range(self.token_down, 0, -1):
            coarse_ids = ids_chain[i]
            fine_ids   = ids_chain[i-1]
            source_ns  = self.in_nside // (2 ** i)
            fine_ns    = self.in_nside // (2 ** (i-1))
            Cfine      = self.level_dims[i-1]

            op_fine = self.hconv_head if fine_ns == self.in_nside else self.hconv_levels[self.nsides_levels.index(fine_ns)]

            y_up = op_fine.Up(y, cell_ids=coarse_ids, o_cell_ids=fine_ids, nside=source_ns)
            y_up = self._as_tensor_batch(y_up if torch.is_tensor(y_up) else torch.as_tensor(y_up, device=self.runtime_device))  # (B, Cfine, N)

            skip_i = skips[i-1]  # (B, Cfine, N)
            q = self.dec_q[dec_idx](y_up.permute(0,2,1))
            k = self.dec_k[dec_idx](skip_i.permute(0,2,1))
            v = self.dec_v[dec_idx](skip_i.permute(0,2,1))
            if self.pos_embed_per_level:
                pos = self.level_pos[dec_idx][:, :q.shape[1], :]
                q = q + pos; k = k + pos
            z, _ = self.dec_attn[dec_idx](q, k, v)
            z = self.dec_mlp[dec_idx](z)
            z = z.permute(0,2,1).contiguous()  # (B, Cfine, N)

            z = op_fine.Convol_torch(z, self.dec_refine_w1[dec_idx], cell_ids=fine_ids)
            z = self._as_tensor_batch(z if torch.is_tensor(z) else torch.as_tensor(z, device=self.runtime_device))
            z = self.dec_refine_bn1[dec_idx](z); z = F.gelu(z)
            z = op_fine.Convol_torch(z, self.dec_refine_w2[dec_idx], cell_ids=fine_ids)
            z = self._as_tensor_batch(z if torch.is_tensor(z) else torch.as_tensor(z, device=self.runtime_device))
            z = self.dec_refine_bn2[dec_idx](z); z = F.gelu(z)

            y = z
            dec_idx += 1

        y = self.hconv_head.Convol_torch(y, self.head_w, cell_ids=fine_ids_runtime)
        y = self._as_tensor_batch(y if torch.is_tensor(y) else torch.as_tensor(y, device=self.runtime_device))
        if self.task == "segmentation" and self.head_bn is not None:
            y = self.head_bn(y)
        if self.final_activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.final_activation == "softmax":
            y = torch.softmax(y, dim=1)
        return y


if __name__ == "__main__":
    in_nside = 4
    npix = 12 * in_nside * in_nside
    cell_ids = np.arange(npix, dtype=np.int64)

    B, T_in, Cin = 2, 3, 4
    x = torch.randn(B, T_in, Cin, npix)

    model = HealpixViTSkip(
        in_nside=in_nside,
        n_chan_in=Cin,
        level_dims=[64, 96, 128],
        depth_token=2,
        num_heads_token=4,
        cell_ids=cell_ids,
        task="regression",
        out_channels=1,
        KERNELSZ=3,
        G=1,
        dropout=0.1,
    ).eval()

    with torch.no_grad():
        y = model(x)
    print("Output:", tuple(y.shape))
