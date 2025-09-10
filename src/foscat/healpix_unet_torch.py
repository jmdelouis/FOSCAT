"""
HEALPix U-Net (nested) with Foscat + PyTorch niceties
----------------------------------------------------
GPU by default (when available), with graceful CPU fallback if Foscat ops are CPU-only.

- ReLU + BatchNorm after each convolution (encoder & decoder)
- Segmentation/Regression heads with optional final activation
- PyTorch-ified: inherits from nn.Module, standard state_dict
- Device management: tries CUDA first; if Foscat SphericalStencil cannot run on CUDA, falls back to CPU

Shape convention: (B, C, Npix)

Requirements: foscat (scat_cov.funct + SphericalStencil.Convol_torch must be differentiable on torch tensors)
"""
from __future__ import annotations
from typing import List, Optional, Literal, Tuple
import numpy as np

import torch
import torch.nn as nn
import healpy as hp

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import foscat.scat_cov as sc
import foscat.SphericalStencil as ho
import matplotlib.pyplot as plt

class HealpixUNet(nn.Module):
    """U-Net-like architecture on the HEALPix sphere using Foscat oriented convolutions.

    Parameters
    ----------
    in_nside : int
        Input HEALPix nside (nested scheme).
    n_chan_in : int
        Number of input channels.
    chanlist : list[int]
        Channels per encoder level (depth = len(chanlist)). Example: [16, 32, 64].
    cell_ids : np.ndarray
        Cell indices for the finest resolution (nside = in_nside) in nested scheme.
    KERNELSZ : int, default 3
        Spatial kernel size K (K x K) for oriented convolution.
    gauge_type : str
        Type of gauge :
        'cosmo' use the same definition than
           https://www.aanda.org/articles/aa/abs/2022/12/aa44566-22/aa44566-22.html
        'phi' is define at the pole, could be better for earth observation not using intensivly the pole
    G : int, default 1
        Number of gauges for the orientation definition.
    task : {'regression','segmentation'}, default 'regression'
        Chooses the head and default activation.
    out_channels : int, default 1
        Number of output channels (e.g. num_classes for segmentation).
    final_activation : {'none','sigmoid','softmax'} | None
        If None, uses sensible default per task: 'none' for regression, 'softmax' for segmentation (multi-class),
        'sigmoid' for segmentation when out_channels==1.
    device : str | torch.device | None, default: 'cuda' if available else 'cpu'
        Preferred device. The module will probe whether Foscat ops can run on CUDA; if not,
        it will fallback to CPU and keep all parameters/buffers on CPU for consistency.
    down_type:
        {"mean","max"}, default "max". Equivalent of max poll during down  
    prefer_foscat_gpu : bool, default True
        When device is CUDA, try to move Foscat operators (internal tensors) to CUDA and do a dry-run.
        If the dry-run fails, everything falls back to CPU.

    Notes
    -----
    - Two oriented convolutions per level. After each conv: BatchNorm1d + ReLU.
    - Downsampling uses foscat ``ud_grade_2``; upsampling uses ``up_grade``.
    - Convolution kernels are explicit parameters (shape [C_in, C_out, K*K]) and applied via ``SphericalStencil.Convol_torch``.
    - Foscat ops device is auto-probed to avoid CPU/CUDA mismatches.
    """

    def __init__(
            self,
            *,
            in_nside: int,
            n_chan_in: int,
            chanlist: List[int],
            cell_ids: np.ndarray,
            KERNELSZ: int = 3,
            task: Literal['regression', 'segmentation'] = 'regression',
            out_channels: int = 1,
            final_activation: Optional[Literal['none', 'sigmoid', 'softmax']] = None,
            device: Optional[torch.device | str] = None,
            prefer_foscat_gpu: bool = True,
            gauge_type: Optional[Literal['cosmo','phi']] = 'cosmo',
            G: int =1,
            down_type: Optional[Literal['mean','max']] = 'max',  
            dtype: Literal['float32','float64'] = 'float32',
            head_reduce: Literal['mean','learned']='mean'
    ) -> None:
        super().__init__()
        
        self.dtype=dtype
        if dtype=='float32':
            self.np_dtype=np.float32
            self.torch_dtype=torch.float32
        else:
            self.np_dtype=np.float64
            self.torch_dtype=torch.float32

        self.gauge_type=gauge_type
        self.G = int(G)

        if self.G < 1:
            raise ValueError("G must be >= 1")
    
        if cell_ids is None:
            raise ValueError("cell_ids must be provided for the finest resolution.")
        if len(chanlist) == 0:
            raise ValueError("chanlist must be non-empty (depth >= 1).")

        self.in_nside = int(in_nside)
        self.n_chan_in = int(n_chan_in)
        self.chanlist = list(map(int, chanlist))
        self.chanlist = [self.chanlist[k]*self.G for k in range(len(self.chanlist))]
        self.KERNELSZ = int(KERNELSZ)
        self.task = task
        self.out_channels = int(out_channels)*self.G
        self.prefer_foscat_gpu = bool(prefer_foscat_gpu)
        if down_type == 'max':
            self.max_poll = True
        else:
            self.max_poll = False
        
        # Choose default final activation if not given
        if final_activation is None:
            if task == 'regression':
                self.final_activation = 'none'
            else:  # segmentation
                self.final_activation = 'sigmoid' if out_channels == 1 else 'softmax'
        else:
            self.final_activation = final_activation

        # Resolve preferred device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Foscat functional wrapper (backend + grade ops)
        self.f = sc.funct(KERNELSZ=self.KERNELSZ)

        # ---------- Build multi-resolution bookkeeping ----------
        depth = len(self.chanlist)
        self.l_cell_ids: List[np.ndarray] = [None] * (depth + 1)  # per encoder level + bottom
        self.l_cell_ids[0] = np.asarray(cell_ids)

        enc_nsides: List[int] = [self.in_nside]
        current_nside = self.in_nside

        # ---------- Oriented convolutions per level (encoder & decoder) ----------
        self.hconv_enc: List[ho.SphericalStencil] = []
        self.hconv_dec: List[ho.SphericalStencil] = []
        
        # dummy data to propagate shapes/ids through ud_grade_2
        l_data = self.f.backend.bk_cast(np.zeros((1, 1, cell_ids.shape[0]), dtype=self.np_dtype))

        for l in range(depth):
            # operator at encoder level l
            hc = ho.SphericalStencil(current_nside,
                                     self.KERNELSZ,
                                     n_gauges = self.G,
                                     gauge_type=self.gauge_type,
                                     cell_ids=self.l_cell_ids[l],
                                     dtype=self.torch_dtype)
            
            self.hconv_enc.append(hc)
            
            # downsample once to get next level ids and new data shape
            l_data, next_ids = hc.Down(
                l_data, cell_ids=self.l_cell_ids[l], nside=current_nside,max_poll=self.max_poll
            )
            self.l_cell_ids[l + 1] = self.f.backend.to_numpy(next_ids)
            current_nside //= 2
            enc_nsides.append(current_nside)

        # encoder conv weights and BN
        self.enc_w1 = nn.ParameterList()
        self.enc_bn1 = nn.ModuleList()
        self.enc_w2 = nn.ParameterList()
        self.enc_bn2 = nn.ModuleList()

        self.enc_nsides = enc_nsides  # [in, in/2, ..., in/2**depth]
        
        inC = self.n_chan_in
        for l, outC in enumerate(self.chanlist):
            if outC % self.G != 0:
                raise ValueError(f"chanlist[{l}] = {outC} must be divisible by G={self.G}")
            outC_g = outC // self.G

            # conv1: inC -> outC (via multi-gauge => noyau (Ci, Co_g, P))
            w1 = torch.empty(inC, outC_g, self.KERNELSZ * self.KERNELSZ)
            nn.init.kaiming_uniform_(w1.view(inC * outC_g, -1), a=np.sqrt(5))
            self.enc_w1.append(nn.Parameter(w1))
            self.enc_bn1.append(self._norm_1d(outC, kind="group"))

            # conv2: outC -> outC  (entrée = total outC ; noyau (outC, outC_g, P))
            w2 = torch.empty(outC, outC_g, self.KERNELSZ * self.KERNELSZ)
            nn.init.kaiming_uniform_(w2.view(outC * outC_g, -1), a=np.sqrt(5))
            self.enc_w2.append(nn.Parameter(w2))
            self.enc_bn2.append(self._norm_1d(outC, kind="group"))

            inC = outC  # next layer sees total channels

        # decoder conv weights and BN (mirrored levels)
        self.dec_w1 = nn.ParameterList()
        self.dec_bn1 = nn.ModuleList()
        self.dec_w2 = nn.ParameterList()
        self.dec_bn2 = nn.ModuleList()

        for d in range(depth):
            level = depth - 1 - d  # encoder level we are going back to
            hc = ho.SphericalStencil(self.enc_nsides[level],
                                     self.KERNELSZ,
                                     n_gauges = self.G,
                                     gauge_type=self.gauge_type,
                                     cell_ids=self.l_cell_ids[level],
                                     dtype=self.torch_dtype)
            #hc.make_idx_weights()
            self.hconv_dec.append(hc)

            upC = self.chanlist[level + 1] if level + 1 < depth else self.chanlist[level]
            skipC = self.chanlist[level]
            inC_dec  = upC + skipC           # total en entrée
            outC_dec = skipC                 # total en sortie (ce que tu avais déjà)

            if outC_dec % self.G != 0:
                raise ValueError(f"decoder outC at level {level} = {outC_dec} must be divisible by G={self.G}")
            outC_dec_g = outC_dec // self.G

            w1 = torch.empty(inC_dec, outC_dec_g, self.KERNELSZ * self.KERNELSZ)
            nn.init.kaiming_uniform_(w1.view(inC_dec * outC_dec_g, -1), a=np.sqrt(5))
            self.dec_w1.append(nn.Parameter(w1))
            self.dec_bn1.append(self._norm_1d(outC_dec, kind="group"))

            w2 = torch.empty(outC_dec, outC_dec_g, self.KERNELSZ * self.KERNELSZ)
            nn.init.kaiming_uniform_(w2.view(outC_dec * outC_dec_g, -1), a=np.sqrt(5))
            self.dec_w2.append(nn.Parameter(w2))
            self.dec_bn2.append(self._norm_1d(outC_dec, kind="group"))

        # Output head (on finest grid, channels = chanlist[0])
        self.head_hconv = ho.SphericalStencil(self.in_nside,
                                              self.KERNELSZ,
                                              n_gauges=self.G,   #Mandatory for the output
                                              gauge_type=self.gauge_type,
                                              cell_ids=self.l_cell_ids[0],
                                              dtype=self.torch_dtype)

        head_inC = self.chanlist[0]
        if self.out_channels % self.G != 0:
            raise ValueError(f"out_channels={self.out_channels} must be divisible by G={self.G}")
        outC_head_g = self.out_channels // self.G

        self.head_w = nn.Parameter(
            torch.empty(head_inC, outC_head_g, self.KERNELSZ * self.KERNELSZ)
        )
        nn.init.kaiming_uniform_(self.head_w.view(head_inC * outC_head_g, -1), a=np.sqrt(5))
        self.head_bn = self._norm_1d(self.out_channels, kind="group") if self.task == 'segmentation' else None
        
        # Choose how to reduce across gauges at head:
        # 'sum' (default), 'mean', or 'learned' (via 1x1 conv).
        self.head_reduce = getattr(self, 'head_reduce', 'mean')  # you can turn this into a ctor arg if you like
        if self.head_reduce == 'learned':
            # Mixer takes G*outC_head_g -> out_channels (K-wise 1x1)
            self.head_mixer = nn.Conv1d(self.G * outC_head_g, self.out_channels, kernel_size=1, bias=True)
        else:
            self.head_mixer = None
    
        # ---- Decide runtime device (probe Foscat on CUDA, else CPU) ----
        self.runtime_device = self._probe_and_set_runtime_device(self.device)

    # -------------------------- define local batchnorm/group -------------------
    def _norm_1d(self, C: int, kind: str = "group", **kwargs) -> nn.Module:
        """
        Return a normalization layer for (B, C, N) tensors.
        kind: "group" | "instance" | "batch"
        kwargs: extra args (e.g., num_groups for GroupNorm)
        """
        if kind == "group":
            num_groups = kwargs.get("num_groups", min(8, max(1, C // 8)) or 1)
            # s’assurer que num_groups divise C
            while C % num_groups != 0 and num_groups > 1:
                num_groups //= 2
            return nn.GroupNorm(num_groups=num_groups, num_channels=C)
        elif kind == "instance":
            return nn.InstanceNorm1d(C, affine=True, track_running_stats=False)
        elif kind == "batch":
            return nn.BatchNorm1d(C)
        else:
            raise ValueError(f"Unknown norm kind: {kind}")
            
    # -------------------------- device plumbing --------------------------
    def _move_hconv_tensors(self, hc: ho.SphericalStencil, device: torch.device) -> None:
        """Best-effort: move any torch.Tensor attribute of SphericalStencil to device."""
        for name, val in list(vars(hc).items()):
            try:
                if torch.is_tensor(val):
                    setattr(hc, name, val.to(device))
                elif isinstance(val, (list, tuple)) and val and torch.is_tensor(val[0]):
                    setattr(hc, name, type(val)([v.to(device) for v in val]))
            except Exception:
                # silently ignore non-tensor or protected attributes
                pass

    @torch.no_grad()
    def _probe_and_set_runtime_device(self, preferred: torch.device) -> torch.device:
        """Try to run a tiny Foscat conv on preferred device; fallback to CPU if it fails."""
        if preferred.type == 'cuda' and self.prefer_foscat_gpu:
            try:
                # move module params/buffers first
                super().to(preferred)
                # move Foscat operator internals
                for hc in self.hconv_enc + self.hconv_dec + [self.head_hconv]:
                    self._move_hconv_tensors(hc, preferred)
                # dry run on level 0
                npix0 = int(len(self.l_cell_ids[0]))
                x_try = torch.zeros(1, self.n_chan_in, npix0, device=preferred)
                y_try = self.hconv_enc[0].Convol_torch(x_try, self.enc_w1[0])
                # success -> stay on CUDA
                self._foscat_device = preferred
                return preferred
            except Exception as e:
                # fallback to CPU; keep error for info
                self._gpu_probe_error = repr(e)
                pass
        # CPU fallback
        cpu = torch.device('cpu')
        super().to(cpu)
        for hc in self.hconv_enc + self.hconv_dec + [self.head_hconv]:
            self._move_hconv_tensors(hc, cpu)
        self._foscat_device = cpu
        return cpu

    def set_device(self, device: torch.device | str) -> torch.device:
        """Request a (re)device; will probe Foscat and return the actual runtime device used."""
        device = torch.device(device)
        self.device = device
        self.runtime_device = self._probe_and_set_runtime_device(device)
        return self.runtime_device
    
    # --- inside HealpixUNet class, add a single-sample forward helper ---
    def _forward_one(self, x1: torch.Tensor, cell_ids1=None) -> torch.Tensor:
        """
        Single-sample forward. x1: (1, C_in, Npix_1). Returns (1, out_channels, Npix_1).
        `cell_ids1` can be None or a 1D array (Npix_1,) for this sample.
        """
        if x1.dim() != 3 or x1.shape[0] != 1:
            raise ValueError(f"_forward_one expects (1, C, Npix), got {tuple(x1.shape)}")
        # Reuse existing forward by calling it with B=1 (your code already supports per-sample ids)
        if cell_ids1 is None:
            return super().forward(x1)
        else:
            # normalize ids to numpy 1D
            if isinstance(cell_ids1, torch.Tensor):
                cell_ids1 = cell_ids1.detach().cpu().numpy()
            elif isinstance(cell_ids1, list):
                cell_ids1 = np.asarray(cell_ids1)
            if cell_ids1.ndim == 1:
                ci = cell_ids1[None, :]  # (1, Npix_1) so the current code path is happy
            else:
                ci = cell_ids1
            return super().forward(x1, cell_ids=ci)
        
    def _as_tensor_batch(self, x):
        """
        Ensure a (B, C, N) tensor.
        - If x is a list of tensors, concatenate if all N are equal.
        - If len==1, keep a batch dim (1, C, N).
        - If x is already a tensor, return as-is.
        """
        if isinstance(x, list):
            if len(x) == 1:
                t = x[0]
                # If t is (C, N) -> make it (1, C, N)
                return t.unsqueeze(0) if t.dim() == 2 else t
            # all same length -> concat along batch
            Ns = [t.shape[-1] for t in x]
            if all(n == Ns[0] for n in Ns):
                return torch.cat([t if t.dim() == 3 else t.unsqueeze(0) for t in x], dim=0)
            # variable-length with B>1 not supported in a single tensor
            raise ValueError("Variable-length batch detected; use batch_size=1 or loop per-sample.")
        return x
    
    # --- replace your current `forward` signature/body with a dispatcher ---
    def forward_any(self, x, cell_ids: Optional[np.ndarray] = None):
        """
        If `x` is a Tensor (B,C,N): standard batched path (requires same N for all).
        If `x` is a list of Tensors: variable-length per-sample path, returns a list of outputs.
        """
        # Variable-length list path
        if isinstance(x, (list, tuple)):
            outs = []
            if cell_ids is None or isinstance(cell_ids, (list, tuple)):
                cids = cell_ids if isinstance(cell_ids, (list, tuple)) else [None] * len(x)
            else:
                raise ValueError("When x is a list, cell_ids must be a list of same length or None.")

            for xb, cb in zip(x, cids):
                if not torch.is_tensor(xb):
                    xb = torch.as_tensor(xb, dtype=torch.float32, device=self.runtime_device)
                if xb.dim() == 2:
                    xb = xb.unsqueeze(0)  # (1,C,Nb)
                elif xb.dim() != 3 or xb.shape[0] != 1:
                    raise ValueError(f"Each sample must be (C,N) or (1,C,N); got {tuple(xb.shape)}")

                yb = self._forward_one(xb.to(self.runtime_device), cell_ids1=cb)  # (1,Co,Nb)
                outs.append(yb.squeeze(0))  # -> (Co, Nb)
            return outs  # List[Tensor] (each length Nb)

        # Fixed-length tensor path (your current implementation)
        return super().forward(x, cell_ids=cell_ids)

    # -------------------------- forward --------------------------
    def forward(self, x: torch.Tensor,cell_ids: Optional[np.ndarray ] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C_in, Npix)
            Input tensor on `in_nside` grid.
        cell_ids : np.ndarray (B, Npix) optional, use another cell_ids than the initial one.
                   if None use the initial cell_ids.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.dim() != 3:
            raise ValueError("Input must be (B, C, Npix)")

        # Ensure input lives on the runtime (probed) device
        x = x.to(self.runtime_device)

        B, C, N = x.shape
        if C != self.n_chan_in:
            raise ValueError(f"Expected {self.n_chan_in} input channels, got {C}")

        # Encoder
        skips: List[torch.Tensor] = []
        l_data = x
        current_nside = self.in_nside
        l_cell_ids=cell_ids
        
        if cell_ids is not None:
            t_cell_ids={}
            t_cell_ids[0]=l_cell_ids
        else:
            t_cell_ids=self.l_cell_ids
            
        for l, outC in enumerate(self.chanlist):
            # conv1 + BN + ReLU
            l_data = self.hconv_enc[l].Convol_torch(l_data,
                                                    self.enc_w1[l],
                                                    cell_ids=l_cell_ids)
            l_data = self._as_tensor_batch(l_data)
            l_data = self.enc_bn1[l](l_data)
            l_data = F.relu(l_data, inplace=True)

            # conv2 + BN + ReLU
            l_data = self.hconv_enc[l].Convol_torch(l_data,
                                                    self.enc_w2[l],
                                                    cell_ids=l_cell_ids)
            l_data = self._as_tensor_batch(l_data)
            l_data = self.enc_bn2[l](l_data)
            l_data = F.relu(l_data, inplace=True)

            # save skip at this resolution
            skips.append(l_data)

            # downsample (except bottom level) -> ensure output is on runtime_device
            if l < len(self.chanlist) - 1:
                l_data, l_cell_ids = self.hconv_enc[l].Down(
                    l_data, cell_ids=t_cell_ids[l], nside=current_nside,max_poll=self.max_poll
                )
                l_data = self._as_tensor_batch(l_data)
                if cell_ids is not None:
                    t_cell_ids[l+1]=l_cell_ids
                else:
                    l_cell_ids=None
                    
                if isinstance(l_data, torch.Tensor) and l_data.device != self.runtime_device:
                    l_data = l_data.to(self.runtime_device)
                current_nside //= 2
                
        # Decoder
        for d in range(len(self.chanlist)):
            level = len(self.chanlist) - 1 - d  # encoder level we are going back to

            if level < len(self.chanlist) - 1:
                # upsample: from encoder level (level+1) [coarser] -> level [finer]
                src_nside = self.enc_nsides[level + 1]    # coarse

                # Use the **decoder** operator at this step (consistent with your hconv_dec stack)
                l_data = self.hconv_dec[d].Up(
                    l_data,
                    cell_ids=t_cell_ids[level + 1],   # source/coarse IDs
                    o_cell_ids=t_cell_ids[level],     # target/fine IDs
                    nside=src_nside,
                )
                l_data = self._as_tensor_batch(l_data)

                if isinstance(l_data, torch.Tensor) and l_data.device != self.runtime_device:
                    l_data = l_data.to(self.runtime_device)

            # concat with skip features at this resolution
            concat = self.f.backend.bk_concat([skips[level], l_data], 1)
            l_data = concat.to(self.runtime_device) if torch.is_tensor(concat) else concat

            # choose the right cell_ids for convolutions at this resolution
            l_cell_ids = t_cell_ids[level] if (cell_ids is not None) else None

            # apply decoder convs on this grid using the matching decoder operator
            hc = self.hconv_dec[d]
            l_data = hc.Convol_torch(l_data, self.dec_w1[d], cell_ids=l_cell_ids)
            l_data = self._as_tensor_batch(l_data)
            l_data = self.dec_bn1[d](l_data)
            l_data = F.relu(l_data, inplace=True)

            l_data = hc.Convol_torch(l_data, self.dec_w2[d], cell_ids=l_cell_ids)
            l_data = self._as_tensor_batch(l_data)
            l_data = self.dec_bn2[d](l_data)
            l_data = F.relu(l_data, inplace=True)

        # Head on finest grid
        # y_head_raw: (B, G*outC_head_g, K)
        y_head_raw = self.head_hconv.Convol_torch(l_data, self.head_w, cell_ids=l_cell_ids)

        B, Ctot, K = y_head_raw.shape
        outC_head_g = int(self.out_channels)//self.G
        assert Ctot == self.G * outC_head_g, \
            f"Head expects G*outC_head_g channels, got {Ctot} != {self.G}*{outC_head_g}"

        if self.head_mixer is not None and self.head_reduce == 'learned':
            # 1x1 learned mixing across G*outC_head_g -> out_channels
            y = self.head_mixer(y_head_raw)  # (B, out_channels, K)
        else:
            # reshape to (B, G, outC_head_g, K) then reduce across G
            y_g = y_head_raw.view(B, self.G, outC_head_g, K)

            y = y_g.mean(dim=1)   # (B, outC_head_g, K)

        y = self._as_tensor_batch(y)
        
        # Optional BN + activation as before
        if self.task == 'segmentation' and self.head_bn is not None:
            y = self.head_bn(y)
            
        if self.final_activation == 'sigmoid':
            y = torch.sigmoid(y)
            
        elif self.final_activation == 'softmax':
            y = torch.softmax(y, dim=1)
            
        return y

    # -------------------------- utilities --------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor, batch_size: int = 8,cell_ids: Optional[np.ndarray ] = None) -> torch.Tensor:
        self.eval()
        outs = []
        if isinstance(x,np.ndarray):
            x=self.to_Tensor(x)
            
        if not isinstance(x, torch.Tensor):
            for i in range(len(x)):
                if cell_ids is not None:
                    outs.append(self.forward(x[i][None,:],cell_ids=cell_ids[i][:]))
                else:
                    outs.append(self.forward(x[i][None,:]))
        else:
            for i in range(0, x.shape[0], batch_size):
                if cell_ids is not None:
                    outs.append(self.forward(x[i : i + batch_size],
                                             cell_ids=cell_ids[i : i + batch_size]))
                else:
                    outs.append(self.forward(x[i : i + batch_size]))
                
        return torch.cat(outs, dim=0)

    def to_tensor(self,x):
        return self.hconv_enc[0].f.backend.bk_cast(x)
    
    def to_numpy(self,x):
        if isinstance(x,np.ndarray):
            return x
        return x.cpu().numpy()
    
    # -----------------------------
    # Kernel extraction & plotting
    # -----------------------------
    def _arch_shapes(self):
        """Return expected (in_c, out_c) per conv for encoder/decoder.

        Returns
        -------
        enc_shapes : list[tuple[tuple[int,int], tuple[int,int]]]
            For each level `l`, ((in1, out1), (in2, out2)) for the two encoder convs.
        dec_shapes : list[tuple[tuple[int,int], tuple[int,int]]]
            For each level `l`, ((in1, out1), (in2, out2)) for the two decoder convs.
        """
        nlayer = len(self.chanlist)
        enc_shapes = []
        l_chan = self.n_chan_in
        for l in range(nlayer):
            enc_shapes.append(((l_chan, self.chanlist[l]), (self.chanlist[l], self.chanlist[l])))
            l_chan = self.chanlist[l] + 1

        dec_shapes = []
        l_chan = self.chanlist[-1] + 1
        for l in range(nlayer):
            in1 = l_chan + 1
            out2 = 1 + (self.chanlist[nlayer - 1 - l] if (nlayer - 1 - l) > 0 else 0)
            dec_shapes.append(((in1, in1), (in1, out2)))
            l_chan = out2
        return enc_shapes, dec_shapes

    def extract_kernels(self, stage: str = "encoder", layer: int = 0, conv: int = 0):
        """Extract raw convolution kernels for a given stage/level/conv.

        Parameters
        ----------
        stage : {"encoder", "decoder"}
            Which part of the network to inspect.
        layer : int
            Pyramid level (0 = finest encoder level / bottommost decoder level).
        conv : int
            0 for the first conv at that level, 1 for the second conv.

        Returns
        -------
        np.ndarray
            Array of shape (in_c, out_c, K, K) containing the spatial kernels.
        """
        assert stage in {"encoder", "decoder"}
        assert conv in {0, 1}
        K = self.KERNELSZ
        enc_shapes, dec_shapes = self._arch_shapes()

        if stage == "encoder":
            if conv==0:
                w = self.enc_w1[layer]
            else:
                w = self.enc_w2[layer]
        else:
            if conv==0:
                w = self.dec_w1[layer]
            else:
                w = self.dec_w2[layer]

        w_np = self.f.backend.to_numpy(w.detach())
        return w_np.reshape(w.shape[0],w.shape[1],K,K)

    def plot_kernels(
        self,
        stage: str = "encoder",
        layer: int = 0,
        conv: int = 0,
        fixed: str = "in",
        index: int = 0,
        max_tiles: int = 16,
    ):
        """Quick visualization of kernels on a grid using matplotlib.

        Parameters
        ----------
        stage : {"encoder", "decoder"}
            Which tower to visualize.
        layer : int
            Level to visualize.
        conv : int
            0 or 1: first or second conv in the level.
        fixed : {"in", "out"}
            If "in", show kernels for a fixed input channel across many outputs.
            If "out", show kernels for a fixed output channel across many inputs.
        index : int
            Channel index to fix (according to `fixed`).
        max_tiles : int
            Maximum number of tiles to display.
        """
        import math
        import matplotlib.pyplot as plt

        W = self.extract_kernels(stage=stage, layer=layer, conv=conv)
        ic, oc, K,_ = W.shape

        if fixed == "in":
            idx = min(index, ic - 1)
            tiles = [W[idx, j] for j in range(oc)]
            title = f"{stage} L{layer} C{conv} | in={idx}"
        else:
            idx = min(index, oc - 1)
            tiles = [W[i, idx] for i in range(ic)]
            title = f"{stage} L{layer} C{conv} | out={idx}"

        tiles = tiles[:max_tiles]
        n = len(tiles)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))

        plt.figure(figsize=(2.5 * cols, 2.5 * rows))
        for i, ker in enumerate(tiles, 1):
            ax = plt.subplot(rows, cols, i)
            ax.imshow(ker)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

# -----------------------------
# Unit tests (smoke tests)
# -----------------------------
# Run with:  python UNET.py  (or)  python UNET.py -q  for quieter output
# These tests assume Foscat and its dependencies are installed.


def _dummy_cell_ids(nside: int) -> np.ndarray:
    """Return a simple identity mapping for HEALPix nested pixel IDs.

    Notes
    -----
    Replace with your pipeline's real `cell_ids` if you have a precomputed
    mapping consistent with Foscat/HEALPix nested ordering.
    """
    return np.arange(12 * nside * nside, dtype=np.int64)


if __name__ == "__main__":
    import unittest

    class TestUNET(unittest.TestCase):
        """Lightweight smoke tests for shape and parameter plumbing."""

        def setUp(self):
            self.nside = 4  # small grid for fast tests (npix = 192)
            self.chanlist = [4, 8]  # two-level encoder/decoder
            self.batch = 2
            self.channels = 1
            self.npix = 12 * self.nside * self.nside
            self.cell_ids = _dummy_cell_ids(self.nside)
            self.net = UNET(
                in_nside=self.nside,
                n_chan_in=self.channels,
                chanlist=self.chanlist,
                cell_ids=self.cell_ids,
            )

        def test_forward_shape(self):
            # random input
            x = np.random.randn(self.batch, self.channels, self.npix).astype(self.np_dtype)
            x = self.net.f.backend.bk_cast(x)
            y = self.net.eval(x)
            # expected output: same npix, 1 channel at the very top
            self.assertEqual(y.shape[0], self.batch)
            self.assertEqual(y.shape[1], 1)
            self.assertEqual(y.shape[2], self.npix)
            # sanity: no NaNs
            y_np = self.net.f.backend.to_numpy(y)
            self.assertFalse(np.isnan(y_np).any())

        def test_param_roundtrip_and_determinism(self):
            x = np.random.randn(self.batch, self.channels, self.npix).astype(self.np_dtype)
            x = self.net.f.backend.bk_cast(x)

            # forward twice -> identical outputs with fixed params
            y1 = self.net.eval(x)
            y2 = self.net.eval(x)
            y1_np = self.net.f.backend.to_numpy(y1)
            y2_np = self.net.f.backend.to_numpy(y2)
            np.testing.assert_allclose(y1_np, y2_np, rtol=0, atol=0)

            # perturb parameters -> output should (very likely) change
            p = self.net.get_param()
            p_np = self.net.f.backend.to_numpy(p).copy()
            if p_np.size > 0:
                p_np[0] += 1.0
                self.net.set_param(p_np)
                y3 = self.net.eval(x)
                y3_np = self.net.f.backend.to_numpy(y3)
                with self.assertRaises(AssertionError):
                    np.testing.assert_allclose(y1_np, y3_np, rtol=0, atol=0)

    unittest.main()
    
from torch.utils.data import Dataset
# 1) Dataset that omits cell_ids when None
from torch.utils.data import Dataset, DataLoader, TensorDataset

class HealpixDataset(Dataset):
    """
    Returns (x, y, cell_ids) per-sample if cell_ids is given, else (x, y).
    Shapes:
      x: (C, Npix)
      y: (C_out or 1, Npix)
      cell_ids: (Npix,) per-sample (or broadcasted from (Npix,))
    """
    def __init__(self, x, y, cell_ids=None, dtype=torch.float32):
        self.x = torch.as_tensor(x, dtype=dtype)
        self.y = torch.as_tensor(y, dtype=dtype)
        assert self.x.shape[0] == self.y.shape[0], "x and y must share batch size"
        self._has_cids = cell_ids is not None
        if self._has_cids:
            cid = torch.as_tensor(cell_ids, dtype=torch.long)
            if cid.dim() == 1:
                cid = cid.unsqueeze(0).expand(self.x.shape[0], -1)
            assert cid.shape[0] == self.x.shape[0], "cell_ids must match batch size"
            self.cids = cid
        else:
            self.cids = None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        if self._has_cids:
            return self.x[i], self.y[i], self.cids[i]
        else:
            return self.x[i], self.y[i]

# ---------------------------
# Datasets / Collate helpers
# ---------------------------

class HealpixDataset(Dataset):
    """
    Fixed-grid dataset (common Npix for all samples).
    Returns (x, y) if cell_ids is None, else (x, y, cell_ids).

    x: (B, C, Npix)
    y: (B, C_out or 1, Npix) or class indices depending on task
    cell_ids: (Npix,) or (B, Npix)
    """
    def __init__(self,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 cell_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 dtype: torch.dtype = torch.float32):
        x = torch.as_tensor(x, dtype=dtype)
        y = torch.as_tensor(y, dtype=dtype if y.ndim == 3 else torch.long)
        assert x.shape[0] == y.shape[0], "x and y must share batch size"
        self.x, self.y = x, y
        self._has_cids = cell_ids is not None
        if self._has_cids:
            c = torch.as_tensor(cell_ids, dtype=torch.long)
            if c.ndim == 1:  # broadcast single (Npix,) to (B, Npix)
                c = c.unsqueeze(0).expand(x.shape[0], -1)
            assert c.shape == (x.shape[0], x.shape[2]), "cell_ids must be (B,Npix) or (Npix,)"
            self.cids = c
        else:
            self.cids = None

    def __len__(self) -> int: return self.x.shape[0]

    def __getitem__(self, i: int):
        if self._has_cids:
            return self.x[i], self.y[i], self.cids[i]
        return self.x[i], self.y[i]

# ---------------------------
# Datasets / Collate helpers
# ---------------------------

class HealpixDataset(Dataset):
    """
    Fixed-grid dataset (common Npix for all samples).
    Returns (x, y) if cell_ids is None, else (x, y, cell_ids).

    x: (B, C, Npix)
    y: (B, C_out or 1, Npix) or class indices depending on task
    cell_ids: (Npix,) or (B, Npix)
    """
    def __init__(self,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 cell_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 dtype: torch.dtype = torch.float32):
        x = torch.as_tensor(x, dtype=dtype)
        y = torch.as_tensor(y, dtype=dtype if y.ndim == 3 else torch.long)
        assert x.shape[0] == y.shape[0], "x and y must share batch size"
        self.x, self.y = x, y
        self._has_cids = cell_ids is not None
        if self._has_cids:
            c = torch.as_tensor(cell_ids, dtype=torch.long)
            if c.ndim == 1:  # broadcast single (Npix,) to (B, Npix)
                c = c.unsqueeze(0).expand(x.shape[0], -1)
            assert c.shape == (x.shape[0], x.shape[2]), "cell_ids must be (B,Npix) or (Npix,)"
            self.cids = c
        else:
            self.cids = None

    def __len__(self) -> int: return self.x.shape[0]

    def __getitem__(self, i: int):
        if self._has_cids:
            return self.x[i], self.y[i], self.cids[i]
        return self.x[i], self.y[i]


class VarLenHealpixDataset(Dataset):
    """
    Variable-length per-sample dataset.

    x_list[b]: (C, Npix_b) or (1, C, Npix_b)
    y_list[b]: (C_out or 1, Npix_b) or (1, C_out, Npix_b)  (regression/segmentation targets)
               For multi-class segmentation with CrossEntropyLoss, you may pass
               class indices of shape (Npix_b,) or (1, Npix_b) (we’ll squeeze later).
    cids_list[b]: (Npix_b,) or None
    """
    def __init__(self,
                 x_list: List[Union[np.ndarray, torch.Tensor]],
                 y_list: List[Union[np.ndarray, torch.Tensor]],
                 cids_list: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
                 dtype: torch.dtype = torch.float32):
        assert len(x_list) == len(y_list), "x_list and y_list must have the same length"
        self.x = [torch.as_tensor(x, dtype=dtype) for x in x_list]
        # y can be float (regression) or long (class indices); we’ll coerce later per task
        self.y = [torch.as_tensor(y) for y in y_list]
        if cids_list is not None:
            assert len(cids_list) == len(x_list), "cids_list must match x_list length"
            self.c = [torch.as_tensor(c, dtype=torch.long) for c in cids_list]
        else:
            self.c = None

    def __len__(self) -> int: return len(self.x)

    def __getitem__(self, i: int):
        ci = None if self.c is None else self.c[i]
        return self.x[i], self.y[i], ci

from torch.utils.data import Dataset, DataLoader

class VarLenHealpixDataset(Dataset):
    """
    x_list: list of (C, Npix_b) tensors or arrays
    y_list: list of (C_out or 1, Npix_b) tensors or arrays
    cids_list: optional list of (Npix_b,) arrays
    """
    def __init__(self, x_list, y_list, cids_list=None, dtype=torch.float32):
        assert len(x_list) == len(y_list)
        self.x = [torch.as_tensor(x, dtype=dtype) for x in x_list]
        self.y = [torch.as_tensor(y, dtype=dtype) for y in y_list]
        self.c = None
        if cids_list is not None:
            assert len(cids_list) == len(x_list)
            self.c = [np.asarray(c) for c in cids_list]

    def __len__(self): return len(self.x)

    def __getitem__(self, i):
        if self.c is None:
            return self.x[i], self.y[i], None
        return self.x[i], self.y[i], self.c[i]

def varlen_collate(batch):
    # Just return lists; do not stack.
    xs, ys, cs = zip(*batch)  # tuples of length B
    # keep None if all Nones, else list
    c_out = None if all(c is None for c in cs) else list(cs)
    return list(xs), list(ys), c_out

def varlen_collate(batch):
    """
    Collate for variable-length samples: keep lists, do NOT stack.
    Returns lists: xs, ys, cs (cs can be None).
    """
    xs, ys, cs = zip(*batch)
    c_out = None if all(c is None for c in cs) else list(cs)
    return list(xs), list(ys), c_out


# ---------------------------
# Training function
# ---------------------------

def fit(
        model,
        x_train: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
        y_train: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
        *,
        cell_ids_train: Optional[Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]] = None,
        n_epoch: int = 10,
        view_epoch: int = 10,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        clip_grad_norm: Optional[float] = None,
        verbose: bool = True,
        optimizer: Literal['ADAM', 'LBFGS'] = 'ADAM',
) -> dict:
    """
    Train helper that supports:
      - Fixed-grid tensors (B,C,N) with optional (B,N) or (N,) cell_ids.
      - Variable-length lists: x=[(C,N_b)], y=[...], cell_ids=[(N_b,)], returning per-sample grids.

    ADAM: standard minibatch update.
    LBFGS: uses a closure that sums losses over the current (variable-length) mini-batch.

    Notes
    -----
    - For segmentation with multiple classes, pass integer class targets for y:
      fixed-grid: (B, N) int64; variable-length: each y[b] of shape (N_b,) or (1,N_b).
    - For regression, pass float targets with the same (C_out, N) channeling.
    """
    device = model.runtime_device if hasattr(model, "runtime_device") else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)

    # Detect variable-length mode
    varlen_mode = isinstance(x_train, (list, tuple))

    # ----- Build DataLoader
    if not varlen_mode:
        # Fixed-grid path
        x_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
        y_is_class = (model.task != 'regression' and getattr(model, "out_channels", 1) > 1)
        y_dtype = torch.long if y_is_class and (not torch.is_tensor(y_train) or y_train.ndim != 3) else torch.float32
        y_t = torch.as_tensor(y_train, dtype=y_dtype, device=device)

        if cell_ids_train is None:
            ds = TensorDataset(x_t, y_t)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
            with_cell_ids = False
        else:
            ds = HealpixDataset(x_t, y_t, cell_ids=cell_ids_train, dtype=torch.float32)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
            with_cell_ids = True
    else:
        # Variable-length path
        ds = VarLenHealpixDataset(x_train, y_train, cids_list=cell_ids_train, dtype=torch.float32)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=varlen_collate)
        with_cell_ids = cell_ids_train is not None

    # ----- Loss
    if getattr(model, "task", "regression") == 'regression':
        criterion = nn.MSELoss(reduction='mean')
        seg_multiclass = False
    else:
        # segmentation
        if getattr(model, "out_channels", 1) == 1:
            # binary
            # assume model head returns logits if final_activation == 'none'
            criterion = nn.BCEWithLogitsLoss() if getattr(model, "final_activation", "none") == 'none' else nn.BCELoss()
            seg_multiclass = False
        else:
            criterion = nn.CrossEntropyLoss()
            seg_multiclass = True

    # ----- Optimizer
    if optimizer.upper() == 'ADAM':
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        outer = n_epoch
        inner = 1
    else:
        optim = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20,
                                  history_size=max(10, n_epoch * 5), line_search_fn="strong_wolfe")
        # emulate "epochs" with multiple inner LBFGS steps
        outer = max(1, n_epoch // 20)
        inner = 20

    # ----- Training loop
    history: List[float] = []
    model.train()

    for epoch in range(outer):
        for _ in range(inner):
            epoch_loss, n_samples = 0.0, 0

            for batch in loader:
                if not varlen_mode:
                    # -------- fixed-grid
                    if with_cell_ids:
                        xb, yb, cb = batch
                        cb_np = cb.detach().cpu().numpy()
                    else:
                        xb, yb = batch
                        cb_np = None

                    xb = xb.to(device, dtype=torch.float32, non_blocking=True)
                    # y type: float for regression or binary; long for CrossEntropy
                    yb = yb.to(device, non_blocking=True)

                    if isinstance(optim, torch.optim.LBFGS):
                        def closure():
                            optim.zero_grad(set_to_none=True)
                            preds = model(xb, cell_ids=cb_np) if cb_np is not None else model(xb)
                            loss = criterion(preds, yb)
                            loss.backward()
                            return loss
                        _ = optim.step(closure)
                        with torch.no_grad():
                            preds = model(xb, cell_ids=cb_np) if cb_np is not None else model(xb)
                            loss = criterion(preds, yb)
                    else:
                        optim.zero_grad(set_to_none=True)
                        preds = model(xb, cell_ids=cb_np) if cb_np is not None else model(xb)
                        loss = criterion(preds, yb)
                        loss.backward()
                        if clip_grad_norm is not None:
                            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                        optim.step()

                    bs = xb.shape[0]
                    epoch_loss += float(loss.item()) * bs
                    n_samples += bs

                else:
                    # -------- variable-length (lists)
                    xs, ys, cs = batch  # lists

                    def _prep_xyc(i):
                        # x_i : (C, N_i)  -> (1, C, N_i)
                        xb = torch.as_tensor(xs[i], device=device, dtype=torch.float32)
                        if xb.dim() == 2:
                            xb = xb.unsqueeze(0)
                        elif xb.dim() != 3 or xb.shape[0] != 1:
                            raise ValueError("Each x[i] must be (C,N) or (1,C,N)")

                        # y_i :
                        yb = torch.as_tensor(ys[i], device=device)
                        if seg_multiclass:
                            # class indices: (N_i,) ou (1, N_i)
                            if yb.dim() == 2 and yb.shape[0] == 1:
                                yb = yb.squeeze(0)           # -> (N_i,)
                            elif yb.dim() != 1:
                                raise ValueError("For multiclass CE, y[i] must be (N,) or (1,N)")
                            # le critère CE recevra (1,C_out,N_i) et (N_i,)
                        else:
                            # régression / binaire: cible de forme (1, C_out, N_i)
                            if yb.dim() == 2:
                                yb = yb.unsqueeze(0)
                            elif yb.dim() != 3 or yb.shape[0] != 1:
                                raise ValueError("For regression/binary, y[i] must be (C_out,N) or (1,C_out,N)")

                        # cell_ids : (N_i,) -> (1, N_i) en numpy (le forward les attend en np.ndarray)
                        if cs is None or cs[i] is None:
                            cb_np = None
                        else:
                            c = cs[i].detach().cpu().numpy() if torch.is_tensor(cs[i]) else np.asarray(cs[i])
                            if c.ndim == 1:
                                c = c[None, :]               # -> (1, N_i)
                            cb_np = c
                        return xb, yb, cb_np

                    if isinstance(optim, torch.optim.LBFGS):
                        def closure():
                            optim.zero_grad(set_to_none=True)
                            total = 0.0
                            for i in range(len(xs)):
                                xb, yb, cb_np = _prep_xyc(i)
                                preds = model(xb, cell_ids=cb_np) if cb_np is not None else model(xb)
                                # adapter la cible à la sortie
                                if seg_multiclass:
                                    loss_i = criterion(preds, yb)       # preds: (1,C_out,N_i), yb: (N_i,)
                                else:
                                    loss_i = criterion(preds, yb)       # preds: (1,C_out,N_i), yb: (1,C_out,N_i)
                                loss_i.backward()
                                total += float(loss_i.item())
                            # retourner un scalaire Tensor pour LBFGS
                            return torch.tensor(total / max(1, len(xs)), device=device, dtype=torch.float32)

                        _ = optim.step(closure)
                        # logging (sans grad)
                        with torch.no_grad():
                            total = 0.0
                            for i in range(len(xs)):
                                xb, yb, cb_np = _prep_xyc(i)
                                preds = model(xb, cell_ids=cb_np) if cb_np is not None else model(xb)
                                if seg_multiclass:
                                    loss_i = criterion(preds, yb)
                                else:
                                    loss_i = criterion(preds, yb)
                                total += float(loss_i.item())
                            loss_val = total / max(1, len(xs))
                    else:
                        optim.zero_grad(set_to_none=True)
                        total = 0.0
                        for i in range(len(xs)):
                            xb, yb, cb_np = _prep_xyc(i)
                            preds = model(xb, cell_ids=cb_np) if cb_np is not None else model(xb)
                            if seg_multiclass:
                                loss_i = criterion(preds, yb)
                            else:
                                loss_i = criterion(preds, yb)
                            loss_i.backward()
                            total += float(loss_i.item())
                        if clip_grad_norm is not None:
                            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                        optim.step()
                        loss_val = total / max(1, len(xs))

                    epoch_loss += loss_val * max(1, len(xs))
                    n_samples  += max(1, len(xs))



            epoch_loss /= max(1, n_samples)
            history.append(epoch_loss)
            # print every view_epoch logical step
            if verbose and ((len(history) % view_epoch == 0) or (len(history) == 1)):
                print(f"[epoch {len(history)}] loss={epoch_loss:.6f}")

    return {"loss": history}
