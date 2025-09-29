from __future__ import annotations
from typing import List, Optional, Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class PlanarUNet(nn.Module):
    """
    U-Net 2D (images HxW) mirroring the parameterization of the HealpixUNet.

    Key compat points with HealpixUNet:
      - Same constructor fields: in_nside, n_chan_in, chanlist, KERNELSZ, task,
        out_channels, final_activation, device, down_type, dtype, head_reduce.
      - Two convs per level (encoder & decoder), GroupNorm + ReLU after each conv.
      - Downsampling by factor 2 at each level; upsampling mirrors back.
      - Head produces `out_channels` with optional BN and final activation.

    Differences vs sphere version:
      - Operates on regular 2D images of size (3*in_nside, 4*in_nside).
      - Standard Conv2d instead of custom spherical stencil.
      - No gauges (G=1 implicit) and no cell_ids.

    Shapes
    ------
    Input  : (B, C_in,  3*in_nside, 4*in_nside)
    Output : (B, C_out, 3*in_nside, 4*in_nside)

    Constraints
    -----------
    `in_nside` must be divisible by 2**depth, where depth == len(chanlist).
    """

    def __init__(
        self,
        *,
        in_nside: int,
        n_chan_in: int,
        chanlist: List[int],
        KERNELSZ: int = 3,
        task: Literal['regression', 'segmentation'] = 'regression',
        out_channels: int = 1,
        final_activation: Optional[Literal['none', 'sigmoid', 'softmax']] = None,
        device: Optional[torch.device | str] = None,
        down_type: Optional[Literal['mean','max']] = 'max',
        dtype: Literal['float32','float64'] = 'float32',
        head_reduce: Literal['mean','learned'] = 'mean',  # kept for API symmetry
    ) -> None:
        super().__init__()

        if len(chanlist) == 0:
            raise ValueError("chanlist must be non-empty (depth >= 1)")
        self.in_nside = int(in_nside)
        self.n_chan_in = int(n_chan_in)
        self.chanlist = list(map(int, chanlist))
        self.KERNELSZ = int(KERNELSZ)
        self.task = task
        self.out_channels = int(out_channels)
        self.down_type = down_type
        self.dtype = torch.float32 if dtype == 'float32' else torch.float64
        self.head_reduce = head_reduce
        
        # default final activation consistent with HealpixUNet
        if final_activation is None:
            if task == 'regression':
                self.final_activation = 'none'
            else:
                self.final_activation = 'sigmoid' if out_channels == 1 else 'softmax'
        else:
            self.final_activation = final_activation

        # Resolve device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        depth = len(self.chanlist)
        # geometry
        H0, W0 = 3 * self.in_nside, 4 * self.in_nside
        # ensure divisibility by 2**depth
        if (self.in_nside % (2 ** depth)) != 0:
            raise ValueError(
                f"in_nside={self.in_nside} must be divisible by 2**depth where depth={depth}"
            )

        padding = self.KERNELSZ // 2

        # --- Encoder ---
        enc_layers = []
        inC = self.n_chan_in
        self.skips_channels: List[int] = []
        for outC in self.chanlist:
            block = nn.Sequential(
                nn.Conv2d(inC, outC, kernel_size=self.KERNELSZ, padding=padding, bias=False),
                _norm_2d(outC, kind="group"),
                nn.ReLU(inplace=True),
                nn.Conv2d(outC, outC, kernel_size=self.KERNELSZ, padding=padding, bias=False),
                _norm_2d(outC, kind="group"),
                nn.ReLU(inplace=True),
            )
            enc_layers.append(block)
            inC = outC
            self.skips_channels.append(outC)
        self.encoder = nn.ModuleList(enc_layers)

        # Pools
        if self.down_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # --- Decoder ---
        dec_layers = []
        upconvs = []
        for l in reversed(range(depth)):
            skipC = self.skips_channels[l]
            upC = self.skips_channels[l + 1] if (l + 1) < depth else self.skips_channels[l]
            inC_dec = upC + skipC
            outC_dec = skipC

            upconvs.append(
                nn.ConvTranspose2d(upC, upC, kernel_size=2, stride=2)
            )
            dec_layers.append(
                nn.Sequential(
                    nn.Conv2d(inC_dec, outC_dec, kernel_size=self.KERNELSZ, padding=padding, bias=False),
                    _norm_2d(outC_dec, kind="group"),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(outC_dec, outC_dec, kernel_size=self.KERNELSZ, padding=padding, bias=False),
                    _norm_2d(outC_dec, kind="group"),
                    nn.ReLU(inplace=True),
                )
            )
        self.upconvs = nn.ModuleList(upconvs)
        self.decoder = nn.ModuleList(dec_layers)

        # --- Head ---
        head_inC = self.chanlist[0]
        self.head_conv = nn.Conv2d(head_inC, self.out_channels, kernel_size=self.KERNELSZ, padding=padding)
        self.head_bn = _norm_2d(self.out_channels, kind="group") if self.task == 'segmentation' else None

        # optional learned mixer kept for API compatibility (no gauges here)
        self.head_mixer = nn.Identity()

        self.to(self.device, dtype=self.dtype)

    def to_tensor(self,x):
        return torch.tensor(x,device=self.device)
    
    def to_numpy(self,x):
        if isinstance(x,np.ndarray):
            return x
        return x.cpu().numpy()
    
    # -------------------------- forward --------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W) with H=3*in_nside, W=4*in_nside"""
        if x.dim() != 4:
            raise ValueError("Input must be (B, C, H, W)")
        if x.shape[1] != self.n_chan_in:
            raise ValueError(f"Expected {self.n_chan_in} input channels, got {x.shape[1]}")

        x = x.to(self.device, dtype=self.dtype)

        skips = []
        z = x
        for l, block in enumerate(self.encoder):
            z = block(z)
            skips.append(z)
            if l < len(self.encoder) - 1:
                z = self.pool(z)

        # Decoder
        for d, l in enumerate(reversed(range(len(self.chanlist)))):
            if l < len(self.chanlist) - 1:
                z = self.upconvs[d](z)
                # pad if odd due to pooling/upsampling asymmetry (shouldn't happen given divisibility)
                sh = skips[l].shape
                if z.shape[-2:] != sh[-2:]:
                    z = _pad_to_match(z, sh[-2], sh[-1])
            z = torch.cat([skips[l], z], dim=1)
            z = self.decoder[d](z)

        y = self.head_conv(z)
        if self.task == 'segmentation' and self.head_bn is not None:
            y = self.head_bn(y)

        if self.final_activation == 'sigmoid':
            y = torch.sigmoid(y)
        elif self.final_activation == 'softmax':
            y = torch.softmax(y, dim=1)
        return y

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        batch_size: int = 8,
        *,
        amp: bool = False,
        out_device: Optional[str] = 'cpu',
        out_dtype: Literal['float32','float16'] = 'float32',
        show_pbar: bool = False,
    ) -> torch.Tensor:
        """Memory-safe prediction.
        - Streams mini-batches with torch.inference_mode() and optional AMP.
        - Moves outputs to `out_device` (default CPU) to **free GPU memory**.
        - Avoids keeping computation graph and GPU tensors around.
        """
        self.eval()
        out_list = []
        x = x if torch.is_tensor(x) else torch.as_tensor(x)
        n = x.shape[0]
        dtype_map = {'float32': torch.float32, 'float16': torch.float16}
        out_dtype_t = dtype_map[out_dtype]

        use_cuda = self.device.type == 'cuda'
        if use_cuda:
            torch.backends.cudnn.benchmark = True

        from math import ceil
        nb = ceil(n / batch_size)
        rng = range(0, n, batch_size)
        if show_pbar:
            try:
                from tqdm import tqdm  # type: ignore
                rng = tqdm(rng, total=nb, desc='predict')
            except Exception:
                pass

        with torch.inference_mode():
            autocast_ctx = (
                torch.cuda.amp.autocast() if (amp and use_cuda) else _nullcontext()
            )
            for i in rng:
                xb = x[i:i+batch_size]
                xb = xb.to(self.device, dtype=self.dtype, non_blocking=True)
                with autocast_ctx:
                    yb = self.forward(xb)
                # Move to out_device to free VRAM
                if out_device is not None:
                    yb = yb.to(out_device, dtype=out_dtype_t)
                else:
                    yb = yb.to(dtype=out_dtype_t)
                out_list.append(yb)
                del xb
                if use_cuda:
                    torch.cuda.empty_cache()
        return torch.cat(out_list, dim=0)



# -----------------------------
# Helpers
# -----------------------------

def _norm_2d(C: int, kind: str = "group", **kwargs) -> nn.Module:
    if kind == "group":
        num_groups = kwargs.get("num_groups", min(8, max(1, C // 8)) or 1)
        while C % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        return nn.GroupNorm(num_groups=num_groups, num_channels=C)
    elif kind == "instance":
        return nn.InstanceNorm2d(C, affine=True, track_running_stats=False)
    elif kind == "batch":
        return nn.BatchNorm2d(C)
    else:
        raise ValueError(f"Unknown norm kind: {kind}")


def _pad_to_match(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Pad x (B,C,h,w) with zeros on right/bottom to reach (H,W)."""
    _, _, h, w = x.shape
    ph = max(0, H - h)
    pw = max(0, W - w)
    if ph == 0 and pw == 0:
        return x
    return F.pad(x, (0, pw, 0, ph), mode='constant', value=0)


# -----------------------------
# Inference utilities
# -----------------------------
from contextlib import nullcontext as _nullcontext
from math import ceil

@torch.no_grad()
def predict_iter(
    model: nn.Module,
    x: torch.Tensor | np.ndarray,
    *,
    batch_size: int = 8,
    amp: bool = False,
    out_device: str = 'cpu',
    out_dtype: Literal['float32','float16'] = 'float32',
):
    """Generator that yields predictions batch-by-batch to avoid big concatenations.
    Useful when outputs are huge (e.g., large HxW or many classes).
    """
    model.eval()
    x = x if torch.is_tensor(x) else torch.as_tensor(x)
    n = x.shape[0]
    use_cuda = next(model.parameters()).is_cuda
    dtype_map = {'float32': torch.float32, 'float16': torch.float16}
    out_dtype_t = dtype_map[out_dtype]
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    with torch.inference_mode():
        autocast_ctx = (
            torch.cuda.amp.autocast() if (amp and use_cuda) else _nullcontext()
        )
        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size]
            xb = xb.to(next(model.parameters()).device, dtype=next(model.parameters()).dtype, non_blocking=True)
            with autocast_ctx:
                yb = model(xb)
            yb = yb.to(out_device, dtype=out_dtype_t)
            yield yb.cpu() if out_device == 'cpu' else yb
            del xb, yb
            if use_cuda:
                torch.cuda.empty_cache()

# -----------------------------
# Training utilities (mirror of Healpix fit)
# -----------------------------
from typing import Union
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def _prepare_targets(y: torch.Tensor, *, seg_multiclass: bool, n_classes: int) -> torch.Tensor:
    """Normalize target shapes/dtypes for losses.
    - Multiclass CE expects (N,H,W) with dtype long and values in [0, C-1].
    - Binary BCE/BCEWithLogits expects (N,1,H,W) float in [0,1].
    Accepts the following forms and converts accordingly:
      * (N,1,H,W) -> squeeze to (N,H,W) and cast long (clamped)
      * (N,C,H,W) one-hot/proba -> argmax along dim=1
      * (N,H,W) -> cast long
    """
    if seg_multiclass:
        # Accept (N,C,H,W) one-hot/proba
        if y.ndim == 4 and y.shape[1] == n_classes:
            y = y.argmax(dim=1)
        # Accept (N,1,H,W)
        if y.ndim == 4 and y.shape[1] == 1:
            y = y[:, 0]
        # Expect (N,H,W) now
        if y.ndim != 3:
            raise RuntimeError(f"For multiclass targets, expected (N,H,W), got shape {tuple(y.shape)}")
        if y.dtype != torch.long:
            y = y.long().clamp_(0, max(0, n_classes-1))
        return y
    else:
        # Binary/Regression path keeps shape (N,1,H,W)
        if y.ndim == 3:
            y = y.unsqueeze(1)
        if y.ndim != 4 or y.shape[1] != 1:
            raise RuntimeError(f"For binary/regression targets, expected (N,1,H,W), got {tuple(y.shape)}")
        return y

def fit(
        model: nn.Module,
        x_train: Union[torch.Tensor, np.ndarray],
        y_train: Union[torch.Tensor, np.ndarray],
        *,
        n_epoch: int = 10,
        view_epoch: int = 10,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        clip_grad_norm: Optional[float] = None,
        verbose: bool = True,
        optimizer: Literal['ADAM', 'LBFGS'] = 'ADAM',
    ) -> dict:
    """Training loop *miroir* de `healpix_unet_torch.fit`, adapté aux images 2D.

    - Entrées fixes: tensors/ndarrays de même taille (B, C, H, W) avec H=3*nside, W=4*nside
    - Perte: MSE (regression) / BCE(BCEWithLogits si final_activation='none') / CrossEntropy (multiclasses)
    - Optimiseur: ADAM ou LBFGS avec closure
    - Logs: renvoie {"loss": history}
    """
    device = next(model.parameters()).device
    model.to(device)

    # ---- DataLoader
    x_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_train, device=device)

    ds = TensorDataset(x_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False) = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # ---- Loss
    if getattr(model, 'task', 'regression') == 'regression':
        criterion = nn.MSELoss(reduction='mean')
        seg_multiclass = False
    else:
        if getattr(model, 'out_channels', 1) == 1:
            # Binary segmentation
            criterion = nn.BCEWithLogitsLoss() if getattr(model, 'final_activation', 'none') == 'none' else nn.BCELoss()
            seg_multiclass = False
        else:
            # Multiclass segmentation
            criterion = nn.CrossEntropyLoss()
            seg_multiclass = True

    # ---- Optim
    if optimizer.upper() == 'ADAM':
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        outer, inner = n_epoch, 1
    elif optimizer.upper() == 'LBFGS':
        optim = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20, history_size=50, line_search_fn='strong_wolfe')
        outer, inner = max(1, n_epoch // 20), 20
    else:
        raise ValueError("optimizer must be 'ADAM' or 'LBFGS'")

    # ---- Train
    history: List[float] = []
    model.train()

    for epoch in range(outer):
        for _ in range(inner):
            epoch_loss, n_samples = 0.0, 0
            for xb, yb in loader:
                xb = xb.to(device, dtype=torch.float32, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                # Normalize targets to what criterion expects
                yb = _prepare_targets(yb, seg_multiclass=seg_multiclass, n_classes=getattr(model, 'out_channels', 1))

                if isinstance(optim, torch.optim.LBFGS):
                    def closure():
                        optim.zero_grad(set_to_none=True)
                        preds = model(xb)
                        loss = criterion(preds, yb)
                        loss.backward()
                        return loss
                    loss_val = float(optim.step(closure).item())
                else:
                    optim.zero_grad(set_to_none=True)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optim.step()
                    loss_val = float(loss.item())

                epoch_loss += loss_val * xb.shape[0]
                n_samples  += xb.shape[0]

            epoch_loss /= max(1, n_samples)
            history.append(epoch_loss)
            if verbose and ((len(history) % view_epoch == 0) or (len(history) == 1)):
                print(f"[epoch {len(history)}] loss={epoch_loss:.6f}")

    return {"loss": history}



# -----------------------------
# Minimal smoke test
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    nside = 32
    chanlist = [16, 32, 64]
    net = PlanarUNet(
        in_nside=nside,
        n_chan_in=3,
        chanlist=chanlist,
        KERNELSZ=3,
        task='regression',
        out_channels=1,
    )
    x = torch.randn(2, 3, 3*nside, 4*nside)
    y = net(x)
    print('input:', x.shape, 'output:', y.shape)
