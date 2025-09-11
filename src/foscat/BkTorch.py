import sys

import numpy as np
import torch
import torch.nn.functional as F

import foscat.BkBase as BackendBase


class BkTorch(BackendBase.BackendBase):

    def __init__(self, *args, **kwargs):
        # Impose que use_2D=True pour la classe scat
        super().__init__(name="torch", *args, **kwargs)
        self.backend = torch
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.float64 = self.backend.float64
        self.float32 = self.backend.float32
        self.int64 = self.backend.int64
        self.int32 = self.backend.int32
        self.complex64 = self.backend.complex128
        self.complex128 = self.backend.complex64

        dtype_map = {
            "float32": (self.backend.float32, self.backend.complex64),
            "float64": (self.backend.float64, self.backend.complex128),
        }

        if self.all_type in dtype_map:
            self.all_bk_type, self.all_cbk_type = dtype_map[self.all_type]
        else:
            raise ValueError(
                f"ERROR INIT foscat: {self.all_type} should be float32 or float64"
            )

        # ===========================================================================
        # INIT
        if self.mpi_rank == 0:
            sys.stdout.flush()

        gpus = torch.cuda.is_available()

        gpuname = "CPU:0"
        self.gpulist = {}
        self.gpulist[0] = gpuname
        self.ngpu = 1

        if gpus:
            try:
                self.ngpu = torch.cuda.device_count()
                self.gpulist = {}
                for k in range(self.ngpu):
                    self.gpulist[k] = torch.cuda.get_device_name(k)

            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        self.torch_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    # ---------------------------------
    # HEALPix binning utilities (nested)
    # ---------------------------------
    # Robust binned_mean that supports arbitrary subsets (N not divisible by 4)
    # and batched cell_ids of shape [B, N]. It returns compact per-parent means
    # even when some parents are missing (sparse coverage).

    def binned_mean_old(self, data, cell_ids, *, padded: bool = False, fill_value: float = float("nan")):
        """Average values over parent HEALPix pixels (nested) when downgrading nside→nside/2.

        Works with full-sky or sparse subsets (no need for N to be divisible by 4).

        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            Shape ``[..., N]`` or ``[B, ..., N]``.
        cell_ids : torch.LongTensor or np.ndarray
            Shape ``[N]`` or ``[B, N]`` (nested indexing at the *child* resolution).
        padded : bool, optional (default: False)
            Only used when ``cell_ids`` is ``[B, N]``. If ``False``, returns Python
            lists (ragged) of per-batch results. If ``True``, returns padded tensors
            plus a boolean mask of valid bins.
        fill_value : float, optional
            Value used for padding when ``padded=True``.

        Returns
        -------
        If ``cell_ids`` is ``[N]``:
            mean  : torch.Tensor, shape ``[..., n_bins]``
            groups: torch.LongTensor, shape ``[n_bins]``  (sorted unique parents)

        If ``cell_ids`` is ``[B, N]`` and ``padded=False``:
            means_list  : List[torch.Tensor] of length B, each shape ``[T, n_bins_b]``
                          where ``T = prod(data.shape[1:-1])`` (or 1 if none).
            groups_list : List[torch.LongTensor] of length B, each shape ``[n_bins_b]``

        If ``cell_ids`` is ``[B, N]`` and ``padded=True``:
            mean_padded : torch.Tensor, shape ``[B, T, max_bins]`` (or ``[B, max_bins]`` if T==1)
            groups_pad  : torch.LongTensor, shape ``[B, max_bins]`` (parents, padded with -1)
            mask        : torch.BoolTensor, shape ``[B, max_bins]`` (True where valid)
        """
        import torch, numpy as np

        # ---- Tensorize & device/dtype plumbing ----
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(dtype=torch.float32, device=getattr(self, 'torch_device', 'cpu'))
        if isinstance(cell_ids, np.ndarray):
            cell_ids = torch.from_numpy(cell_ids).to(dtype=torch.long, device=data.device)

        data = data.to(device=getattr(self, 'torch_device', data.device))
        cell_ids = cell_ids.to(device=data.device, dtype=torch.long)

        if data.ndim < 1:
            raise ValueError("`data` must have at least 1 dimension (last is N).")
        N = data.shape[-1]

        # Flatten leading dims (rows) for scatter convenience
        orig = data.shape[:-1]
        T = int(np.prod(orig[1:])) if len(orig) > 1 else 1  # repeats per batch row
        if cell_ids.ndim == 1:
            # Shared mapping for all rows
            groups = (cell_ids // 4).to(torch.long)  # [N]
            # Unique parent ids + inverse indices
            parents, inv = torch.unique(groups, sorted=True, return_inverse=True)
            n_bins = parents.numel()

            R = int(np.prod(orig)) if len(orig) > 0 else 1
            data_flat = data.reshape(R, N)  # [R, N]

            # Row offsets -> independent bins per row
            row_offsets = torch.arange(R, device=data.device).unsqueeze(1) * n_bins  # [R,1]
            idx = inv.unsqueeze(0).expand(R, -1) + row_offsets                         # [R,N]

            vals_flat = data_flat.reshape(-1)
            idx_flat  = idx.reshape(-1)

            out_sum = torch.zeros(R * n_bins, dtype=data.dtype, device=data.device)
            out_cnt = torch.zeros_like(out_sum)
            out_sum.scatter_add_(0, idx_flat, vals_flat)
            out_cnt.scatter_add_(0, idx_flat, torch.ones_like(vals_flat))
            out_cnt.clamp_(min=1)

            mean = (out_sum / out_cnt).view(*orig, n_bins)
            return mean, parents

        elif cell_ids.ndim == 2:
            B = cell_ids.shape[0]
            if data.shape[0] % B != 0:
                raise ValueError(f"Leading dim of data ({data.shape[0]}) must be a multiple of cell_ids batch ({B}).")
            R = int(np.prod(orig)) if len(orig) > 0 else 1
            data_flat = data.reshape(R, N)  # [R, N]
            B_data = data.shape[0]
            T = R // B_data                 # repeats per batch row (product of extra leading dims)

            means_list, groups_list = [], []
            max_bins = 0
            # First pass: compute per-batch parents/inv and scatter means
            for b in range(B):
                groups_b = (cell_ids[b] // 4).to(torch.long)  # [N]
                parents_b, inv_b = torch.unique(groups_b, sorted=True, return_inverse=True)
                n_bins_b = parents_b.numel()
                max_bins = max(max_bins, n_bins_b)

                # rows for this batch in data_flat
                start = b * T
                stop  = (b + 1) * T
                rows  = slice(start, stop)                    # T rows

                row_offsets = (torch.arange(T, device=data.device).unsqueeze(1) * n_bins_b)
                idx = inv_b.unsqueeze(0).expand(T, -1) + row_offsets  # [T, N]

                vals_flat = data_flat[rows].reshape(-1)
                idx_flat  = idx.reshape(-1)

                out_sum = torch.zeros(T * n_bins_b, dtype=data.dtype, device=data.device)
                out_cnt = torch.zeros_like(out_sum)
                out_sum.scatter_add_(0, idx_flat, vals_flat)
                out_cnt.scatter_add_(0, idx_flat, torch.ones_like(vals_flat))
                out_cnt.clamp_(min=1)
                mean_bt = (out_sum / out_cnt).view(T, n_bins_b)  # [T, n_bins_b]

                means_list.append(mean_bt)
                groups_list.append(parents_b)

            if not padded:
                return means_list, groups_list

            # Padded output
            # mean_padded: [B, T, max_bins]; groups_pad: [B, max_bins]; mask: [B, max_bins]
            mean_pad = torch.full((B, T, max_bins), fill_value, dtype=data.dtype, device=data.device)
            groups_pad = torch.full((B, max_bins), -1, dtype=torch.long, device=data.device)
            mask = torch.zeros((B, max_bins), dtype=torch.bool, device=data.device)
            for b, (m_b, g_b) in enumerate(zip(means_list, groups_list)):
                nb = g_b.numel()
                mean_pad[b, :, :nb] = m_b
                groups_pad[b, :nb] = g_b
                mask[b, :nb] = True

            # Reshape back to [B, (*extra leading dims), max_bins] if needed
            if len(orig) > 1:
                extra = orig[1:]  # e.g., (D1, D2, ...)
                mean_pad = mean_pad.view(B, *extra, max_bins)
            else:
                mean_pad = mean_pad.view(B, max_bins)

            return mean_pad, groups_pad, mask

        else:
            raise ValueError("`cell_ids` must be of shape [N] or [B, N].")

    def binned_mean(
                self,
                data,
                cell_ids,
                *,
                reduce: str = "mean",          # <-- NEW: "mean" (par défaut) ou "max"
                padded: bool = False,
                fill_value: float = float("nan"),
    ):
        """
            Reduce values over parent HEALPix pixels (nested) when downgrading nside→nside/2.

        Parameters
            ----------
            data : torch.Tensor | np.ndarray
            Shape [..., N] or [B, ..., N].
            cell_ids : torch.LongTensor | np.ndarray
            Shape [N] or [B, N] (nested indexing at the child resolution).
            reduce : {"mean","max"}, default "mean"
            Aggregation to apply within each parent group of 4 children.
            padded : bool, default False
            Only used when `cell_ids` is [B, N]. If False, returns ragged Python lists.
            If True, returns padded tensors + mask.
            fill_value : float, default NaN
            Padding value when `padded=True`.

        Returns
            -------
            # idem à ta doc existante, mais la valeur est une moyenne (reduce="mean")
            # ou un maximum (reduce="max").
        """
        
        # ---- Tensorize & device/dtype plumbing ----
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(
                dtype=torch.float32, device=getattr(self, "torch_device", "cpu")
            )
        if isinstance(cell_ids, np.ndarray):
            cell_ids = torch.from_numpy(cell_ids).to(
                dtype=torch.long, device=data.device
            )
            data = data.to(device=getattr(self, "torch_device", data.device))
            cell_ids = cell_ids.to(device=data.device, dtype=torch.long)

        if data.ndim < 1:
            raise ValueError("`data` must have at least 1 dimension (last is N).")
        N = data.shape[-1]

        # Utilitaires pour 'max' (fallback si scatter_reduce_ indisponible)
        def _segment_max(vals_flat, idx_flat, out_size):
            """Retourne out[out_idx] = max(vals[ idx==out_idx ]), vectorisé si possible."""
            # PyTorch >= 1.12 / 2.0: scatter_reduce_ disponible
            if hasattr(torch.Tensor, "scatter_reduce_"):
                out = torch.full((out_size,), -float("inf"),
                                 dtype=vals_flat.dtype, device=vals_flat.device)
                out.scatter_reduce_(0, idx_flat, vals_flat, reduce="amax", include_self=True)
                return out
            # Fallback simple (boucle sur indices uniques) – OK pour du downsample
            out = torch.full((out_size,), -float("inf"),
                             dtype=vals_flat.dtype, device=vals_flat.device)
            uniq = torch.unique(idx_flat)
            for u in uniq.tolist():
                m = (idx_flat == u)
                # éviter max() sur tensor vide
                if m.any():
                    out[u] = torch.max(vals_flat[m])
            return out

        # ---- Flatten leading dims for scatter convenience ----
        orig = data.shape[:-1]
        if cell_ids.ndim == 1:
            # Shared mapping for all rows
            groups = (cell_ids // 4).to(torch.long)                      # [N]
            parents, inv = torch.unique(groups, sorted=True, return_inverse=True)
            n_bins = parents.numel()

            R = int(np.prod(orig)) if len(orig) > 0 else 1
            data_flat = data.reshape(R, N)                                # [R, N]
            row_offsets = torch.arange(R, device=data.device).unsqueeze(1) * n_bins
            idx = inv.unsqueeze(0).expand(R, -1) + row_offsets            # [R, N]

            vals_flat = data_flat.reshape(-1)
            idx_flat = idx.reshape(-1)
            out_size = R * n_bins

            if reduce == "mean":
                out_sum = torch.zeros(out_size, dtype=data.dtype, device=data.device)
                out_cnt = torch.zeros_like(out_sum)
                out_sum.scatter_add_(0, idx_flat, vals_flat)
                out_cnt.scatter_add_(0, idx_flat, torch.ones_like(vals_flat))
                out_cnt.clamp_(min=1)
                reduced = out_sum / out_cnt
            elif reduce == "max":
                reduced = _segment_max(vals_flat, idx_flat, out_size)
            else:
                raise ValueError("reduce must be 'mean' or 'max'.")

            output = reduced.view(*orig, n_bins)
            return output, parents

        elif cell_ids.ndim == 2:
            # Per-batch mapping
            B = cell_ids.shape[0]
            R = int(np.prod(orig)) if len(orig) > 0 else 1
            data_flat = data.reshape(R, N)                                # [R, N]
            B_data = data.shape[0] if len(orig) > 0 else 1
            if B_data % B != 0:
                raise ValueError(
                    f"Leading dim of data ({B_data}) must be a multiple of cell_ids batch ({B})."
                )
            # T = repeats per batch row (product of extra leading dims)
            T = (R // B_data) if B_data > 0 else 1

            means_list, groups_list = [], []
            max_bins = 0

            for b in range(B):
                groups_b = (cell_ids[b] // 4).to(torch.long)              # [N]
                parents_b, inv_b = torch.unique(groups_b, sorted=True, return_inverse=True)
                n_bins_b = parents_b.numel()
                max_bins = max(max_bins, n_bins_b)

                # rows for this batch in data_flat
                start, stop = b * T, (b + 1) * T
                rows = slice(start, stop)                                  # T rows

                row_offsets = torch.arange(T, device=data.device).unsqueeze(1) * n_bins_b
                idx = inv_b.unsqueeze(0).expand(T, -1) + row_offsets       # [T, N]

                vals_flat = data_flat[rows].reshape(-1)
                idx_flat = idx.reshape(-1)
                out_size = T * n_bins_b

                if reduce == "mean":
                    out_sum = torch.zeros(out_size, dtype=data.dtype, device=data.device)
                    out_cnt = torch.zeros_like(out_sum)
                    out_sum.scatter_add_(0, idx_flat, vals_flat)
                    out_cnt.scatter_add_(0, idx_flat, torch.ones_like(vals_flat))
                    out_cnt.clamp_(min=1)
                    reduced_bt = (out_sum / out_cnt).view(T, n_bins_b)
                elif reduce == "max":
                    reduced_bt = _segment_max(vals_flat, idx_flat, out_size).view(T, n_bins_b)
                else:
                    raise ValueError("reduce must be 'mean' or 'max'.")

                means_list.append(reduced_bt)
                groups_list.append(parents_b)

            if not padded:
                return means_list, groups_list

            # Padded output (B, T, max_bins) [+ mask]
            mean_pad = torch.full((B, T, max_bins), fill_value, dtype=data.dtype, device=data.device)
            groups_pad = torch.full((B, max_bins), -1, dtype=torch.long, device=data.device)
            mask = torch.zeros((B, max_bins), dtype=torch.bool, device=data.device)
            for b, (m_b, g_b) in enumerate(zip(means_list, groups_list)):
                nb = g_b.numel()
                mean_pad[b, :, :nb] = m_b
                groups_pad[b, :nb] = g_b
                mask[b, :nb] = True

            # Reshape back to [B, (*extra dims), max_bins] si besoin
            if len(orig) > 1:
                extra = orig[1:]
                mean_pad = mean_pad.view(B, *extra, max_bins)
            else:
                mean_pad = mean_pad.view(B, max_bins)

            return mean_pad, groups_pad, mask

        else:
            raise ValueError("`cell_ids` must be of shape [N] or [B, N].")
        
    def average_by_cell_group(data, cell_ids):
        """
        data: tensor of shape [..., N, ...] (ex: [B, N, C])
        cell_ids: tensor of shape [N]
        Returns: mean_data of shape [..., G, ...] where G = number of unique cell_ids//4
        """
        original_shape = data.shape
        leading = data.shape[:-2]  # all dims before N
        N = data.shape[-2]
        trailing = data.shape[-1:]  # all dims after N

        groups = (cell_ids // 4).long()  # [N]
        unique_groups, group_indices, counts = torch.unique(
            groups, return_inverse=True, return_counts=True
        )

        return torch.bincount(group_indices, weights=data) / counts, unique_groups

    # ---------------------------------------------−---------
    # --             BACKEND DEFINITION                    --
    # ---------------------------------------------−---------
    def bk_len(self,S):
        if S is None:
            return 0
        return S.numel()
    
    def bk_SparseTensor(self, indice, w, dense_shape=[]):
        return self.backend.sparse_coo_tensor(indice, w, dense_shape).coalesce().to_sparse_csr().to(self.torch_device)

    def bk_stack(self, list, axis=0):
        return self.backend.stack(list, axis=axis).to(self.torch_device)

    def bk_sparse_dense_matmul(self, smat, mat):
        return smat.matmul(mat)

    def conv2d(self, x, w):
        """
        Perform 2D convolution using PyTorch format.

        Args:
            x: Tensor of shape [..., Nx, Ny] – input
            w: Tensor of shape [O_c, wx, wy] – conv weights

        Returns:
            Tensor of shape [..., O_c, Nx, Ny]
        """
        *leading_dims, Nx, Ny = x.shape  # extract leading dims
        O_c, wx, wy = w.shape

        # Flatten leading dims into batch dimension
        B = int(torch.prod(torch.tensor(leading_dims))) if leading_dims else 1
        x = x.reshape(B, 1, Nx, Ny)  # [B, 1, Nx, Ny]

        # Reshape filters to match conv2d format [O_c, 1, wx, wy]
        w = w[:, None, :, :]  # [O_c, 1, wx, wy]

        pad_x = wx // 2
        pad_y = wy // 2

        # Reflective padding to reduce edge artifacts
        x_padded = F.pad(x, (pad_y, pad_y, pad_x, pad_x), mode="reflect")

        # Apply convolution
        y = F.conv2d(x_padded, w)  # [B, O_c, Nx, Ny]

        # Restore original leading dimensions
        y = y.reshape(*leading_dims, O_c, Nx, Ny)

        return y
    
    def conv1d(self, x, w, strides=[1, 1, 1], padding="SAME"):
        """
        Performs 1D convolution along the last axis of a 2D tensor x[n, m] with kernel w[K].

        Parameters:
        - x: torch.Tensor of shape [n, m]
        - w: torch.Tensor of shape [K]
        - strides: list of 3 ints; only strides[1] (along axis -1) is used
        - padding: "SAME" or "VALID"

        Returns:
        - torch.Tensor of shape [n, m] (if SAME) or smaller (if VALID)
        """
        assert x.ndim == 2, "Input x must be a 2D tensor [n, m]"
        assert w.ndim == 1, "Kernel w must be a 1D tensor [K]"
        stride = strides[1]

        # Reshape for PyTorch conv1d: [batch, channels, width]
        x_reshaped = x.unsqueeze(1)         # [n, 1, m]
        w_flipped = w.flip(0).view(1, 1, -1)  # [out_channels=1, in_channels=1, kernel_size]

        if padding.upper() == "SAME":
            pad_total = w.shape[0] - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x_reshaped = F.pad(x_reshaped, (pad_left, pad_right), mode='constant', value=0)
            padding_mode = 'valid'
        elif padding.upper() == "VALID":
            padding_mode = 'valid'
        else:
            raise ValueError("padding must be either 'SAME' or 'VALID'")

        out = F.conv1d(x_reshaped, w_flipped, stride=stride, padding=0)  # manual padding applied above
        return out.squeeze(1)  # [n, m_out]

    def bk_threshold(self, x, threshold, greater=True):

        x.to(x.dtype)
        return (x > threshold) * x

    def bk_maximum(self, x1, x2):
        return self.backend.maximum(x1, x2)

    def bk_device(self, device_name):
        return self.backend.device(device_name)

    def bk_ones(self, shape, dtype=None):
        if dtype is None:
            dtype = self.all_type
        return self.bk_cast(np.ones(shape))

    def bk_conv1d(self, x, w):
        # Torch not yet done !!!
        return self.backend.nn.conv1d(x, w, stride=1, padding="SAME")

    def bk_flattenR(self, x):
        if self.bk_is_complex(x):
            rr = self.backend.reshape(
                self.bk_real(x), [np.prod(np.array(list(x.shape)))]
            )
            ii = self.backend.reshape(
                self.bk_imag(x), [np.prod(np.array(list(x.shape)))]
            )
            return self.bk_concat([rr, ii], axis=0)
        else:
            return self.backend.reshape(x, [np.prod(np.array(list(x.shape)))])

    def bk_flatten(self, x):
        return self.backend.reshape(x, [np.prod(np.array(list(x.shape)))])

    def bk_resize_image(self, x, shape):
        tmp = self.backend.nn.functional.interpolate(
            x.permute(0, 3, 1, 2), size=shape, mode="bilinear", align_corners=False
        )
        return self.bk_cast(tmp.permute(0, 2, 3, 1))

    def bk_L1(self, x):
        if x.dtype == self.all_cbk_type:
            xr = self.bk_real(x)
            # xi = self.bk_imag(x)

            r = self.backend.sign(xr) * self.backend.sqrt(self.backend.sign(xr) * xr + 1E-16)
            # return r
            # i = self.backend.sign(xi) * self.backend.sqrt(self.backend.sign(xi) * xi)

            return r
        else:
            return self.backend.sign(x) * self.backend.sqrt(self.backend.sign(x) * x + 1E-16)

    def bk_square_comp(self, x):
        if x.dtype == self.all_cbk_type:
            xr = self.bk_real(x)
            xi = self.bk_imag(x)

            r = xr * xr
            i = xi * xi
            return self.bk_complex(r, i)
        else:
            return x * x

    def bk_reduce_sum(self, data, axis=None):

        if axis is None:
            return self.backend.sum(data)
        else:
            return self.backend.sum(data, axis)

    # ---------------------------------------------−---------
    # return a tensor size

    def bk_size(self, data):
        return data.numel()

    def constant(self, data):
        return data

    def bk_reduce_mean(self, data, axis=None):

        if axis is None:
            return self.backend.mean(data)
        else:
            return self.backend.mean(data, axis)

    def bk_reduce_min(self, data, axis=None):

        if axis is None:
            return self.backend.min(data)
        else:
            return self.backend.min(data, axis)

    def bk_random_seed(self, value):

        return self.backend.random.set_seed(value)

    def bk_random_uniform(self, shape):

        return self.backend.random.uniform(shape)

    def bk_reduce_std(self, data, axis=None):
        if axis is None:
            r = self.backend.std(data)
        else:
            r = self.backend.std(data, axis)

        if self.bk_is_complex(data):
            return self.bk_complex(r, 0 * r)
        else:
            return r

    def bk_sqrt(self, data):

        return self.backend.sqrt(self.backend.abs(data))

    def bk_abs(self, data):
        return self.backend.abs(data)

    def bk_is_complex(self, data):

        if isinstance(data, np.ndarray):
            return data.dtype == "complex64" or data.dtype == "complex128"

        return data.dtype.is_complex

    def bk_distcomp(self, data):
        if self.bk_is_complex(data):
            res = self.bk_square(self.bk_real(data)) + self.bk_square(
                self.bk_imag(data)
            )
            return res
        else:
            return self.bk_square(data)

    def bk_norm(self, data):
        if self.bk_is_complex(data):
            res = self.bk_square(self.bk_real(data)) + self.bk_square(
                self.bk_imag(data)
            )
            return self.bk_sqrt(res)

        else:
            return self.bk_abs(data)

    def bk_square(self, data):

        return self.backend.square(data)

    def bk_log(self, data):
        return self.backend.log(data)

    def bk_matmul(self, a, b):
        return self.backend.matmul(a, b)

    def bk_tensor(self, data):
        return self.backend.constant(data).to(self.torch_device)

    def bk_shape_tensor(self, shape):
        return self.backend.tensor(shape=shape).to(self.torch_device)

    def bk_complex(self, real, imag):
        return self.backend.complex(real, imag).to(self.torch_device)

    def bk_exp(self, data):

        return self.backend.exp(data)

    def bk_min(self, data):

        return self.backend.reduce_min(data)

    def bk_argmin(self, data):

        return self.backend.argmin(data)

    def bk_tanh(self, data):

        return self.backend.math.tanh(data)

    def bk_max(self, data):

        return self.backend.reduce_max(data)

    def bk_argmax(self, data):

        return self.backend.argmax(data)

    def bk_reshape(self, data, shape):
        #if isinstance(data, np.ndarray):
        #    return data.reshape(shape)
        return data.reshape(shape)

    def bk_repeat(self, data, nn, axis=0):
        return self.backend.repeat_interleave(data, repeats=nn, dim=axis)

    def bk_tile(self, data, nn, axis=0):

        return self.backend.tile(data, dims=[nn])

    def bk_roll(self, data, nn, axis=0):
        return self.backend.roll(data, nn, axis=axis)

    def bk_expand_dims(self, data, axis=0):
        if isinstance(data, np.ndarray):
            data = self.backend.from_numpy(data)
        return self.backend.unsqueeze(data, axis)

    def bk_transpose(self, data, thelist):
        return self.backend.transpose(data, thelist[0], thelist[1])

    def bk_concat(self, data, axis=None):

        if axis is None:
            if data[0].dtype == self.all_cbk_type:
                ndata = len(data)
                xr = self.backend.concat([self.bk_real(data[k]) for k in range(ndata)])
                xi = self.backend.concat([self.bk_imag(data[k]) for k in range(ndata)])
                return self.bk_complex(xr, xi)
            else:
                return self.backend.concat(data)
        else:
            if data[0].dtype == self.all_cbk_type:
                ndata = len(data)
                xr = self.backend.concat(
                    [self.bk_real(data[k]) for k in range(ndata)], axis=axis
                )
                    
                xi = self.backend.concat(
                    [self.bk_imag(data[k]) for k in range(ndata)], axis=axis
                )
                return self.bk_complex(xr, xi)
            else:
                return self.backend.concat(data, axis=axis)

    def bk_zeros(self, shape, dtype=None):
        return self.backend.zeros(shape, dtype=dtype).to(self.torch_device)

    def bk_gather(self, data, idx, axis=0):
        if axis == -1:
            return data[...,idx]
        elif axis == 0:
            return data[idx]
        elif axis == 1:
            return data[:, idx]
        elif axis == 2:
            return data[:, :, idx]
        elif axis == 3:
            return data[:, :, :, idx]
        return data[idx,...]

    def bk_reverse(self, data, axis=0):
        return self.backend.flip(data, dims=[axis])

    def bk_fft(self, data):
        return self.backend.fft.fft(data)

    def bk_fftn(self, data, dim=None):
        return self.backend.fft.fftn(data, dim=dim)

    def bk_ifftn(self, data, dim=None, norm=None):
        return self.backend.fft.ifftn(data, dim=dim, norm=norm)

    def bk_rfft(self, data):
        return self.backend.fft.rfft(data)

    def bk_irfft(self, data):
        return self.backend.fft.irfft(data)

    def bk_conjugate(self, data):

        return self.backend.conj(data)

    def bk_real(self, data):
        return data.real

    def bk_imag(self, data):
        if data.dtype == self.all_cbk_type:
            return data.imag
        else:
            return 0

    def bk_relu(self, x):
        return self.backend.relu(x)

    def bk_clip_by_value(self, x, xmin, xmax):
        if isinstance(x, np.ndarray):
            x = np.clip(x, xmin, xmax)
        x = (
            self.backend.tensor(x, dtype=self.backend.float32)
            if not isinstance(x, self.backend.Tensor)
            else x
        )
        xmin = (
            self.backend.tensor(xmin, dtype=self.backend.float32)
            if not isinstance(xmin, self.backend.Tensor)
            else xmin
        )
        xmax = (
            self.backend.tensor(xmax, dtype=self.backend.float32)
            if not isinstance(xmax, self.backend.Tensor)
            else xmax
        )
        return self.backend.clamp(x, min=xmin, max=xmax)

    def bk_cast(self, x):
        if isinstance(x, np.float64):
            if self.all_bk_type == "float32":
                return self.backend.tensor(np.float32(x)).to(self.torch_device)
            else:
                return self.backend.tensor(x).to(self.torch_device)
        if isinstance(x, np.float32):
            if self.all_bk_type == "float64":
                return self.backend.tensor(np.float64(x)).to(self.torch_device)
            else:
                return self.backend.tensor(x).to(self.torch_device)
        if isinstance(x, np.complex128):
            if self.all_bk_type == "float32":
                return self.backend.tensor(np.complex64(x)).to(self.torch_device)
            else:
                return self.backend.tensor(x).to(self.torch_device)
        if isinstance(x, np.complex64):
            if self.all_bk_type == "float64":
                return self.backend.tensor(np.complex128(x)).to(self.torch_device)
            else:
                return self.backend.tensor(x).to(self.torch_device)

        if isinstance(x, np.int32) or isinstance(x, np.int64) or isinstance(x, int):
            if self.all_bk_type == "float64":
                return self.backend.tensor(np.float64(x)).to(self.torch_device)
            else:
                return self.backend.tensor(np.float32(x)).to(self.torch_device)

        if self.bk_is_complex(x):
            out_type = self.all_cbk_type
        else:
            out_type = self.all_bk_type

        if isinstance(x, np.ndarray):
            x = self.backend.from_numpy(x).to(self.torch_device)

        if x.dtype.is_complex:
            out_type = self.all_cbk_type
        else:
            out_type = self.all_bk_type

        return x.type(out_type).to(self.torch_device)

    def bk_variable(self, x):
        return self.bk_cast(x)

    def bk_assign(self, x, y):
        return y

    def bk_constant(self, x):

        return self.bk_cast(x)

    def bk_cos(self, x):
        return self.backend.cos(x)

    def bk_sin(self, x):
        return self.backend.sin(x)

    def bk_arctan2(self, c, s):
        return self.backend.arctan2(c, s)

    def bk_empty(self, list):
        return self.backend.empty(list)

    def to_numpy(self, x):
        if isinstance(x, np.ndarray):
            return x

        return x.cpu().numpy()
