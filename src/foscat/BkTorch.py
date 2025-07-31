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

    import torch

    def binned_mean(self, data, cell_ids):
        """
        Compute the mean over groups of 4 nested HEALPix cells (nside → nside/2).

        Args:
            data (torch.Tensor): Tensor of shape [..., N], where N is the number of HEALPix cells.
            cell_ids (torch.LongTensor): Tensor of shape [N], with cell indices (nested ordering).

        Returns:
            torch.Tensor: Tensor of shape [..., n_bins], with averaged values per group of 4 cells.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(
                dtype=torch.float32, device=self.torch_device
            )
        if isinstance(cell_ids, np.ndarray):
            cell_ids = torch.from_numpy(cell_ids).to(
                dtype=torch.long, device=self.torch_device
            )

        # Compute supercell ids by grouping 4 nested cells together
        groups = cell_ids // 4

        # Get unique group ids and inverse mapping
        unique_groups, inverse_indices = torch.unique(groups, return_inverse=True)
        n_bins = unique_groups.shape[0]

        # Flatten all leading dimensions into a single batch dimension
        original_shape = data.shape[:-1]
        N = data.shape[-1]
        data_flat = data.reshape(-1, N)  # Shape: [B, N]

        # Prepare to compute sums using scatter_add
        B = data_flat.shape[0]

        # Repeat inverse indices for each batch element
        idx = inverse_indices.repeat(B, 1)  # Shape: [B, N]

        # Offset indices to simulate a per-batch scatter into [B * n_bins]
        batch_offsets = torch.arange(B, device=data.device).unsqueeze(1) * n_bins
        idx_offset = idx + batch_offsets  # Shape: [B, N]

        # Flatten everything for scatter
        idx_offset_flat = idx_offset.flatten()
        data_flat_flat = data_flat.flatten()

        # Accumulate sums per bin
        out = torch.zeros(B * n_bins, dtype=data.dtype, device=data.device)
        out = out.scatter_add(0, idx_offset_flat, data_flat_flat)

        # Count number of elements per bin (to compute mean)
        ones = torch.ones_like(data_flat_flat)
        counts = torch.zeros(B * n_bins, dtype=data.dtype, device=data.device)
        counts = counts.scatter_add(0, idx_offset_flat, ones)

        # Compute mean
        mean = out / counts  # Shape: [B * n_bins]
        mean = mean.view(B, n_bins)

        # Restore original leading dimensions
        return mean.view(*original_shape, n_bins), unique_groups

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
        return self.backend.sparse_coo_tensor(indice.T, w, dense_shape).to_sparse_csr().to(self.torch_device)

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
