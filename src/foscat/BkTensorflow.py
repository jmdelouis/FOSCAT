import sys

import numpy as np
import tensorflow as tf

import foscat.BkBase as BackendBase


class BkTensorflow(BackendBase.BackendBase):

    def __init__(self, *args, **kwargs):
        # Impose que use_2D=True pour la classe scat
        super().__init__(name="tensorflow", *args, **kwargs)

        # ===========================================================================
        # INIT

        self.backend = tf
        # tf.config.threading.set_inter_op_parallelism_threads(1)
        # tf.config.threading.set_intra_op_parallelism_threads(1)
        self.tf_function = tf.function

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

        if self.mpi_rank == 0:
            if not self.silent:
                print(
                    "Num GPUs Available: ",
                    len(self.backend.config.experimental.list_physical_devices("GPU")),
                )
            sys.stdout.flush()

        self.backend.debugging.set_log_device_placement(False)
        self.backend.config.set_soft_device_placement(True)

        gpus = self.backend.config.experimental.list_physical_devices("GPU")

        gpuname = "CPU:0"
        self.gpulist = {}
        self.gpulist[0] = gpuname
        self.ngpu = 1

        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    self.backend.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = self.backend.config.experimental.list_logical_devices(
                    "GPU"
                )
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                sys.stdout.flush()
                self.ngpu = len(logical_gpus)
                if self.ngpu > 0:
                    gpuname = logical_gpus[self.gpupos % self.ngpu].name
                    self.gpulist = {}
                    for i in range(self.ngpu):
                        self.gpulist[i] = logical_gpus[i].name
                else:
                    gpuname = "CPU:0"
                    self.gpulist = {}
                    self.gpulist[0] = gpuname
                    self.ngpu = 1

            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    def tf_loc_function(self, func):
        return func

    # ---------------------------------------------−---------
    # --             BACKEND DEFINITION                    --
    # ---------------------------------------------−---------
    def bk_len(self,S):
        if S is None:
            return 0
        return tf.size(S)
    
    def bk_SparseTensor(self, indice, w, dense_shape=[]):
        return self.backend.SparseTensor(indice, w, dense_shape=dense_shape)

    def bk_stack(self, list, axis=0):
        return self.backend.stack(list, axis=axis)

    def bk_sparse_dense_matmul(self, smat, mat):
        return self.backend.sparse.sparse_dense_matmul(smat, mat)

    # for tensorflow wrapping only
    def periodic_pad(self, x, pad_height, pad_width):
        """
        Applies periodic ('wrap') padding to a 4D TensorFlow tensor (N, H, W, C).

        Args:
        x (tf.Tensor): Input tensor with shape (batch_size, height, width, channels).
            pad_height (tuple): Tuple (top, bottom) defining the vertical padding size.
            pad_width (tuple): Tuple (left, right) defining the horizontal padding size.

        Returns:
            tf.Tensor: Tensor with periodic padding applied.
        """
        # Vertical padding: take slices from bottom and top to wrap around
        top_pad = x[:, -pad_height:, :, :]  # Top padding from the bottom rows
        bottom_pad = x[:, :pad_height, :, :]  # Bottom padding from the top rows
        x_padded = self.backend.concat(
            [top_pad, x, bottom_pad], axis=1
        )  # Concatenate vertically

        # Horizontal padding: take slices from right and left to wrap around
        left_pad = x_padded[:, :, -pad_width:, :]  # Left padding from right columns
        right_pad = x_padded[:, :, :pad_width, :]  # Right padding from left columns

        x_padded = self.backend.concat(
            [left_pad, x_padded, right_pad], axis=2
        )  # Concatenate horizontally

        return x_padded

    def binned_mean(self, data, cell_ids):
        """
        data: Tensor of shape [..., N] (float32 or float64)
        cell_ids: Tensor of shape [N], int indices in [0, n_bins)
        Returns: mean per bin, shape [..., n_bins]
        """
        ishape = list(data.shape)
        A = 1
        for k in range(len(ishape) - 1):
            A *= ishape[k]
        N = tf.shape(data)[-1]

        # Step 1: group indices
        groups = tf.math.floordiv(cell_ids, 4)  # [N]
        unique_groups, I = tf.unique(groups)  # I: [N]
        n_bins = tf.shape(unique_groups)[0]

        # Step 2: build I_tiled with batch + channel offsets
        I_tiled = tf.tile(I[None, :], [A, 1])  # shape [, N]

        # Offset index to flatten across [A, n_bins]
        batch_channel_offsets = tf.range(A)[:, None] * n_bins
        I_offset = I_tiled + batch_channel_offsets  # shape [A, N]]

        # Step 3: flatten data to shape [A, N]
        data_reshaped = tf.reshape(data, [A, N])  # shape [A, N]

        # Flatten all for scatter_nd
        indices = tf.reshape(I_offset, [-1])  # [A*N]
        values = tf.reshape(data_reshaped, [-1])  # [A*N]

        """
        # Prepare for scatter: indices → [A*N, 1]
        scatter_indices = tf.expand_dims(indices, axis=1)
        scatter_indices = tf.cast(scatter_indices, tf.int64)
        """
        total_bins = A * n_bins

        # Step 4: sum per bin
        sum_per_bin = tf.math.unsorted_segment_sum(values, indices, total_bins)
        sum_per_bin = tf.reshape(sum_per_bin, ishape[0:-1] + [n_bins])  # [A, n_bins]

        # Step 5: count per bin (same indices)
        counts = tf.math.unsorted_segment_sum(1.0 + 0 * values, indices, total_bins)
        # counts = tf.math.bincount(indices, minlength=total_bins, maxlength=total_bins)
        counts = tf.reshape(counts, ishape[0:-1] + [n_bins])
        # counts = tf.maximum(counts, 1)  # Avoid division by zero
        # counts = tf.cast(counts, dtype=data.dtype)

        # Step 6: mean
        mean_per_bin = sum_per_bin / counts  # [B, A, n_bins]

        return mean_per_bin, unique_groups

    def conv2d(self, x, w):
        """
        Perform 2D convolution using TensorFlow.

        Args:
            x: Tensor of shape [..., Nx, Ny] – input
            w: Tensor of shape [O_c, wx, wy] – conv weights

        Returns:
            Tensor of shape [..., O_c, Nx, Ny]
        """
        # Extract shape
        *leading_dims, Nx, Ny = x.shape
        O_c, wx, wy = w.shape

        # Flatten leading dims into a batch dimension
        B = tf.reduce_prod(leading_dims) if leading_dims else 1
        x = tf.reshape(x, [B, Nx, Ny, 1])  # TensorFlow format: [B, H, W, C_in=1]

        # Reshape weights to [wx, wy, in_channels=1, out_channels]
        w = tf.reshape(w, [O_c, wx, wy])
        w = tf.transpose(w, perm=[1, 2, 0])  # [wx, wy, O_c]
        w = tf.reshape(w, [wx, wy, 1, O_c])  # [wx, wy, C_in=1, C_out]

        # Apply 'reflect' padding manually
        pad_x = wx // 2
        pad_y = wy // 2
        x_padded = tf.pad(
            x, [[0, 0], [pad_x, pad_x], [pad_y, pad_y], [0, 0]], mode="REFLECT"
        )

        # Perform convolution
        y = tf.nn.conv2d(
            x_padded, w, strides=[1, 1, 1, 1], padding="VALID"
        )  # [B, Nx, Ny, O_c]

        # Transpose back to match original format: [..., O_c, Nx, Ny]
        y = tf.transpose(y, [0, 3, 1, 2])  # [B, O_c, Nx, Ny]
        y = tf.reshape(y, [*leading_dims, O_c, Nx, Ny])

        return y

    def conv1d(self, x, w):
        """
        Perform 1D convolution using TensorFlow.

        Args:
            x: Tensor of shape [..., N] – input
            w: Tensor of shape [k] – conv weights

        Returns:
            Tensor of shape [...,N]
        """
        # Extract shapes
        *leading_dims, N = x.shape
        k = w.shape[0]

        # Flatten leading dims into batch dimension
        B = tf.reduce_prod(leading_dims) if leading_dims else 1
        x = tf.reshape(x, [B, N, 1])  # TensorFlow 1D format: [B, L, C=1]

        # Prepare weights: [k, in_channels=1, out_channels=O_c]
        w = tf.reshape(w, [k, 1, 1])

        # Apply 'reflect' padding
        pad = k // 2
        x_padded = tf.pad(x, [[0, 0], [pad, pad], [0, 0]], mode="REFLECT")

        # Perform convolution
        y = tf.nn.conv1d(x_padded, w, stride=1, padding="VALID")  # [B, N, O_c]

        # Transpose to [B, O_c, N] and reshape back
        y = tf.transpose(y, [0, 2, 1])  # [B, 1, N]
        y = tf.reshape(y, [*leading_dims, N])  # [..., N]

        return y

    def bk_threshold(self, x, threshold, greater=True):

        return self.backend.cast(x > threshold, x.dtype) * x

    def bk_maximum(self, x1, x2):
        return self.backend.maximum(x1, x2)

    def bk_device(self, device_name):
        return self.backend.device(device_name)

    def bk_ones(self, shape, dtype=None):
        if dtype is None:
            dtype = self.all_type
        return self.backend.ones(shape, dtype=dtype)

    def bk_conv1d(self, x, w):
        return self.backend.nn.conv1d(x, w, stride=[1, 1, 1], padding="SAME")

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
        return self.backend.flatten(x)

    def bk_resize_image(self, x, shape):
        return self.bk_cast(self.backend.image.resize(x, shape, method="bilinear"))

    def bk_L1(self, x):
        if x.dtype == self.all_cbk_type:
            xr = self.bk_real(x)
            xi = self.bk_imag(x)

            r = self.backend.sign(xr) * self.backend.sqrt(self.backend.sign(xr) * xr)
            # return r
            i = self.backend.sign(xi) * self.backend.sqrt(self.backend.sign(xi) * xi)

            return self.bk_complex(r, i)
        else:
            return self.backend.sign(x) * self.backend.sqrt(self.backend.sign(x) * x)

    def bk_square_comp(self, x):
        xr = self.bk_real(x)
        xi = self.bk_imag(x)

        r = xr * xr
        i = xi * xi
        return self.bk_complex(r, i)

    def bk_reduce_sum(self, data, axis=None):

        if axis is None:
            return self.backend.reduce_sum(data)
        else:
            return self.backend.reduce_sum(data, axis=axis)

    # ---------------------------------------------−---------
    # return a tensor size

    def bk_size(self, data):
        return self.backend.size(data)

    def bk_reduce_mean(self, data, axis=None):

        if axis is None:
            return self.backend.reduce_mean(data)
        else:
            return self.backend.reduce_mean(data, axis=axis)

    def bk_reduce_min(self, data, axis=None):

        if axis is None:
            return self.backend.reduce_min(data)
        else:
            return self.backend.reduce_min(data, axis=axis)

    def bk_random_seed(self, value):

        return self.backend.random.set_seed(value)

    def bk_random_uniform(self, shape):

        return self.backend.random.uniform(shape)

    def bk_reduce_std(self, data, axis=None):
        if axis is None:
            r = self.backend.math.reduce_std(data)
        else:
            r = self.backend.math.reduce_std(data, axis=axis)
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
        return self.backend.math.log(data)

    def bk_matmul(self, a, b):
        return self.backend.matmul(a, b)

    def bk_tensor(self, data):
        return self.backend.constant(data)

    def bk_shape_tensor(self, shape):
        return self.backend.tensor(shape=shape)

    def bk_complex(self, real, imag):
        return self.backend.dtypes.complex(real, imag)

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
        return self.backend.reshape(data, shape)

    def bk_repeat(self, data, nn, axis=0):
        return self.backend.repeat(data, nn, axis=axis)

    def bk_tile(self, data, nn, axis=0):
        order = [1 for k in data.shape]
        order[axis] = nn
        return self.backend.tile(data, self.backend.constant(order, tf.int32))

    def bk_roll(self, data, nn, axis=0):
        return self.backend.roll(data, nn, axis=axis)

    def bk_expand_dims(self, data, axis=0):
        return self.backend.expand_dims(data, axis=axis)

    def bk_transpose(self, data, thelist):
        return self.backend.transpose(data, thelist)

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
        return self.backend.zeros(shape, dtype=dtype)

    def bk_gather(self, data, idx, axis=0):
        return self.backend.gather(data, idx, axis=axis)

    def bk_reverse(self, data, axis=0):
        return self.backend.reverse(data, axis=[axis])

    def bk_fft(self, data):
        return self.backend.signal.fft(data)

    def bk_fftn(self, data, dim=None):
        # Equivalent of torch.fft.fftn(x, dim=dims) in TensorFlow
        if len(dim) == 2:
            return self.backend.signal.fft2d(self.bk_complex(data, 0 * data))
        else:
            return self.backend.signal.fft1d(self.bk_complex(data, 0 * data))

    def bk_ifftn(self, data, dim=None, norm=None):
        if norm is not None:
            if len(dim) == 2:
                normalization = self.backend.sqrt(
                    self.backend.cast(
                        data.shape[dim[0]] * data.shape[dim[1]], self.all_cbk_type
                    )
                )
                return self.backend.signal.ifft2d(data) * normalization

            else:
                normalization = self.backend.sqrt(
                    self.backend.cast(data.shape[dim[0]], self.all_cbk_type)
                )
                return self.backend.signal.ifft1d(data) * normalization
        else:
            if len(dim) == 2:
                return self.backend.signal.ifft2d(data)
            else:
                return self.backend.signal.ifft1d(data)

    def bk_rfft(self, data):
        return self.backend.signal.rfft(data)

    def bk_irfft(self, data):
        return self.backend.signal.irfft(data)

    def bk_conjugate(self, data):

        return self.backend.math.conj(data)

    def bk_real(self, data):
        return self.backend.math.real(data)

    def bk_imag(self, data):
        return self.backend.math.imag(data)

    def bk_relu(self, x):
        if x.dtype == self.all_cbk_type:
            xr = self.backend.nn.relu(self.bk_real(x))
            xi = self.backend.nn.relu(self.bk_imag(x))
            return self.bk_complex(xr, xi)
        else:
            return self.backend.nn.relu(x)

    def bk_clip_by_value(self, x, xmin, xmax):
        if isinstance(x, np.ndarray):
            x = np.clip(x, xmin, xmax)
        return self.backend.clip_by_value(x, xmin, xmax)

    def bk_cast(self, x):
        if isinstance(x, np.float64):
            if self.all_bk_type == "float32":
                return np.float32(x)
            else:
                return x
        if isinstance(x, np.float32):
            if self.all_bk_type == "float64":
                return np.float64(x)
            else:
                return x
        if isinstance(x, np.complex128):
            if self.all_bk_type == "float32":
                return np.complex64(x)
            else:
                return x
        if isinstance(x, np.complex64):
            if self.all_bk_type == "float64":
                return np.complex128(x)
            else:
                return x

        if isinstance(x, np.int32) or isinstance(x, np.int64) or isinstance(x, int):
            if self.all_bk_type == "float64":
                return np.float64(x)
            else:
                return np.float32(x)

        if self.bk_is_complex(x):
            out_type = self.all_cbk_type
        else:
            out_type = self.all_bk_type

        return self.backend.cast(x, out_type)

    def bk_variable(self, x):
        return self.backend.Variable(x)

    def bk_assign(self, x, y):
        return x.assign(y)

    def bk_constant(self, x):
        return self.backend.constant(x)

    def bk_cos(self, x):
        return self.backend.cos(x)

    def bk_sin(self, x):
        return self.backend.sin(x)

    def bk_arctan2(self, c, s):
        return self.backend.arctan2(c, s)

    def bk_empty(self, list):
        return self.backend.constant(list)

    def to_numpy(self, x):
        if isinstance(x, np.ndarray):
            return x

        return x.numpy()
