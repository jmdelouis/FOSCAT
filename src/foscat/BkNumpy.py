import numpy as np

import foscat.BkBase as BackendBase


class BkNumpy(BackendBase.BackendBase):

    def __init__(self, *args, **kwargs):
        # Impose que use_2D=True pour la classe scat
        super().__init__(name="tensorflow", *args, **kwargs)

        # ===========================================================================
        # INIT

        self.backend = np
        import scipy as scipy

        self.scipy = scipy

        self.float64 = self.backend.float64
        self.float32 = self.backend.float32
        self.int64 = self.backend.int64
        self.int32 = self.backend.int32
        self.complex64 = self.backend.complex128
        self.complex128 = self.backend.complex64

        if self.all_type == "float32":
            self.all_bk_type = self.backend.float32
            self.all_cbk_type = self.backend.complex64
        else:
            if self.all_type == "float64":
                self.all_type = "float64"
                self.all_bk_type = self.backend.float64
                self.all_cbk_type = self.backend.complex128
            else:
                print(
                    "ERROR INIT FOCUS ", self.all_type, " should be float32 or float64"
                )
                return None

        # ===========================================================================
        # INIT

        gpuname = "CPU:0"
        self.gpulist = {}
        self.gpulist[0] = gpuname
        self.ngpu = 1

    # ---------------------------------------------−---------
    # --             BACKEND DEFINITION                    --
    # ---------------------------------------------−---------
    def bk_len(self,S):
        if S is None:
            return 0
        return S.size
    
    def bk_SparseTensor(self, indice, w, dense_shape=[]):
        return self.scipy.sparse.coo_matrix(
            (w, (indice[:, 0], indice[:, 1])), shape=dense_shape
        )

    def bk_stack(self, list, axis=0):
        return self.backend.stack(list, axis=axis)

    def bk_sparse_dense_matmul(self, smat, mat):
        return smat.dot(mat)

    def conv2d(self, x, w, strides=[1, 1, 1, 1], padding="SAME"):
        res = np.zeros([x.shape[0], x.shape[1], x.shape[2], w.shape[3]], dtype=x.dtype)
        for k in range(w.shape[2]):
            for l_orient in range(w.shape[3]):
                for j in range(res.shape[0]):
                    tmp = self.scipy.signal.convolve2d(
                        x[j, :, :, k],
                        w[:, :, k, l_orient],
                        mode="same",
                        boundary="symm",
                    )
                    res[j, :, :, l_orient] += tmp
                    del tmp
        return res

    def conv1d(self, x, w, strides=[1, 1, 1], padding="SAME"):
        res = np.zeros([x.shape[0], x.shape[1], w.shape[2]], dtype=x.dtype)
        for k in range(w.shape[2]):
            for j in range(res.shape[0]):
                tmp = self.scipy.signal.convolve1d(
                    x[j, :, k], w[:, k], mode="same", boundary="symm"
                )
                res[j, :, :] += tmp
                del tmp
        return res

    def bk_threshold(self, x, threshold, greater=True):

        return (x > threshold) * x

    def bk_maximum(self, x1, x2):
        return x1 * (x1 > x2) + x2 * (x2 > x1)

    def bk_device(self, device_name):
        return self.backend.device(device_name)

    def bk_ones(self, shape, dtype=None):
        if dtype is None:
            dtype = self.all_type
        return self.backend.ones(shape, dtype=dtype)

    def bk_conv1d(self, x, w):
        res = np.zeros([x.shape[0], x.shape[1], w.shape[1]], dtype=x.dtype)
        for k in range(w.shape[1]):
            for l_orient in range(w.shape[2]):
                res[:, :, l_orient] += self.scipy.ndimage.convolve1d(
                    x[:, :, k], w[:, k, l_orient], axis=1, mode="constant", cval=0.0
                )
        return res

    def bk_flattenR(self, x):
        if self.bk_is_complex(x):
            return np.concatenate([x.real.flatten(), x.imag.flatten()], 0)
        else:
            return x.flatten()

    def bk_flatten(self, x):
        return x.flatten()

    def bk_size(self, x):
        return x.size

    def bk_resize_image(self, x, shape):
        return self.bk_cast(self.backend.image.resize(x, shape, method="bilinear"))

    def bk_L1(self, x):
        return self.backend.sign(x) * self.backend.sqrt(self.backend.sign(x) * x)

    def bk_square_comp(self, x):
        return x * x

    def bk_reduce_sum(self, data, axis=None):

        if axis is None:
            return np.sum(data)
        else:
            return np.sum(data, axis)

    # ---------------------------------------------−---------
    # return a tensor size

    def bk_reduce_mean(self, data, axis=None):

        if axis is None:
            return np.mean(data)
        else:
            return np.mean(data, axis)

    def bk_reduce_min(self, data, axis=None):

        if axis is None:
            return np.min(data)
        else:
            return np.min(data, axis)

    def bk_random_seed(self, value):

        return np.random.seed(value)

    def bk_random_uniform(self, shape):

        return np.random.rand(shape)

    def bk_reduce_std(self, data, axis=None):
        if axis is None:
            r = np.std(data)
            return self.bk_complex(r, 0 * r)
        else:
            r = np.std(data, axis)
        if self.bk_is_complex(data):
            return self.bk_complex(r, 0 * r)
        else:
            return r

    def bk_sqrt(self, data):

        return self.backend.sqrt(self.backend.abs(data))

    def bk_abs(self, data):
        return self.backend.abs(data)

    def bk_is_complex(self, data):

        return data.dtype == "complex64" or data.dtype == "complex128"

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

        return data * data

    def bk_log(self, data):
        return np.log(data)

    def bk_matmul(self, a, b):
        return np.dot(a, b)

    def bk_tensor(self, data):
        return data

    def bk_shape_tensor(self, shape):
        return np.zeros(shape)

    def bk_complex(self, real, imag):
        return real + 1j * imag

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

        return self.backend.tile(data, nn)

    def bk_roll(self, data, nn, axis=0):
        return self.backend.roll(data, nn, axis=axis)

    def bk_expand_dims(self, data, axis=0):
        return np.expand_dims(data, axis)

    def bk_transpose(self, data, thelist):
        return np.transpose(data, thelist)

    def bk_concat(self, data, axis=None):

        if axis is None:
            return np.concatenate(data, axis=0)
        else:
            return np.concatenate(data, axis=axis)

    def bk_zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def bk_gather(self, data, idx, axis=0):
        if axis == 0:
            return data[idx]
        elif axis == 1:
            return data[:, idx]
        elif axis == 2:
            return data[:, :, idx]
        elif axis == 3:
            return data[:, :, :, idx]
        return data[:, :, :, :, idx]

    def bk_reverse(self, data, axis=0):
        return np.reverse(data, axis=axis)

    def bk_fft(self, data):
        return self.backend.fft.fft(data)

    def bk_fftn(self, data, dim=None):
        return self.backend.fft.fftn(data)

    def bk_ifftn(self, data, dim=None, norm=None):
        return self.backend.fft.ifftn(data)

    def bk_rfft(self, data):
        return self.backend.fft.rfft(data)

    def bk_irfft(self, data):
        return self.backend.fft.irfft(data)

    def bk_conjugate(self, data):

        return data.conjugate()

    def bk_real(self, data):
        return data.real

    def bk_imag(self, data):
        return data.imag

    def bk_relu(self, x):
        return (x > 0) * x

    def bk_clip_by_value(self, x, xmin, xmax):
        return self.backend.clip(x, xmin, xmax)

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

        return x.astype(out_type)

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
        return x
