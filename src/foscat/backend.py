import sys

import numpy as np


class foscat_backend:

    def __init__(self, name, mpi_rank=0, all_type="float64", gpupos=0, silent=False):

        self.TENSORFLOW = 1
        self.TORCH = 2
        self.NUMPY = 3

        # table use to compute the iso orientation rotation
        self._iso_orient = {}
        self._iso_orient_T = {}
        self._iso_orient_C = {}
        self._iso_orient_C_T = {}
        self._fft_1_orient = {}
        self._fft_1_orient_C = {}
        self._fft_2_orient = {}
        self._fft_2_orient_C = {}
        self._fft_3_orient = {}
        self._fft_3_orient_C = {}

        self.BACKEND = name

        if name not in ["tensorflow", "torch", "numpy"]:
            print('Backend "%s" not yet implemented' % (name))
            print(" Choose inside the next 3 available backends :")
            print(" - tensorflow")
            print(" - torch")
            print(" - numpy (Impossible to do synthesis using numpy)")
            return None

        if self.BACKEND == "tensorflow":
            import tensorflow as tf

            self.backend = tf
            self.BACKEND = self.TENSORFLOW
            # tf.config.threading.set_inter_op_parallelism_threads(1)
            # tf.config.threading.set_intra_op_parallelism_threads(1)
            self.tf_function = tf.function

        if self.BACKEND == "torch":
            import torch

            self.BACKEND = self.TORCH
            self.backend = torch
            self.tf_function = self.tf_loc_function

        if self.BACKEND == "numpy":
            self.BACKEND = self.NUMPY
            self.backend = np
            import scipy as scipy

            self.scipy = scipy
            self.tf_function = self.tf_loc_function

        self.float64 = self.backend.float64
        self.float32 = self.backend.float32
        self.int64 = self.backend.int64
        self.int32 = self.backend.int32
        self.complex64 = self.backend.complex128
        self.complex128 = self.backend.complex64

        if all_type == "float32":
            self.all_bk_type = self.backend.float32
            self.all_cbk_type = self.backend.complex64
        else:
            if all_type == "float64":
                self.all_type = "float64"
                self.all_bk_type = self.backend.float64
                self.all_cbk_type = self.backend.complex128
            else:
                print("ERROR INIT FOCUS ", all_type, " should be float32 or float64")
                return None
        # ===========================================================================
        # INIT
        if mpi_rank == 0:
            if self.BACKEND == self.TENSORFLOW and not silent:
                print(
                    "Num GPUs Available: ",
                    len(self.backend.config.experimental.list_physical_devices("GPU")),
                )
            sys.stdout.flush()

        if self.BACKEND == self.TENSORFLOW:
            self.backend.debugging.set_log_device_placement(False)
            self.backend.config.set_soft_device_placement(True)

            gpus = self.backend.config.experimental.list_physical_devices("GPU")

        if self.BACKEND == self.TORCH:
            gpus = torch.cuda.is_available()

        if self.BACKEND == self.NUMPY:
            gpus = []
        gpuname = "CPU:0"
        self.gpulist = {}
        self.gpulist[0] = gpuname
        self.ngpu = 1

        if gpus:
            try:
                if self.BACKEND == self.TENSORFLOW:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        self.backend.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = (
                        self.backend.config.experimental.list_logical_devices("GPU")
                    )
                    print(
                        len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
                    )
                    sys.stdout.flush()
                    self.ngpu = len(logical_gpus)
                    gpuname = logical_gpus[gpupos % self.ngpu].name
                    self.gpulist = {}
                    for i in range(self.ngpu):
                        self.gpulist[i] = logical_gpus[i].name
                if self.BACKEND == self.TORCH:
                    self.ngpu = torch.cuda.device_count()
                    self.gpulist = {}
                    for k in range(self.ngpu):
                        self.gpulist[k] = torch.cuda.get_device_name(0)

            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    def tf_loc_function(self, func):
        return func

    def calc_iso_orient(self, norient):
        tmp = np.zeros([norient * norient, norient])
        for i in range(norient):
            for j in range(norient):
                tmp[j * norient + (j + i) % norient, i] = 0.25

        self._iso_orient[norient] = self.constant(self.bk_cast(tmp))
        self._iso_orient_T[norient] = self.constant(self.bk_cast(4 * tmp.T))
        self._iso_orient_C[norient] = self.bk_complex(
            self._iso_orient[norient], 0 * self._iso_orient[norient]
        )
        self._iso_orient_C_T[norient] = self.bk_complex(
            self._iso_orient_T[norient], 0 * self._iso_orient_T[norient]
        )

    def calc_fft_orient(self, norient, nharm, imaginary):

        x = np.arange(norient) / norient * 2 * np.pi

        if imaginary:
            tmp = np.zeros([norient, 1 + nharm * 2])
            tmp[:, 0] = 1.0
            for k in range(nharm):
                tmp[:, k * 2 + 1] = np.cos(x * (k + 1))
                tmp[:, k * 2 + 2] = np.sin(x * (k + 1))

            self._fft_1_orient[(norient, nharm, imaginary)] = self.bk_cast(
                self.constant(tmp)
            )
            self._fft_1_orient_C[(norient, nharm, imaginary)] = self.bk_complex(
                self._fft_1_orient[(norient, nharm, imaginary)],
                0 * self._fft_1_orient[(norient, nharm, imaginary)],
            )
        else:
            tmp = np.zeros([norient, 1 + nharm])
            for k in range(nharm + 1):
                tmp[:, k] = np.cos(x * k)

            self._fft_1_orient[(norient, nharm, imaginary)] = self.bk_cast(
                self.constant(tmp)
            )
            self._fft_1_orient_C[(norient, nharm, imaginary)] = self.bk_complex(
                self._fft_1_orient[(norient, nharm, imaginary)],
                0 * self._fft_1_orient[(norient, nharm, imaginary)],
            )

        x = np.repeat(x, norient).reshape(norient, norient)

        if imaginary:
            tmp = np.zeros([norient, norient, (1 + nharm * 2), (1 + nharm * 2)])
            tmp[:, :, 0, 0] = 1.0
            for k in range(nharm):
                tmp[:, :, k * 2 + 1, 0] = np.cos(x * (k + 1))
                tmp[:, :, k * 2 + 2, 0] = np.sin(x * (k + 1))
                tmp[:, :, 0, k * 2 + 1] = np.cos((x.T) * (k + 1))
                tmp[:, :, 0, k * 2 + 2] = np.sin((x.T) * (k + 1))
                for l_orient in range(nharm):
                    tmp[:, :, k * 2 + 1, l_orient * 2 + 1] = np.cos(
                        x * (k + 1)
                    ) * np.cos((x.T) * (l_orient + 1))
                    tmp[:, :, k * 2 + 2, l_orient * 2 + 1] = np.sin(
                        x * (k + 1)
                    ) * np.cos((x.T) * (l_orient + 1))
                    tmp[:, :, k * 2 + 1, l_orient * 2 + 2] = np.cos(
                        x * (k + 1)
                    ) * np.sin((x.T) * (l_orient + 1))
                    tmp[:, :, k * 2 + 2, l_orient * 2 + 2] = np.sin(
                        x * (k + 1)
                    ) * np.sin((x.T) * (l_orient + 1))

            self._fft_2_orient[(norient, nharm, imaginary)] = self.bk_cast(
                self.constant(
                    tmp.reshape(norient * norient, (1 + 2 * nharm) * (1 + 2 * nharm))
                )
            )
            self._fft_2_orient_C[(norient, nharm, imaginary)] = self.bk_complex(
                self._fft_2_orient[(norient, nharm, imaginary)],
                0 * self._fft_2_orient[(norient, nharm, imaginary)],
            )
        else:
            tmp = np.zeros([norient, norient, (1 + nharm), (1 + nharm)])

            for k in range(nharm + 1):
                for l_orient in range(nharm + 1):
                    tmp[:, :, k, l_orient] = np.cos(x * k) * np.cos((x.T) * l_orient)

            self._fft_2_orient[(norient, nharm, imaginary)] = self.bk_cast(
                self.constant(tmp.reshape(norient * norient, (1 + nharm) * (1 + nharm)))
            )
            self._fft_2_orient_C[(norient, nharm, imaginary)] = self.bk_complex(
                self._fft_2_orient[(norient, nharm, imaginary)],
                0 * self._fft_2_orient[(norient, nharm, imaginary)],
            )

        x = np.arange(norient) / norient * 2 * np.pi
        xx = np.zeros([norient, norient, norient])
        yy = np.zeros([norient, norient, norient])
        zz = np.zeros([norient, norient, norient])
        for i in range(norient):
            for j in range(norient):
                xx[:, i, j] = x
                yy[i, :, j] = x
                zz[i, j, :] = x

        if imaginary:
            tmp = np.ones(
                [
                    norient,
                    norient,
                    norient,
                    (1 + nharm * 2),
                    (1 + nharm * 2),
                    (1 + nharm * 2),
                ]
            )

            for k in range(nharm):
                tmp[:, :, :, k * 2 + 1, 0, 0] = np.cos(xx * (k + 1))
                tmp[:, :, :, 0, k * 2 + 1, 0] = np.cos(yy * (k + 1))
                tmp[:, :, :, 0, 0, k * 2 + 1] = np.cos(zz * (k + 1))

                tmp[:, :, :, k * 2 + 2, 0, 0] = np.sin(xx * (k + 1))
                tmp[:, :, :, 0, k * 2 + 2, 0] = np.sin(yy * (k + 1))
                tmp[:, :, :, 0, 0, k * 2 + 2] = np.sin(zz * (k + 1))
                for l_orient in range(nharm):
                    tmp[:, :, :, k * 2 + 1, l_orient * 2 + 1, 0] = np.cos(
                        xx * (k + 1)
                    ) * np.cos(yy * (l_orient + 1))
                    tmp[:, :, :, k * 2 + 1, l_orient * 2 + 2, 0] = np.cos(
                        xx * (k + 1)
                    ) * np.sin(yy * (l_orient + 1))
                    tmp[:, :, :, k * 2 + 2, l_orient * 2 + 1, 0] = np.sin(
                        xx * (k + 1)
                    ) * np.cos(yy * (l_orient + 1))
                    tmp[:, :, :, k * 2 + 2, l_orient * 2 + 2, 0] = np.sin(
                        xx * (k + 1)
                    ) * np.sin(yy * (l_orient + 1))

                    tmp[:, :, :, k * 2 + 1, 0, l_orient * 2 + 1] = np.cos(
                        xx * (k + 1)
                    ) * np.cos(zz * (l_orient + 1))
                    tmp[:, :, :, k * 2 + 1, 0, l_orient * 2 + 2] = np.cos(
                        xx * (k + 1)
                    ) * np.sin(zz * (l_orient + 1))
                    tmp[:, :, :, k * 2 + 2, 0, l_orient * 2 + 1] = np.sin(
                        xx * (k + 1)
                    ) * np.cos(zz * (l_orient + 1))
                    tmp[:, :, :, k * 2 + 2, 0, l_orient * 2 + 2] = np.sin(
                        xx * (k + 1)
                    ) * np.sin(zz * (l_orient + 1))

                    tmp[:, :, :, 0, k * 2 + 1, l_orient * 2 + 1] = np.cos(
                        yy * (k + 1)
                    ) * np.cos(zz * (l_orient + 1))
                    tmp[:, :, :, 0, k * 2 + 1, l_orient * 2 + 2] = np.cos(
                        yy * (k + 1)
                    ) * np.sin(zz * (l_orient + 1))
                    tmp[:, :, :, 0, k * 2 + 2, l_orient * 2 + 1] = np.sin(
                        yy * (k + 1)
                    ) * np.cos(zz * (l_orient + 1))
                    tmp[:, :, :, 0, k * 2 + 2, l_orient * 2 + 2] = np.sin(
                        yy * (k + 1)
                    ) * np.sin(zz * (l_orient + 1))

                    for m in range(nharm):
                        tmp[:, :, :, k * 2 + 1, l_orient * 2 + 1, m * 2 + 1] = (
                            np.cos(xx * (k + 1))
                            * np.cos(yy * (l_orient + 1))
                            * np.cos(zz * (m + 1))
                        )
                        tmp[:, :, :, k * 2 + 1, l_orient * 2 + 1, m * 2 + 2] = (
                            np.cos(xx * (k + 1))
                            * np.cos(yy * (l_orient + 1))
                            * np.sin(zz * (m + 1))
                        )
                        tmp[:, :, :, k * 2 + 1, l_orient * 2 + 2, m * 2 + 1] = (
                            np.cos(xx * (k + 1))
                            * np.sin(yy * (l_orient + 1))
                            * np.cos(zz * (m + 1))
                        )
                        tmp[:, :, :, k * 2 + 1, l_orient * 2 + 2, m * 2 + 2] = (
                            np.cos(xx * (k + 1))
                            * np.sin(yy * (l_orient + 1))
                            * np.sin(zz * (m + 1))
                        )
                        tmp[:, :, :, k * 2 + 2, l_orient * 2 + 1, m * 2 + 1] = (
                            np.sin(xx * (k + 1))
                            * np.cos(yy * (l_orient + 1))
                            * np.cos(zz * (m + 1))
                        )
                        tmp[:, :, :, k * 2 + 2, l_orient * 2 + 1, m * 2 + 2] = (
                            np.sin(xx * (k + 1))
                            * np.cos(yy * (l_orient + 1))
                            * np.sin(zz * (m + 1))
                        )
                        tmp[:, :, :, k * 2 + 2, l_orient * 2 + 2, m * 2 + 1] = (
                            np.sin(xx * (k + 1))
                            * np.sin(yy * (l_orient + 1))
                            * np.cos(zz * (m + 1))
                        )
                        tmp[:, :, :, k * 2 + 2, l_orient * 2 + 2, m * 2 + 2] = (
                            np.sin(xx * (k + 1))
                            * np.sin(yy * (l_orient + 1))
                            * np.sin(zz * (m + 1))
                        )

            self._fft_3_orient[(norient, nharm, imaginary)] = self.bk_cast(
                self.constant(
                    tmp.reshape(
                        norient * norient * norient,
                        (1 + nharm * 2) * (1 + nharm * 2) * (1 + nharm * 2),
                    )
                )
            )
            self._fft_3_orient_C[(norient, nharm, imaginary)] = self.bk_complex(
                self._fft_3_orient[(norient, nharm, imaginary)],
                0 * self._fft_3_orient[(norient, nharm, imaginary)],
            )
        else:
            tmp = np.zeros(
                [norient, norient, norient, (1 + nharm), (1 + nharm), (1 + nharm)]
            )

            for k in range(nharm + 1):
                for l_orient in range(nharm + 1):
                    for m in range(nharm + 1):
                        tmp[:, :, :, k, l_orient, m] = (
                            np.cos(xx * k) * np.cos(yy * l_orient) * np.cos(zz * m)
                        )

            self._fft_3_orient[(norient, nharm, imaginary)] = self.bk_cast(
                self.constant(
                    tmp.reshape(
                        norient * norient * norient,
                        (1 + nharm) * (1 + nharm) * (1 + nharm),
                    )
                )
            )
            self._fft_3_orient_C[(norient, nharm, imaginary)] = self.bk_complex(
                self._fft_3_orient[(norient, nharm, imaginary)],
                0 * self._fft_3_orient[(norient, nharm, imaginary)],
            )

    # ---------------------------------------------−---------
    # --             BACKEND DEFINITION                    --
    # ---------------------------------------------−---------
    def bk_SparseTensor(self, indice, w, dense_shape=[]):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.SparseTensor(indice, w, dense_shape=dense_shape)
        if self.BACKEND == self.TORCH:
            return self.backend.sparse_coo_tensor(indice.T, w, dense_shape)
        if self.BACKEND == self.NUMPY:
            return self.scipy.sparse.coo_matrix(
                (w, (indice[:, 0], indice[:, 1])), shape=dense_shape
            )

    def bk_stack(self, list, axis=0):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.stack(list, axis=axis)
        if self.BACKEND == self.TORCH:
            return self.backend.stack(list, axis=axis)
        if self.BACKEND == self.NUMPY:
            return self.backend.stack(list, axis=axis)

    def bk_sparse_dense_matmul(self, smat, mat):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.sparse.sparse_dense_matmul(smat, mat)
        if self.BACKEND == self.TORCH:
            return smat.matmul(mat)
        if self.BACKEND == self.NUMPY:
            return smat.dot(mat)

    def conv2d(self, x, w, strides=[1, 1, 1, 1], padding="SAME"):
        if self.BACKEND == self.TENSORFLOW:
            kx = w.shape[0]
            ky = w.shape[1]
            paddings = self.backend.constant(
                [[0, 0], [kx // 2, kx // 2], [ky // 2, ky // 2], [0, 0]]
            )
            tmp = self.backend.pad(x, paddings, "SYMMETRIC")
            return self.backend.nn.conv2d(tmp, w, strides=strides, padding="VALID")
        # to be written!!!
        if self.BACKEND == self.TORCH:
            return x
        if self.BACKEND == self.NUMPY:
            res = np.zeros(
                [x.shape[0], x.shape[1], x.shape[2], w.shape[3]], dtype=x.dtype
            )
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
        if self.BACKEND == self.TENSORFLOW:
            kx = w.shape[0]
            paddings = self.backend.constant([[0, 0], [kx // 2, kx // 2], [0, 0]])
            tmp = self.backend.pad(x, paddings, "SYMMETRIC")

            return self.backend.nn.conv1d(tmp, w, stride=strides, padding="VALID")
        # to be written!!!
        if self.BACKEND == self.TORCH:
            return x
        if self.BACKEND == self.NUMPY:
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

        if self.BACKEND == self.TENSORFLOW:
            return self.backend.cast(x > threshold, x.dtype) * x
        if self.BACKEND == self.TORCH:
            x.to(x.dtype)
            return (x > threshold) * x
            # return(self.backend.cast(x>threshold,x.dtype)*x)
        if self.BACKEND == self.NUMPY:
            return (x > threshold) * x

    def bk_maximum(self, x1, x2):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.maximum(x1, x2)
        if self.BACKEND == self.TORCH:
            return self.backend.maximum(x1, x2)
        if self.BACKEND == self.NUMPY:
            return x1 * (x1 > x2) + x2 * (x2 > x1)

    def bk_device(self, device_name):
        return self.backend.device(device_name)

    def bk_ones(self, shape, dtype=None):
        if dtype is None:
            dtype = self.all_type
        if self.BACKEND == self.TORCH:
            return self.bk_cast(np.ones(shape))
        return self.backend.ones(shape, dtype=dtype)

    def bk_conv1d(self, x, w):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.nn.conv1d(x, w, stride=[1, 1, 1], padding="SAME")
        if self.BACKEND == self.TORCH:
            # Torch not yet done !!!
            return self.backend.nn.conv1d(x, w, stride=1, padding="SAME")
        if self.BACKEND == self.NUMPY:
            res = np.zeros([x.shape[0], x.shape[1], w.shape[1]], dtype=x.dtype)
            for k in range(w.shape[1]):
                for l_orient in range(w.shape[2]):
                    res[:, :, l_orient] += self.scipy.ndimage.convolve1d(
                        x[:, :, k], w[:, k, l_orient], axis=1, mode="constant", cval=0.0
                    )
            return res

    def bk_flattenR(self, x):
        if self.BACKEND == self.TENSORFLOW or self.BACKEND == self.TORCH:
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

        if self.BACKEND == self.NUMPY:
            if self.bk_is_complex(x):
                return np.concatenate([x.real.flatten(), x.imag.flatten()], 0)
            else:
                return x.flatten()

    def bk_flatten(self, x):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.reshape(x, [np.prod(np.array(list(x.shape)))])
        if self.BACKEND == self.TORCH:
            return self.backend.reshape(x, [np.prod(np.array(list(x.shape)))])
        if self.BACKEND == self.NUMPY:
            return x.flatten()

    def bk_size(self, x):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.size(x)
        if self.BACKEND == self.TORCH:
            return x.numel()

        if self.BACKEND == self.NUMPY:
            return x.size

    def bk_resize_image(self, x, shape):
        if self.BACKEND == self.TENSORFLOW:
            return self.bk_cast(self.backend.image.resize(x, shape, method="bilinear"))

        if self.BACKEND == self.TORCH:
            tmp = self.backend.nn.functional.interpolate(
                x, size=shape, mode="bilinear", align_corners=False
            )
            return self.bk_cast(tmp)
        if self.BACKEND == self.NUMPY:
            return self.bk_cast(self.backend.image.resize(x, shape, method="bilinear"))

    def bk_L1(self, x):
        if x.dtype == self.all_cbk_type:
            xr = self.bk_real(x)
            xi = self.bk_imag(x)

            r = self.backend.sign(xr) * self.backend.sqrt(self.backend.sign(xr) * xr)
            # return r
            i = self.backend.sign(xi) * self.backend.sqrt(self.backend.sign(xi) * xi)

            if self.BACKEND == self.TORCH:
                return r
            else:
                return self.bk_complex(r, i)
        else:
            return self.backend.sign(x) * self.backend.sqrt(self.backend.sign(x) * x)

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
            if self.BACKEND == self.TENSORFLOW:
                return self.backend.reduce_sum(data)
            if self.BACKEND == self.TORCH:
                return self.backend.sum(data)
            if self.BACKEND == self.NUMPY:
                return np.sum(data)
        else:
            if self.BACKEND == self.TENSORFLOW:
                return self.backend.reduce_sum(data, axis=axis)
            if self.BACKEND == self.TORCH:
                return self.backend.sum(data, axis)
            if self.BACKEND == self.NUMPY:
                return np.sum(data, axis)

    # ---------------------------------------------−---------
    # return a tensor size
    
    def bk_size(self, data):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.size(data)
        if self.BACKEND == self.TORCH:
            return data.numel()
        if self.BACKEND == self.NUMPY:
            return data.size
        
    # ---------------------------------------------−---------

    def iso_mean(self, x, use_2D=False):
        shape = list(x.shape)

        i_orient = 2
        if use_2D:
            i_orient = 3
        norient = shape[i_orient]

        if len(shape) == i_orient + 1:
            return self.bk_reduce_mean(x, -1)

        if norient not in self._iso_orient:
            self.calc_iso_orient(norient)

        if self.bk_is_complex(x):
            lmat = self._iso_orient_C[norient]
        else:
            lmat = self._iso_orient[norient]

        oshape = shape[0]
        for k in range(1, len(shape) - 2):
            oshape *= shape[k]

        oshape2 = [shape[k] for k in range(0, len(shape) - 1)]

        return self.bk_reshape(
            self.backend.matmul(self.bk_reshape(x, [oshape, norient * norient]), lmat),
            oshape2,
        )

    def fft_ang(self, x, nharm=1, imaginary=False, use_2D=False):
        shape = list(x.shape)

        i_orient = 2
        if use_2D:
            i_orient = 3

        norient = shape[i_orient]
        nout = 1 + nharm

        oshape_1 = shape[0]
        for k in range(1, i_orient):
            oshape_1 *= shape[k]
        oshape_2 = norient
        for k in range(i_orient, len(shape) - 1):
            oshape_2 *= shape[k]
        oshape = [oshape_1, oshape_2]

        if imaginary:
            nout = 1 + nharm * 2

        oshape2 = [shape[k] for k in range(0, i_orient)] + [
            nout for k in range(i_orient, len(shape))
        ]

        if (norient, nharm) not in self._fft_1_orient:
            self.calc_fft_orient(norient, nharm, imaginary)

        if len(shape) == i_orient + 1:
            if self.bk_is_complex(x):
                lmat = self._fft_1_orient_C[(norient, nharm, imaginary)]
            else:
                lmat = self._fft_1_orient[(norient, nharm, imaginary)]

        if len(shape) == i_orient + 2:
            if self.bk_is_complex(x):
                lmat = self._fft_2_orient_C[(norient, nharm, imaginary)]
            else:
                lmat = self._fft_2_orient[(norient, nharm, imaginary)]

        if len(shape) == i_orient + 3:
            if self.bk_is_complex(x):
                lmat = self._fft_3_orient_C[(norient, nharm, imaginary)]
            else:
                lmat = self._fft_3_orient[(norient, nharm, imaginary)]

        return self.bk_reshape(
            self.backend.matmul(self.bk_reshape(x, oshape), lmat), oshape2
        )

    def constant(self, data):

        if self.BACKEND == self.TENSORFLOW:
            return self.backend.constant(data)
        return data

    def bk_reduce_mean(self, data, axis=None):

        if axis is None:
            if self.BACKEND == self.TENSORFLOW:
                return self.backend.reduce_mean(data)
            if self.BACKEND == self.TORCH:
                return self.backend.mean(data)
            if self.BACKEND == self.NUMPY:
                return np.mean(data)
        else:
            if self.BACKEND == self.TENSORFLOW:
                return self.backend.reduce_mean(data, axis=axis)
            if self.BACKEND == self.TORCH:
                return self.backend.mean(data, axis)
            if self.BACKEND == self.NUMPY:
                return np.mean(data, axis)

    def bk_reduce_min(self, data, axis=None):

        if axis is None:
            if self.BACKEND == self.TENSORFLOW:
                return self.backend.reduce_min(data)
            if self.BACKEND == self.TORCH:
                return self.backend.min(data)
            if self.BACKEND == self.NUMPY:
                return np.min(data)
        else:
            if self.BACKEND == self.TENSORFLOW:
                return self.backend.reduce_min(data, axis=axis)
            if self.BACKEND == self.TORCH:
                return self.backend.min(data, axis)
            if self.BACKEND == self.NUMPY:
                return np.min(data, axis)

    def bk_random_seed(self, value):

        if self.BACKEND == self.TENSORFLOW:
            return self.backend.random.set_seed(value)
        if self.BACKEND == self.TORCH:
            return self.backend.random.set_seed(value)
        if self.BACKEND == self.NUMPY:
            return np.random.seed(value)

    def bk_random_uniform(self, shape):

        if self.BACKEND == self.TENSORFLOW:
            return self.backend.random.uniform(shape)
        if self.BACKEND == self.TORCH:
            return self.backend.random.uniform(shape)
        if self.BACKEND == self.NUMPY:
            return np.random.rand(shape)

    def bk_reduce_std(self, data, axis=None):
        if axis is None:
            if self.BACKEND == self.TENSORFLOW:
                r = self.backend.math.reduce_std(data)
            if self.BACKEND == self.TORCH:
                r = self.backend.std(data)
            if self.BACKEND == self.NUMPY:
                r = np.std(data)
            return self.bk_complex(r, 0 * r)
        else:
            if self.BACKEND == self.TENSORFLOW:
                r = self.backend.math.reduce_std(data, axis=axis)
            if self.BACKEND == self.TORCH:
                r = self.backend.std(data, axis)
            if self.BACKEND == self.NUMPY:
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

        if self.BACKEND == self.TENSORFLOW:
            if isinstance(data, np.ndarray):
                return data.dtype == "complex64" or data.dtype == "complex128"
            return data.dtype.is_complex

        if self.BACKEND == self.TORCH:
            if isinstance(data, np.ndarray):
                return data.dtype == "complex64" or data.dtype == "complex128"

            return data.dtype.is_complex

        if self.BACKEND == self.NUMPY:
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

        if self.BACKEND == self.TENSORFLOW:
            return self.backend.square(data)
        if self.BACKEND == self.TORCH:
            return self.backend.square(data)
        if self.BACKEND == self.NUMPY:
            return data * data

    def bk_log(self, data):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.math.log(data)
        if self.BACKEND == self.TORCH:
            return self.backend.log(data)
        if self.BACKEND == self.NUMPY:
            return np.log(data)

    def bk_matmul(self, a, b):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.matmul(a, b)
        if self.BACKEND == self.TORCH:
            return self.backend.matmul(a, b)
        if self.BACKEND == self.NUMPY:
            return np.dot(a, b)

    def bk_tensor(self, data):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.constant(data)
        if self.BACKEND == self.TORCH:
            return self.backend.constant(data)
        if self.BACKEND == self.NUMPY:
            return data

    def bk_complex(self, real, imag):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.dtypes.complex(real, imag)
        if self.BACKEND == self.TORCH:
            return self.backend.complex(real, imag)
        if self.BACKEND == self.NUMPY:
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
        if self.BACKEND == self.TORCH:
            if isinstance(data, np.ndarray):
                return data.reshape(shape)

        return self.backend.reshape(data, shape)

    def bk_repeat(self, data, nn, axis=0):
        return self.backend.repeat(data, nn, axis=axis)

    def bk_tile(self, data, nn,axis=0):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.tile(data, [nn])
        
        return self.backend.tile(data, nn)

    def bk_roll(self, data, nn, axis=0):
        return self.backend.roll(data, nn, axis=axis)

    def bk_expand_dims(self, data, axis=0):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.expand_dims(data, axis=axis)
        if self.BACKEND == self.TORCH:
            if isinstance(data, np.ndarray):
                data = self.backend.from_numpy(data)
            return self.backend.unsqueeze(data, axis)
        if self.BACKEND == self.NUMPY:
            return np.expand_dims(data, axis)

    def bk_transpose(self, data, thelist):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.transpose(data, thelist)
        if self.BACKEND == self.TORCH:
            return self.backend.transpose(data, thelist)
        if self.BACKEND == self.NUMPY:
            return np.transpose(data, thelist)

    def bk_concat(self, data, axis=None):

        if self.BACKEND == self.TENSORFLOW or self.BACKEND == self.TORCH:
            if axis is None:
                if data[0].dtype == self.all_cbk_type:
                    ndata = len(data)
                    xr = self.backend.concat(
                        [self.bk_real(data[k]) for k in range(ndata)]
                    )
                    xi = self.backend.concat(
                        [self.bk_imag(data[k]) for k in range(ndata)]
                    )
                    return self.backend.complex(xr, xi)
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
                    return self.backend.complex(xr, xi)
                else:
                    return self.backend.concat(data, axis=axis)
        else:
            if axis is None:
                return np.concatenate(data, axis=0)
            else:
                return np.concatenate(data, axis=axis)

    def bk_zeros(self, shape,dtype=None):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.zeros(shape,dtype=dtype)
        if self.BACKEND == self.TORCH:
            return self.backend.zeros(shape,dtype=dtype)
        if self.BACKEND == self.NUMPY:
            return np.zeros(shape,dtype=dtype)

    def bk_gather(self, data,idx):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.gather(data,idx)
        if self.BACKEND == self.TORCH:
            return data[idx]
        if self.BACKEND == self.NUMPY:
            return data[idx]
        
    def bk_reverse(self, data,axis=0):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.reverse(data,axis=[axis])
        if self.BACKEND == self.TORCH:
            return self.backend.reverse(data,axis=axis)
        if self.BACKEND == self.NUMPY:
            return np.reverse(data,axis=axis)
        
    def bk_fft(self, data):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.signal.fft(data)
        if self.BACKEND == self.TORCH:
            return self.backend.fft(data)
        if self.BACKEND == self.NUMPY:
            return self.backend.fft.fft(data)
        
    def bk_rfft(self, data):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.signal.rfft(data)
        if self.BACKEND == self.TORCH:
            return self.backend.rfft(data)
        if self.BACKEND == self.NUMPY:
            return self.backend.fft.rfft(data)

        
    def bk_irfft(self, data):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.signal.irfft(data)
        if self.BACKEND == self.TORCH:
            return self.backend.irfft(data)
        if self.BACKEND == self.NUMPY:
            return self.backend.fft.irfft(data)
        
    def bk_conjugate(self, data):

        if self.BACKEND == self.TENSORFLOW:
            return self.backend.math.conj(data)
        if self.BACKEND == self.TORCH:
            return self.backend.conj(data)
        if self.BACKEND == self.NUMPY:
            return data.conjugate()

    def bk_real(self, data):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.math.real(data)
        if self.BACKEND == self.TORCH:
            return data.real
        if self.BACKEND == self.NUMPY:
            return data.real

    def bk_imag(self, data):
        if self.BACKEND == self.TENSORFLOW:
            return self.backend.math.imag(data)
        if self.BACKEND == self.TORCH:
            if data.dtype == self.all_cbk_type:
                return data.imag
            else:
                return 0

        if self.BACKEND == self.NUMPY:
            return data.imag

    def bk_relu(self, x):
        if self.BACKEND == self.TENSORFLOW:
            if x.dtype == self.all_cbk_type:
                xr = self.backend.nn.relu(self.bk_real(x))
                xi = self.backend.nn.relu(self.bk_imag(x))
                return self.backend.complex(xr, xi)
            else:
                return self.backend.nn.relu(x)
        if self.BACKEND == self.TORCH:
            return self.backend.relu(x)
        if self.BACKEND == self.NUMPY:
            return (x > 0) * x

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

        if isinstance(x, np.int32) or isinstance(x, np.int64) or isinstance(x, int):
            if self.all_bk_type == "float64":
                return np.float64(x)
            else:
                return np.float32(x)

        if self.bk_is_complex(x):
            out_type = self.all_cbk_type
        else:
            out_type = self.all_bk_type

        if self.BACKEND == self.TENSORFLOW:
            return self.backend.cast(x, out_type)

        if self.BACKEND == self.TORCH:
            if isinstance(x, np.ndarray):
                x = self.backend.from_numpy(x)

            if x.dtype.is_complex:
                out_type = self.all_cbk_type
            else:
                out_type = self.all_bk_type

            return x.type(out_type)

        if self.BACKEND == self.NUMPY:
            return x.astype(out_type)
