import numpy as np


class BackendBase:

    def __init__(self, name, mpi_rank=0, all_type="float64", gpupos=0, silent=False):

        self.BACKEND = name
        self.mpi_rank = mpi_rank
        self.all_type = all_type
        self.gpupos = gpupos
        self.silent = silent
        # ---------------------------------------------−---------
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

    def to_dict(self):
        return {
            "name": self.BACKEND,
            "mpi_rank": self.mpi_rank,
            "all_type": self.all_type,
            "gpupos": self.gpupos,
            "silent": self.silent,
        }

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

    def calc_iso_orient(self, norient):
        tmp = np.zeros([norient * norient, norient])
        for i in range(norient):
            for j in range(norient):
                tmp[j * norient + (j + i) % norient, i] = 0.25

        self._iso_orient[norient] = self.bk_constant(self.bk_cast(tmp))
        self._iso_orient_T[norient] = self.bk_constant(self.bk_cast(4 * tmp.T))
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
                self.bk_constant(tmp)
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
                self.bk_constant(tmp)
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
                self.bk_constant(
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
                self.bk_constant(
                    tmp.reshape(norient * norient, (1 + nharm) * (1 + nharm))
                )
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
                self.bk_constant(
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
                self.bk_constant(
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
    def bk_len(self,S):
        raise NotImplementedError("This is an abstract class.")
        
    def bk_SparseTensor(self, indice, w, dense_shape=[]):
        raise NotImplementedError("This is an abstract class.")

    def bk_stack(self, list, axis=0):
        raise NotImplementedError("This is an abstract class.")

    def bk_sparse_dense_matmul(self, smat, mat):
        raise NotImplementedError("This is an abstract class.")

    def conv2d(self, x, w, strides=[1, 1, 1, 1], padding="SAME"):
        raise NotImplementedError("This is an abstract class.")

    def conv1d(self, x, w, strides=[1, 1, 1], padding="SAME"):
        raise NotImplementedError("This is an abstract class.")

    def bk_threshold(self, x, threshold, greater=True):
        raise NotImplementedError("This is an abstract class.")

    def bk_maximum(self, x1, x2):
        raise NotImplementedError("This is an abstract class.")

    def bk_device(self, device_name):
        raise NotImplementedError("This is an abstract class.")

    def bk_ones(self, shape, dtype=None):
        raise NotImplementedError("This is an abstract class.")

    def bk_conv1d(self, x, w):
        raise NotImplementedError("This is an abstract class.")

    def bk_flattenR(self, x):
        raise NotImplementedError("This is an abstract class.")

    def bk_flatten(self, x):
        raise NotImplementedError("This is an abstract class.")

    def bk_size(self, x):
        raise NotImplementedError("This is an abstract class.")

    def bk_resize_image(self, x, shape):
        raise NotImplementedError("This is an abstract class.")

    def bk_L1(self, x):
        raise NotImplementedError("This is an abstract class.")

    def bk_square_comp(self, x):
        raise NotImplementedError("This is an abstract class.")

    def bk_reduce_sum(self, data, axis=None):
        raise NotImplementedError("This is an abstract class.")

    def bk_reduce_mean(self, data, axis=None):
        raise NotImplementedError("This is an abstract class.")

    def bk_reduce_min(self, data, axis=None):
        raise NotImplementedError("This is an abstract class.")

    def bk_random_seed(self, value):
        raise NotImplementedError("This is an abstract class.")

    def bk_random_uniform(self, shape):
        raise NotImplementedError("This is an abstract class.")

    def bk_reduce_std(self, data, axis=None):
        raise NotImplementedError("This is an abstract class.")

    def bk_sqrt(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_abs(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_is_complex(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_distcomp(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_norm(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_square(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_log(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_matmul(self, a, b):
        raise NotImplementedError("This is an abstract class.")

    def bk_tensor(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_shape_tensor(self, shape):
        raise NotImplementedError("This is an abstract class.")

    def bk_complex(self, real, imag):
        raise NotImplementedError("This is an abstract class.")

    def bk_exp(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_min(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_argmin(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_tanh(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_max(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_argmax(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_reshape(self, data, shape):
        raise NotImplementedError("This is an abstract class.")

    def bk_repeat(self, data, nn, axis=0):
        raise NotImplementedError("This is an abstract class.")

    def bk_tile(self, data, nn, axis=0):
        raise NotImplementedError("This is an abstract class.")

    def bk_roll(self, data, nn, axis=0):
        raise NotImplementedError("This is an abstract class.")

    def bk_expand_dims(self, data, axis=0):
        raise NotImplementedError("This is an abstract class.")

    def bk_transpose(self, data, thelist):
        raise NotImplementedError("This is an abstract class.")

    def bk_concat(self, data, axis=None):
        raise NotImplementedError("This is an abstract class.")

    def bk_zeros(self, shape, dtype=None):
        raise NotImplementedError("This is an abstract class.")

    def bk_gather(self, data, idx):
        raise NotImplementedError("This is an abstract class.")

    def bk_reverse(self, data, axis=0):
        raise NotImplementedError("This is an abstract class.")

    def bk_fft(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_fftn(self, data, dim=None):
        raise NotImplementedError("This is an abstract class.")

    def bk_ifftn(self, data, dim=None, norm=None):
        raise NotImplementedError("This is an abstract class.")

    def bk_rfft(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_irfft(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_conjugate(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_real(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_imag(self, data):
        raise NotImplementedError("This is an abstract class.")

    def bk_relu(self, x):
        raise NotImplementedError("This is an abstract class.")

    def bk_clip_by_value(self, x, xmin, xmax):
        raise NotImplementedError("This is an abstract class.")

    def bk_cast(self, x):
        raise NotImplementedError("This is an abstract class.")

    def bk_variable(self, x):
        raise NotImplementedError("This is an abstract class.")

    def bk_assign(self, x, y):
        raise NotImplementedError("This is an abstract class.")

    def bk_constant(self, x):
        raise NotImplementedError("This is an abstract class.")

    def bk_cos(self, x):
        raise NotImplementedError("This is an abstract class.")

    def bk_sin(self, x):
        raise NotImplementedError("This is an abstract class.")

    def bk_arctan2(self, c, s):
        raise NotImplementedError("This is an abstract class.")

    def bk_empty(self, list):
        raise NotImplementedError("This is an abstract class.")

    def to_numpy(self, x):
        raise NotImplementedError("This is an abstract class.")
