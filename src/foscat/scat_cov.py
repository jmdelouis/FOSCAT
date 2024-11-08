import pickle
import sys

import healpy as hp
import numpy as np

import foscat as foscat

# import foscat.backend as bk
import foscat.FoCUS as FOC

# Vérifier si TensorFlow est importé et défini
tf_defined = "tensorflow" in sys.modules

if tf_defined:
    import tensorflow as tf

    tf_function = (
        tf.function
    )  # Facultatif : si vous voulez utiliser TensorFlow dans ce script
else:

    def tf_function(func):
        return func


def read(filename):
    thescat = scat_cov(1, 1, 1, 1)
    return thescat.read(filename)


testwarn = 0


class scat_cov:
    def __init__(
        self, s0, p00, c01, c11, s1=None, c10=None, backend=None, use_1D=False
    ):
        self.S0 = s0
        self.P00 = p00
        self.C01 = c01
        self.C11 = c11
        self.S1 = s1
        self.C10 = c10
        self.backend = backend
        self.idx1 = None
        self.idx2 = None
        self.use_1D = use_1D

    def numpy(self):
        if self.BACKEND == "numpy":
            return self

        if self.S1 is None:
            s1 = None
        else:
            s1 = self.S1.numpy()
        if self.C10 is None:
            c10 = None
        else:
            c10 = self.C10.numpy()

        return scat_cov(
            (self.S0.numpy()),
            (self.P00.numpy()),
            (self.C01.numpy()),
            (self.C11.numpy()),
            s1=s1,
            c10=c10,
            backend=self.backend,
            use_1D=self.use_1D,
        )

    def constant(self):

        if self.S1 is None:
            s1 = None
        else:
            s1 = self.backend.constant(self.S1)
        if self.C10 is None:
            c10 = None
        else:
            c10 = self.backend.constant(self.C10)

        return scat_cov(
            self.backend.constant(self.S0),
            self.backend.constant(self.P00),
            self.backend.constant(self.C01),
            self.backend.constant(self.C11),
            s1=s1,
            c10=c10,
            backend=self.backend,
            use_1D=self.use_1D,
        )

    def conv2complex(self, val):
        if val.dtype == "complex64" or val.dtype == "complex128":
            return val
        else:
            return self.backend.bk_complex(val, 0 * val)
        return val

    # ---------------------------------------------−---------
    def flatten(self):
        tmp = [
            self.conv2complex(
                self.backend.bk_reshape(self.S0, [self.S1.shape[0], self.S0.shape[1]])
            )
        ]
        if self.use_1D:
            if self.S1 is not None:
                tmp = tmp + [
                    self.conv2complex(
                        self.backend.bk_reshape(
                            self.S1,
                            [self.S1.shape[0], self.S1.shape[1] * self.S1.shape[2]],
                        )
                    )
                ]
            tmp = tmp + [
                self.conv2complex(
                    self.backend.bk_reshape(
                        self.P00,
                        [self.S1.shape[0], self.S1.shape[1] * self.S1.shape[2]],
                    )
                ),
                self.conv2complex(
                    self.backend.bk_reshape(
                        self.C01,
                        [self.C01.shape[0], self.C01.shape[1] * self.C01.shape[2]],
                    )
                ),
            ]
            if self.C10 is not None:
                tmp = tmp + [
                    self.conv2complex(
                        self.backend.bk_reshape(
                            self.C10,
                            [self.C01.shape[0], self.C01.shape[1] * self.C01.shape[2]],
                        )
                    )
                ]

            tmp = tmp + [
                self.conv2complex(
                    self.backend.bk_reshape(
                        self.C11,
                        [self.C01.shape[0], self.C11.shape[1] * self.C11.shape[2]],
                    )
                )
            ]

            return self.backend.bk_concat(tmp, 1)

        if self.S1 is not None:
            tmp = tmp + [
                self.conv2complex(
                    self.backend.bk_reshape(
                        self.S1,
                        [
                            self.S1.shape[0],
                            self.S1.shape[1] * self.S1.shape[2] * self.S1.shape[3],
                        ],
                    )
                )
            ]
        tmp = tmp + [
            self.conv2complex(
                self.backend.bk_reshape(
                    self.P00,
                    [
                        self.S1.shape[0],
                        self.S1.shape[1] * self.S1.shape[2] * self.S1.shape[3],
                    ],
                )
            ),
            self.conv2complex(
                self.backend.bk_reshape(
                    self.C01,
                    [
                        self.C01.shape[0],
                        self.C01.shape[1]
                        * self.C01.shape[2]
                        * self.C01.shape[3]
                        * self.C01.shape[4],
                    ],
                )
            ),
        ]
        if self.C10 is not None:
            tmp = tmp + [
                self.conv2complex(
                    self.backend.bk_reshape(
                        self.C10,
                        [
                            self.C01.shape[0],
                            self.C01.shape[1]
                            * self.C01.shape[2]
                            * self.C01.shape[3]
                            * self.C01.shape[4],
                        ],
                    )
                )
            ]

        tmp = tmp + [
            self.conv2complex(
                self.backend.bk_reshape(
                    self.C11,
                    [
                        self.C01.shape[0],
                        self.C11.shape[1]
                        * self.C11.shape[2]
                        * self.C11.shape[3]
                        * self.C11.shape[4]
                        * self.C11.shape[5],
                    ],
                )
            )
        ]

        return self.backend.bk_concat(tmp, 1)

    # ---------------------------------------------−---------
    def flattenMask(self):
        if isinstance(self.P00, np.ndarray):
            if self.S1 is None:
                if self.C10 is None:
                    tmp = np.concatenate(
                        [
                            self.S0[0].flatten(),
                            self.P00[0].flatten(),
                            self.C01[0].flatten(),
                            self.C11[0].flatten(),
                        ],
                        0,
                    )
                else:
                    tmp = np.concatenate(
                        [
                            self.S0[0].flatten(),
                            self.P00[0].flatten(),
                            self.C01[0].flatten(),
                            self.C10[0].flatten(),
                            self.C11[0].flatten(),
                        ],
                        0,
                    )
            else:
                if self.C10 is None:
                    tmp = np.concatenate(
                        [
                            self.S0[0].flatten(),
                            self.S1[0].flatten(),
                            self.P00[0].flatten(),
                            self.C01[0].flatten(),
                            self.C11[0].flatten(),
                        ],
                        0,
                    )
                else:
                    tmp = np.concatenate(
                        [
                            self.S0[0].flatten(),
                            self.S1[0].flatten(),
                            self.P00[0].flatten(),
                            self.C01[0].flatten(),
                            self.C10[0].flatten(),
                            self.C11[0].flatten(),
                        ],
                        0,
                    )
            tmp = np.expand_dims(tmp, 0)

            for k in range(1, self.P00.shape[0]):
                if self.S1 is None:
                    if self.C10 is None:
                        ltmp = np.concatenate(
                            [
                                self.S0[k].flatten(),
                                self.P00[k].flatten(),
                                self.C01[k].flatten(),
                                self.C11[k].flatten(),
                            ],
                            0,
                        )
                    else:
                        ltmp = np.concatenate(
                            [
                                self.S0[k].flatten(),
                                self.P00[k].flatten(),
                                self.C01[k].flatten(),
                                self.C10[k].flatten(),
                                self.C11[k].flatten(),
                            ],
                            0,
                        )
                else:
                    if self.C10 is None:
                        ltmp = np.concatenate(
                            [
                                self.S0[k].flatten(),
                                self.S1[k].flatten(),
                                self.P00[k].flatten(),
                                self.C01[k].flatten(),
                                self.C11[k].flatten(),
                            ],
                            0,
                        )
                    else:
                        ltmp = np.concatenate(
                            [
                                self.S0[k].flatten(),
                                self.S1[k].flatten(),
                                self.P00[k].flatten(),
                                self.C01[k].flatten(),
                                self.C10[k].flatten(),
                                self.C11[k].flatten(),
                            ],
                            0,
                        )

                tmp = np.concatenate([tmp, np.expand_dims(ltmp, 0)], 0)

            return tmp
        else:
            if self.S1 is None:
                if self.C10 is None:
                    tmp = self.backend.bk_concat(
                        [
                            self.backend.bk_flattenR(self.S0[0]),
                            self.backend.bk_flattenR(self.P00[0]),
                            self.backend.bk_flattenR(self.C01[0]),
                            self.backend.bk_flattenR(self.C11[0]),
                        ],
                        0,
                    )
                else:
                    tmp = self.backend.bk_concat(
                        [
                            self.backend.bk_flattenR(self.S0[0]),
                            self.backend.bk_flattenR(self.P00[0]),
                            self.backend.bk_flattenR(self.C01[0]),
                            self.backend.bk_flattenR(self.C10[0]),
                            self.backend.bk_flattenR(self.C11[0]),
                        ],
                        0,
                    )
            else:
                if self.C10 is None:
                    tmp = self.backend.bk_concat(
                        [
                            self.backend.bk_flattenR(self.S0[0]),
                            self.backend.bk_flattenR(self.S1[0]),
                            self.backend.bk_flattenR(self.P00[0]),
                            self.backend.bk_flattenR(self.C01[0]),
                            self.backend.bk_flattenR(self.C11[0]),
                        ],
                        0,
                    )
                else:
                    tmp = self.backend.bk_concat(
                        [
                            self.backend.bk_flattenR(self.S0[0]),
                            self.backend.bk_flattenR(self.S1[0]),
                            self.backend.bk_flattenR(self.P00[0]),
                            self.backend.bk_flattenR(self.C01[0]),
                            self.backend.bk_flattenR(self.C10[0]),
                            self.backend.bk_flattenR(self.C11[0]),
                        ],
                        0,
                    )
            tmp = self.backend.bk_expand_dims(tmp, 0)

            for k in range(1, self.P00.shape[0]):
                if self.S1 is None:
                    if self.C10 is None:
                        ltmp = self.backend.bk_concat(
                            [
                                self.backend.bk_flattenR(self.S0[k]),
                                self.backend.bk_flattenR(self.P00[k]),
                                self.backend.bk_flattenR(self.C01[k]),
                                self.backend.bk_flattenR(self.C11[k]),
                            ],
                            0,
                        )
                    else:
                        ltmp = self.backend.bk_concat(
                            [
                                self.backend.bk_flattenR(self.S0[k]),
                                self.backend.bk_flattenR(self.P00[k]),
                                self.backend.bk_flattenR(self.C01[k]),
                                self.backend.bk_flattenR(self.C10[k]),
                                self.backend.bk_flattenR(self.C11[k]),
                            ],
                            0,
                        )
                else:
                    if self.C10 is None:
                        ltmp = self.backend.bk_concat(
                            [
                                self.backend.bk_flattenR(self.S0[k]),
                                self.backend.bk_flattenR(self.S1[k]),
                                self.backend.bk_flattenR(self.P00[k]),
                                self.backend.bk_flattenR(self.C01[k]),
                                self.backend.bk_flattenR(self.C11[k]),
                            ],
                            0,
                        )
                    else:
                        ltmp = self.backend.bk_concat(
                            [
                                self.backend.bk_flattenR(self.S0[k]),
                                self.backend.bk_flattenR(self.S1[k]),
                                self.backend.bk_flattenR(self.P00[k]),
                                self.backend.bk_flattenR(self.C01[k]),
                                self.backend.bk_flattenR(self.C10[k]),
                                self.backend.bk_flattenR(self.C11[k]),
                            ],
                            0,
                        )

                tmp = self.backend.bk_concat(
                    [tmp, self.backend.bk_expand_dims(ltmp, 0)], 0
                )

            return tmp

    def get_S0(self):
        return self.S0

    def get_S1(self):
        return self.S1

    def get_P00(self):
        return self.P00

    def reset_P00(self):
        self.P00 = 0 * self.P00

    def get_C01(self):
        return self.C01

    def get_C10(self):
        return self.C10

    def get_C11(self):
        return self.C11

    def get_j_idx(self):
        shape = list(self.P00.shape)
        if len(shape) == 3:
            nscale = shape[2]
        elif len(shape) == 4:
            nscale = shape[2]
        else:
            nscale = shape[3]

        n = nscale * (nscale + 1) // 2
        j1 = np.zeros([n], dtype="int")
        j2 = np.zeros([n], dtype="int")
        n = 0
        for i in range(nscale):
            for j in range(i + 1):
                j1[n] = j
                j2[n] = i
                n = n + 1

        return j1, j2

    def get_jc11_idx(self):
        shape = list(self.P00.shape)
        nscale = shape[2]
        n = nscale * np.max([nscale - 1, 1]) * np.max([nscale - 2, 1])
        j1 = np.zeros([n * 4], dtype="int")
        j2 = np.zeros([n * 4], dtype="int")
        j3 = np.zeros([n * 4], dtype="int")
        n = 0
        for i in range(nscale):
            for j in range(i + 1):
                for k in range(j + 1):
                    j1[n] = k
                    j2[n] = j
                    j3[n] = i
                    n = n + 1
        return (j1[0:n], j2[0:n], j3[0:n])

    def __add__(self, other):
        assert (
            isinstance(other, float)
            or isinstance(other, np.float32)
            or isinstance(other, int)
            or isinstance(other, bool)
            or isinstance(other, scat_cov)
        )

        if self.S1 is None:
            s1 = None
        else:
            if isinstance(other, scat_cov):
                if other.S1 is None:
                    s1 = None
                else:
                    s1 = self.S1 + other.S1
            else:
                s1 = self.S1 + other

        if self.C10 is None:
            c10 = None
        else:
            if isinstance(other, scat_cov):
                if other.C10 is None:
                    c10 = None
                else:
                    c10 = self.doadd(self.C10, other.C10)
            else:
                c10 = self.C10 + other

        if self.C11 is None:
            c11 = None
        else:
            if isinstance(other, scat_cov):
                if other.C11 is None:
                    c11 = None
                else:
                    c11 = self.doadd(self.C11, other.C11)
            else:
                c11 = self.C11 + other

        if isinstance(other, scat_cov):
            return scat_cov(
                self.doadd(self.S0, other.S0),
                self.doadd(self.P00, other.P00),
                (self.C01 + other.C01),
                c11,
                s1=s1,
                c10=c10,
                backend=self.backend,
                use_1D=self.use_1D,
            )
        else:
            return scat_cov(
                (self.S0 + other),
                (self.P00 + other),
                (self.C01 + other),
                c11,
                s1=s1,
                c10=c10,
                backend=self.backend,
                use_1D=self.use_1D,
            )

    def relu(self):

        if self.S1 is None:
            s1 = None
        else:
            s1 = self.backend.bk_relu(self.S1)

        if self.C10 is None:
            c10 = None
        else:
            c10 = self.backend.bk_relu(self.c10)

        if self.C11 is None:
            c11 = None
        else:
            c11 = self.backend.bk_relu(self.c11)

        return scat_cov(
            self.backend.bk_relu(self.S0),
            self.backend.bk_relu(self.P00),
            self.backend.bk_relu(self.C01),
            c11,
            s1=s1,
            c10=c10,
            backend=self.backend,
            use_1D=self.use_1D,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __truediv__(self, other):
        assert (
            isinstance(other, float)
            or isinstance(other, np.float32)
            or isinstance(other, int)
            or isinstance(other, bool)
            or isinstance(other, scat_cov)
        )

        if self.S1 is None:
            s1 = None
        else:
            if isinstance(other, scat_cov):
                if other.S1 is None:
                    s1 = None
                else:
                    s1 = self.dodiv(self.S1, other.S1)
            else:
                s1 = self.dodiv(self.S1, other)

        if self.C10 is None:
            c10 = None
        else:
            if isinstance(other, scat_cov):
                if other.C10 is None:
                    c10 = None
                else:
                    c10 = self.dodiv(self.C10, other.C10)
            else:
                c10 = self.dodiv(self.C10, other)

        if self.C11 is None:
            c11 = None
        else:
            if isinstance(other, scat_cov):
                if other.C11 is None:
                    c11 = None
                else:
                    c11 = self.dodiv(self.C11, other.C11)
            else:
                c11 = self.C11 / other

        if isinstance(other, scat_cov):
            return scat_cov(
                self.dodiv(self.S0, other.S0),
                self.dodiv(self.P00, other.P00),
                self.dodiv(self.C01, other.C01),
                c11,
                s1=s1,
                c10=c10,
                backend=self.backend,
                use_1D=self.use_1D,
            )
        else:
            return scat_cov(
                (self.S0 / other),
                (self.P00 / other),
                (self.C01 / other),
                c11,
                s1=s1,
                c10=c10,
                backend=self.backend,
                use_1D=self.use_1D,
            )

    def __rtruediv__(self, other):
        assert (
            isinstance(other, float)
            or isinstance(other, np.float32)
            or isinstance(other, int)
            or isinstance(other, bool)
            or isinstance(other, scat_cov)
        )

        if self.S1 is None:
            s1 = None
        else:
            if isinstance(other, scat_cov):
                s1 = other.S1 / self.S1
            else:
                s1 = other / self.S1

        if self.C10 is None:
            c10 = None
        else:
            if isinstance(other, scat_cov):
                c10 = self.dodiv(other.C10, self.C10)
            else:
                c10 = other / self.C10

        if self.C11 is None:
            c11 = None
        else:
            if isinstance(other, scat_cov):
                if other.C11 is None:
                    c11 = None
                else:
                    c11 = self.dodiv(other.C11, self.C11)
            else:
                c11 = other / self.C11

        if isinstance(other, scat_cov):
            return scat_cov(
                self.dodiv(other.S0, self.S0),
                self.dodiv(other.P00, self.P00),
                (other.C01 / self.C01),
                c11,
                s1=s1,
                c10=c10,
                backend=self.backend,
                use_1D=self.use_1D,
            )
        else:
            return scat_cov(
                (other / self.S0),
                (other / self.P00),
                (other / self.C01),
                (other / self.C11),
                s1=s1,
                c10=c10,
                backend=self.backend,
                use_1D=self.use_1D,
            )

    def __rsub__(self, other):

        assert (
            isinstance(other, float)
            or isinstance(other, np.float32)
            or isinstance(other, int)
            or isinstance(other, bool)
            or isinstance(other, scat_cov)
        )

        if self.S1 is None:
            s1 = None
        else:
            if isinstance(other, scat_cov):
                if other.S1 is None:
                    s1 = None
                else:
                    s1 = other.S1 - self.S1
            else:
                s1 = other - self.S1

        if self.C10 is None:
            c10 = None
        else:
            if isinstance(other, scat_cov):
                if other.C10 is None:
                    c10 = None
                else:
                    c10 = self.domin(other.C10, self.C10)
            else:
                c10 = other - self.C10

        if self.C11 is None:
            c11 = None
        else:
            if isinstance(other, scat_cov):
                if other.C11 is None:
                    c11 = None
                else:
                    c11 = self.domin(other.C11, self.C11)
            else:
                c11 = other - self.C11

        if isinstance(other, scat_cov):
            return scat_cov(
                self.domin(other.S0, self.S0),
                self.domin(other.P00, self.P00),
                (other.C01 - self.C01),
                c11,
                s1=s1,
                c10=c10,
                backend=self.backend,
                use_1D=self.use_1D,
            )
        else:
            return scat_cov(
                (other - self.S0),
                (other - self.P00),
                (other - self.C01),
                c11,
                s1=s1,
                c10=c10,
                backend=self.backend,
                use_1D=self.use_1D,
            )

    def __sub__(self, other):
        assert (
            isinstance(other, float)
            or isinstance(other, np.float32)
            or isinstance(other, int)
            or isinstance(other, bool)
            or isinstance(other, scat_cov)
        )

        if self.S1 is None:
            s1 = None
        else:
            if isinstance(other, scat_cov):
                if other.S1 is None:
                    s1 = None
                else:
                    s1 = self.S1 - other.S1
            else:
                s1 = self.S1 - other

        if self.C10 is None:
            c10 = None
        else:
            if isinstance(other, scat_cov):
                if other.C10 is None:
                    c10 = None
                else:
                    c10 = self.domin(self.C10, other.C10)
            else:
                c10 = self.C10 - other

        if self.C11 is None:
            c11 = None
        else:
            if isinstance(other, scat_cov):
                if other.C11 is None:
                    c11 = None
                else:
                    c11 = self.domin(self.C11, other.C11)
            else:
                c11 = self.C11 - other

        if isinstance(other, scat_cov):
            return scat_cov(
                self.domin(self.S0, other.S0),
                self.domin(self.P00, other.P00),
                (self.C01 - other.C01),
                c11,
                s1=s1,
                c10=c10,
                backend=self.backend,
                use_1D=self.use_1D,
            )
        else:
            return scat_cov(
                (self.S0 - other),
                (self.P00 - other),
                (self.C01 - other),
                c11,
                s1=s1,
                c10=c10,
                backend=self.backend,
                use_1D=self.use_1D,
            )

    def domult(self, x, y):
        try:
            return x * y
        except:
            if x.dtype == y.dtype:
                return x * y
            if self.backend.bk_is_complex(x):

                return self.backend.bk_complex(
                    self.backend.bk_real(x) * y, self.backend.bk_imag(x) * y
                )
            else:
                return self.backend.bk_complex(
                    self.backend.bk_real(y) * x, self.backend.bk_imag(y) * x
                )

    def dodiv(self, x, y):
        try:
            return x / y
        except:
            if x.dtype == y.dtype:
                return x / y
            if self.backend.bk_is_complex(x):

                return self.backend.bk_complex(
                    self.backend.bk_real(x) / y, self.backend.bk_imag(x) / y
                )
            else:
                return self.backend.bk_complex(
                    x / self.backend.bk_real(y), x / self.backend.bk_imag(y)
                )

    def domin(self, x, y):
        try:
            return x - y
        except:
            if x.dtype == y.dtype:
                return x - y

            if self.backend.bk_is_complex(x):

                return self.backend.bk_complex(
                    self.backend.bk_real(x) - y, self.backend.bk_imag(x) - y
                )
            else:
                return self.backend.bk_complex(
                    x - self.backend.bk_real(y), x - self.backend.bk_imag(y)
                )

    def doadd(self, x, y):
        try:
            return x + y
        except:
            if x.dtype == y.dtype:
                return x + y
            if self.backend.bk_is_complex(x):

                return self.backend.bk_complex(
                    self.backend.bk_real(x) + y, self.backend.bk_imag(x) + y
                )
            else:
                return self.backend.bk_complex(
                    x + self.backend.bk_real(y), x + self.backend.bk_imag(y)
                )

    def __mul__(self, other):
        assert (
            isinstance(other, float)
            or isinstance(other, np.float32)
            or isinstance(other, int)
            or isinstance(other, bool)
            or isinstance(other, scat_cov)
        )

        if self.S1 is None:
            s1 = None
        else:
            if isinstance(other, scat_cov):
                if other.S1 is None:
                    s1 = None
                else:
                    s1 = self.S1 * other.S1
            else:
                s1 = self.S1 * other

        if self.C10 is None:
            c10 = None
        else:
            if isinstance(other, scat_cov):
                if other.C10 is None:
                    c10 = None
                else:
                    c10 = self.domult(self.C10, other.C10)
            else:
                c10 = self.C10 * other

        if self.C11 is None:
            c11 = None
        else:
            if isinstance(other, scat_cov):
                if other.C11 is None:
                    c11 = None
                else:
                    c11 = self.domult(self.C11, other.C11)
            else:
                c11 = self.C11 * other

        if isinstance(other, scat_cov):
            return scat_cov(
                self.domult(self.S0, other.S0),
                self.domult(self.P00, other.P00),
                self.domult(self.C01, other.C01),
                c11,
                s1=s1,
                c10=c10,
                backend=self.backend,
                use_1D=self.use_1D,
            )
        else:
            return scat_cov(
                (self.S0 * other),
                (self.P00 * other),
                (self.C01 * other),
                c11,
                s1=s1,
                c10=c10,
                backend=self.backend,
                use_1D=self.use_1D,
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    # ---------------------------------------------−---------
    def interp(self, nscale, extend=True, constant=False):

        if nscale + 2 > self.P00.shape[2]:
            print(
                "Can not *interp* %d with a statistic described over %d"
                % (nscale, self.P00.shape[2])
            )
            return scat_cov(
                self.P00,
                self.C01,
                self.C11,
                s1=self.S1,
                c10=self.C10,
                backend=self.backend,
            )

        if self.S1 is not None:
            if self.BACKEND == "numpy":
                s1 = self.S1
            else:
                s1 = self.S1.numpy()
        else:
            s1 = self.S1

        if self.BACKEND == "numpy":
            p0 = self.P00
        else:
            p0 = self.P00.numpy()

        for k in range(nscale):
            if constant:
                if self.S1 is not None:
                    s1[:, :, nscale - 1 - k, :] = s1[:, :, nscale - k, :]
                p0[:, :, nscale - 1 - k, :] = p0[:, :, nscale - k, :]
            else:
                if self.S1 is not None:
                    s1[:, :, nscale - 1 - k, :] = np.exp(
                        2 * np.log(s1[:, :, nscale - k, :])
                        - np.log(s1[:, :, nscale + 1 - k, :])
                    )
                p0[:, :, nscale - 1 - k, :] = np.exp(
                    2 * np.log(p0[:, :, nscale - k, :])
                    - np.log(p0[:, :, nscale + 1 - k, :])
                )

        j1, j2 = self.get_j_idx()

        if self.C10 is not None:
            if self.BACKEND == "numpy":
                c10 = self.C10
            else:
                c10 = self.C10.numpy()
        else:
            c10 = self.C10
        if self.BACKEND == "numpy":
            c01 = self.C01
        else:
            c01 = self.C01.numpy()

        for k in range(nscale):

            for l_orient in range(nscale - k):
                i0 = np.where(
                    (j1 == nscale - 1 - k - l_orient) * (j2 == nscale - 1 - k)
                )[0]
                i1 = np.where((j1 == nscale - 1 - k - l_orient) * (j2 == nscale - k))[0]
                i2 = np.where(
                    (j1 == nscale - 1 - k - l_orient) * (j2 == nscale + 1 - k)
                )[0]
                if constant:
                    c10[:, :, i0] = c10[:, :, i1]
                    c01[:, :, i0] = c01[:, :, i1]
                else:
                    c10[:, :, i0] = np.exp(
                        2 * np.log(c10[:, :, i1]) - np.log(c10[:, :, i2])
                    )
                    c01[:, :, i0] = np.exp(
                        2 * np.log(c01[:, :, i1]) - np.log(c01[:, :, i2])
                    )

        if self.BACKEND == "numpy":
            c11 = self.C11
        else:
            c11 = self.C11.numpy()

        j1, j2, j3 = self.get_jc11_idx()

        for k in range(nscale):

            for l_orient in range(nscale - k):
                for m in range(nscale - k - l_orient):
                    i0 = np.where(
                        (j1 == nscale - 1 - k - l_orient - m)
                        * (j2 == nscale - 1 - k - l_orient)
                        * (j3 == nscale - 1 - k)
                    )[0]
                    i1 = np.where(
                        (j1 == nscale - 1 - k - l_orient - m)
                        * (j2 == nscale - 1 - k - l_orient)
                        * (j3 == nscale - k)
                    )[0]
                    i2 = np.where(
                        (j1 == nscale - 1 - k - l_orient - m)
                        * (j2 == nscale - 1 - k - l_orient)
                        * (j3 == nscale + 1 - k)
                    )[0]
                if constant:
                    c11[:, :, i0] = c11[:, :, i1]
                else:
                    c11[:, :, i0] = np.exp(
                        2 * np.log(c11[:, :, i1]) - np.log(c11[:, :, i2])
                    )

        if s1 is not None:
            s1 = self.backend.constant(s1)
        if c10 is not None:
            c10 = self.backend.constant(c10)

        return scat_cov(
            self.S0,
            self.backend.constant(p0),
            self.backend.constant(c01),
            self.backend.constant(c11),
            s1=s1,
            c10=c10,
            backend=self.backend,
            use_1D=self.use_1D,
        )

    def plot(self, name=None, hold=True, color="blue", lw=1, legend=True, norm=False):

        import matplotlib.pyplot as plt

        if name is None:
            name = ""

        j1, j2 = self.get_j_idx()

        if hold:
            plt.figure(figsize=(16, 8))

        test = None
        plt.subplot(2, 2, 2)
        tmp = abs(self.get_np(self.P00))
        ntmp = np.sqrt(tmp)
        if len(tmp.shape) > 3:
            for k in range(tmp.shape[3]):
                for i1 in range(tmp.shape[0]):
                    for i2 in range(tmp.shape[1]):
                        if test is None:
                            test = 1
                            plt.plot(
                                tmp[i1, i2, :, k],
                                color=color,
                                label=r"%s $P_{00}$" % (name),
                                lw=lw,
                            )
                        else:
                            plt.plot(tmp[i1, i2, :, k], color=color, lw=lw)
        else:
            for i1 in range(tmp.shape[0]):
                for i2 in range(tmp.shape[1]):
                    if test is None:
                        test = 1
                        plt.plot(
                            tmp[i1, i2, :],
                            color=color,
                            label=r"%s $P_{00}$" % (name),
                            lw=lw,
                        )
                    else:
                        plt.plot(tmp[i1, i2, :], color=color, lw=lw)
        plt.yscale("log")
        plt.ylabel("P00")
        plt.xlabel(r"$j_{1}$")
        plt.legend()

        if self.S1 is not None:
            plt.subplot(2, 2, 1)
            tmp = abs(self.get_np(self.S1))
            test = None
            if len(tmp.shape) > 3:
                for k in range(tmp.shape[3]):
                    for i1 in range(tmp.shape[0]):
                        for i2 in range(tmp.shape[1]):
                            if test is None:
                                test = 1
                                if norm:
                                    plt.plot(
                                        tmp[
                                            i1,
                                            i2,
                                            :,
                                            k,
                                        ]
                                        / ntmp[i1, i2, :, k],
                                        color=color,
                                        label=r"%s norm. $S_1$" % (name),
                                        lw=lw,
                                    )
                                else:
                                    plt.plot(
                                        tmp[i1, i2, :, k],
                                        color=color,
                                        label=r"%s $S_1$" % (name),
                                        lw=lw,
                                    )
                            else:
                                if norm:
                                    plt.plot(
                                        tmp[i1, i2, :, k] / ntmp[i1, i2, :, k],
                                        color=color,
                                        lw=lw,
                                    )
                                else:
                                    plt.plot(tmp[i1, i2, :, k], color=color, lw=lw)
            else:
                for i1 in range(tmp.shape[0]):
                    for i2 in range(tmp.shape[1]):
                        if test is None:
                            test = 1
                            plt.plot(
                                tmp[i1, i2, :],
                                color=color,
                                label=r"%s $S_1$" % (name),
                                lw=lw,
                            )
                        else:
                            plt.plot(tmp[i1, i2, :], color=color, lw=lw)
            plt.yscale("log")
            plt.legend()
            if norm:
                plt.ylabel(r"$\frac{S_1}{\sqrt{P_{00}}}$")
            else:
                plt.ylabel("$S_1$")
            plt.xlabel(r"$j_{1}$")

        ax1 = plt.subplot(2, 2, 3)
        ax2 = ax1.twiny()
        n = 0
        tmp = abs(self.get_np(self.C01))
        if norm:
            lname = r"%s norm. $C_{01}$" % (name)
            ax1.set_ylabel(r"$\frac{C_{01}}{\sqrt{P_{00,j_1}P_{00,j_2}}}$")
        else:
            lname = r"%s $C_{01}$" % (name)
            ax1.set_ylabel(r"$C_{01}$")

        if self.C10 is not None:
            tmp = abs(self.get_np(self.C01))
            if norm:
                lname = r"%s norm. $C_{10}$" % (name)
                ax1.set_ylabel(r"$\frac{C_{10}}{\sqrt{P_{00,j_1}P_{00,j_2}}}$")
            else:
                lname = r"%s $C_{10}$" % (name)
                ax1.set_ylabel(r"$C_{10}$")

        test = None
        tabx = []
        tabnx = []
        tab2x = []
        tab2nx = []
        if len(tmp.shape) > 4:
            for i0 in range(tmp.shape[0]):
                for i1 in range(tmp.shape[1]):
                    for i2 in range(j1.max() + 1):
                        for i3 in range(tmp.shape[3]):
                            for i4 in range(tmp.shape[4]):
                                dtmp = tmp[i0, i1, j1 == i2, i3, i4]
                                if norm:
                                    dtmp = dtmp / (
                                        ntmp[i0, i1, i2, i3]
                                        * ntmp[i0, i1, j2[j1 == i2], i3]
                                    )
                                if j2[j1 == i2].shape[0] == 1:
                                    ax1.plot(
                                        j2[j1 == i2] + n, dtmp, ".", color=color, lw=lw
                                    )
                                else:
                                    if legend and test is None:
                                        ax1.plot(
                                            j2[j1 == i2] + n,
                                            dtmp,
                                            color=color,
                                            label=lname,
                                            lw=lw,
                                        )
                                        test = 1
                                    ax1.plot(j2[j1 == i2] + n, dtmp, color=color, lw=lw)
                        tabnx = tabnx + [r"%d" % (k) for k in j2[j1 == i2]]
                        tabx = tabx + [k + n for k in j2[j1 == i2]]
                        tab2x = tab2x + [(j2[j1 == i2] + n).mean()]
                        tab2nx = tab2nx + ["%d" % (i2)]
                        ax1.axvline(
                            (j2[j1 == i2] + n).max() + 0.5, ls=":", color="gray"
                        )
                        n = n + j2[j1 == i2].shape[0] - 1
        elif len(tmp.shape) == 3:
            for i0 in range(tmp.shape[0]):
                for i1 in range(tmp.shape[1]):
                    for i2 in range(j1.max() + 1):
                        dtmp = tmp[i0, i1, j1 == i2]
                        if norm:
                            dtmp = dtmp / (
                                ntmp[i0, i1, i2] * ntmp[i0, i1, j2[j1 == i2]]
                            )
                        if j2[j1 == i2].shape[0] == 1:
                            ax1.plot(j2[j1 == i2] + n, dtmp, ".", color=color, lw=lw)
                        else:
                            if legend and test is None:
                                ax1.plot(
                                    j2[j1 == i2] + n,
                                    dtmp,
                                    color=color,
                                    label=lname,
                                    lw=lw,
                                )
                                test = 1
                            ax1.plot(j2[j1 == i2] + n, dtmp, color=color, lw=lw)
                        tabnx = tabnx + [r"%d" % (k) for k in j2[j1 == i2]]
                        tabx = tabx + [k + n for k in j2[j1 == i2]]
                        tab2x = tab2x + [(j2[j1 == i2] + n).mean()]
                        tab2nx = tab2nx + ["%d" % (i2)]
                        ax1.axvline(
                            (j2[j1 == i2] + n).max() + 0.5, ls=":", color="gray"
                        )
                        n = n + j2[j1 == i2].shape[0] - 1

        else:
            for i0 in range(tmp.shape[0]):
                for i1 in range(tmp.shape[1]):
                    for i2 in range(j1.max() + 1):
                        for i3 in range(tmp.shape[3]):
                            dtmp = tmp[i0, i1, j1 == i2, i3]
                            if norm:
                                dtmp = dtmp / (
                                    ntmp[i0, i1, i2] * ntmp[i0, i1, j2[j1 == i2]]
                                )
                            if j2[j1 == i2].shape[0] == 1:
                                ax1.plot(
                                    j2[j1 == i2] + n, dtmp, ".", color=color, lw=lw
                                )
                            else:
                                if legend and test is None:
                                    ax1.plot(
                                        j2[j1 == i2] + n,
                                        dtmp,
                                        color=color,
                                        label=lname,
                                        lw=lw,
                                    )
                                    test = 1
                                ax1.plot(j2[j1 == i2] + n, dtmp, color=color, lw=lw)
                        tabnx = tabnx + [r"%d" % (k) for k in j2[j1 == i2]]
                        tabx = tabx + [k + n for k in j2[j1 == i2]]
                        tab2x = tab2x + [(j2[j1 == i2] + n).mean()]
                        tab2nx = tab2nx + ["%d" % (i2)]
                        ax1.axvline(
                            (j2[j1 == i2] + n).max() + 0.5, ls=":", color="gray"
                        )
                        n = n + j2[j1 == i2].shape[0] - 1
        plt.yscale("log")
        ax1.set_xlim(0, n + 2)
        ax1.set_xticks(tabx)
        ax1.set_xticklabels(tabnx, fontsize=6)
        ax1.set_xlabel(r"$j_{2}$", fontsize=6)

        # Move twinned axis ticks and label from top to bottom
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        # Offset the twin axis below the host
        ax2.spines["bottom"].set_position(("axes", -0.15))

        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)

        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.spines["bottom"].set_visible(True)
        ax2.set_xlim(0, n + 2)
        ax2.set_xticks(tab2x)
        ax2.set_xticklabels(tab2nx, fontsize=6)
        ax2.set_xlabel(r"$j_{1}$", fontsize=6)
        ax1.legend(frameon=0)

        ax1 = plt.subplot(2, 2, 4)
        j1, j2, j3 = self.get_jc11_idx()
        ax2 = ax1.twiny()
        n = 1
        tmp = abs(self.get_np(self.C11))
        lname = r"%s $C_{11}$" % (name)
        test = None
        tabx = []
        tabnx = []
        tab2x = []
        tab2nx = []
        ntmp = ntmp
        if len(tmp.shape) > 4:
            for i0 in range(tmp.shape[0]):
                for i1 in range(tmp.shape[1]):
                    for i2 in range(j1.max() + 1):
                        nprev = n
                        for i2b in range(j2[j1 == i2].max() + 1):
                            idx = np.where((j1 == i2) * (j2 == i2b))[0]
                            for i3 in range(tmp.shape[3]):
                                for i4 in range(tmp.shape[4]):
                                    for i5 in range(tmp.shape[5]):
                                        dtmp = tmp[i0, i1, idx, i3, i4, i5]
                                        if norm:
                                            dtmp = dtmp / (
                                                ntmp[i0, i1, i2, i3]
                                                * ntmp[i0, i1, i2b, i3]
                                            )
                                        if len(idx) == 1:
                                            ax1.plot(
                                                np.arange(len(idx)) + n,
                                                dtmp,
                                                ".",
                                                color=color,
                                                lw=lw,
                                            )
                                        else:
                                            if legend and test is None:
                                                ax1.plot(
                                                    np.arange(len(idx)) + n,
                                                    dtmp,
                                                    color=color,
                                                    label=lname,
                                                    lw=lw,
                                                )
                                                test = 1
                                            ax1.plot(
                                                np.arange(len(idx)) + n,
                                                dtmp,
                                                color=color,
                                                lw=lw,
                                            )
                            tabnx = tabnx + [r"%d,%d" % (j2[k], j3[k]) for k in idx]
                            tabx = tabx + [k + n for k in range(len(idx))]
                            n = n + idx.shape[0]
                        tab2x = tab2x + [(n + nprev - 1) / 2]
                        tab2nx = tab2nx + ["%d" % (i2)]
                        ax1.axvline(n - 0.5, ls=":", color="gray")
        elif len(tmp.shape) == 3:
            for i0 in range(tmp.shape[0]):
                for i1 in range(tmp.shape[1]):
                    for i2 in range(j1.max() + 1):
                        nprev = n
                        for i2b in range(j2[j1 == i2].max() + 1):
                            idx = np.where((j1 == i2) * (j2 == i2b))[0]
                            dtmp = tmp[i0, i1, idx]
                            if norm:
                                dtmp = dtmp / (ntmp[i0, i1, i2] * ntmp[i0, i1, i2b])
                            if len(idx) == 1:
                                ax1.plot(
                                    np.arange(len(idx)) + n,
                                    dtmp,
                                    ".",
                                    color=color,
                                    lw=lw,
                                )
                            else:
                                if legend and test is None:
                                    ax1.plot(
                                        np.arange(len(idx)) + n,
                                        dtmp,
                                        color=color,
                                        label=lname,
                                        lw=lw,
                                    )
                                    test = 1
                                ax1.plot(
                                    np.arange(len(idx)) + n, dtmp, color=color, lw=lw
                                )
                            tabnx = tabnx + [r"%d,%d" % (j2[k], j3[k]) for k in idx]
                            tabx = tabx + [k + n for k in range(len(idx))]
                            n = n + idx.shape[0]
                        tab2x = tab2x + [(n + nprev - 1) / 2]
                        tab2nx = tab2nx + ["%d" % (i2)]
                        ax1.axvline(n - 0.5, ls=":", color="gray")
        else:
            for i0 in range(tmp.shape[0]):
                for i1 in range(tmp.shape[1]):
                    for i2 in range(j1.max() + 1):
                        nprev = n
                        for i2b in range(j2[j1 == i2].max() + 1):
                            idx = np.where((j1 == i2) * (j2 == i2b))[0]
                            for i3 in range(tmp.shape[3]):
                                dtmp = tmp[i0, i1, idx, i3]
                                if norm:
                                    dtmp = dtmp / (ntmp[i0, i1, i2] * ntmp[i0, i1, i2b])
                                if len(idx) == 1:
                                    ax1.plot(
                                        np.arange(len(idx)) + n,
                                        dtmp,
                                        ".",
                                        color=color,
                                        lw=lw,
                                    )
                                else:
                                    if legend and test is None:
                                        ax1.plot(
                                            np.arange(len(idx)) + n,
                                            dtmp,
                                            color=color,
                                            label=lname,
                                            lw=lw,
                                        )
                                        test = 1
                                    ax1.plot(
                                        np.arange(len(idx)) + n,
                                        dtmp,
                                        color=color,
                                        lw=lw,
                                    )
                            tabnx = tabnx + [r"%d,%d" % (j2[k], j3[k]) for k in idx]
                            tabx = tabx + [k + n for k in range(len(idx))]
                            n = n + idx.shape[0]
                        tab2x = tab2x + [(n + nprev - 1) / 2]
                        tab2nx = tab2nx + ["%d" % (i2)]
                        ax1.axvline(n - 0.5, ls=":", color="gray")
        plt.yscale("log")
        if norm:
            ax1.set_ylabel(r"$\frac{C_{11}}{\sqrt{P_{00,j_1}P_{00,j_2}}}$")
        else:
            ax1.set_ylabel(r"$C_{11}$")

        ax1.set_xticks(tabx)
        ax1.set_xticklabels(tabnx, fontsize=6)
        ax1.set_xlabel(r"$j_{2},j_{3}$", fontsize=6)
        ax1.set_xlim(0, n)

        # Move twinned axis ticks and label from top to bottom
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        # Offset the twin axis below the host
        ax2.spines["bottom"].set_position(("axes", -0.15))

        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)

        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.spines["bottom"].set_visible(True)
        ax2.set_xlim(0, n)
        ax2.set_xticks(tab2x)
        ax2.set_xticklabels(tab2nx, fontsize=6)
        ax2.set_xlabel(r"$j_{1}$", fontsize=6)
        ax1.legend(frameon=0)

    def get_np(self, x):
        if x is not None:
            if isinstance(x, np.ndarray):
                return x
            else:
                return x.numpy()
        else:
            return None

    def save(self, filename):

        outlist = [
            self.get_np(self.S0),
            self.get_np(self.S1),
            self.get_np(self.C10),
            self.get_np(self.C01),
            self.get_np(self.C11),
            self.get_np(self.P00),
        ]

        myout = open("%s.pkl" % (filename), "wb")
        pickle.dump(outlist, myout)
        myout.close()

    def read(self, filename):

        outlist = pickle.load(open("%s.pkl" % (filename), "rb"))

        return scat_cov(
            outlist[0],
            outlist[5],
            outlist[3],
            outlist[4],
            s1=outlist[1],
            c10=outlist[2],
            backend=self.backend,
            use_1D=self.use_1D,
        )

    def std(self):
        if self.S1 is not None:  # Auto
            return np.sqrt(
                (
                    (abs(self.get_np(self.S0)).std()) ** 2
                    + (abs(self.get_np(self.S1)).std()) ** 2
                    + (abs(self.get_np(self.C01)).std()) ** 2
                    + (abs(self.get_np(self.C11)).std()) ** 2
                    + (abs(self.get_np(self.P00)).std()) ** 2
                )
                / 4
            )
        else:  # Cross
            return np.sqrt(
                (
                    (abs(self.get_np(self.S0)).std()) ** 2
                    + (abs(self.get_np(self.C01)).std()) ** 2
                    + (abs(self.get_np(self.C10)).std()) ** 2
                    + (abs(self.get_np(self.C11)).std()) ** 2
                    + (abs(self.get_np(self.P00)).std()) ** 2
                )
                / 4
            )

    def mean(self):
        if self.S1 is not None:  # Auto
            return (
                abs(self.get_np(self.S0)).mean()
                + abs(self.get_np(self.S1)).mean()
                + abs(self.get_np(self.C01)).mean()
                + abs(self.get_np(self.C11)).mean()
                + abs(self.get_np(self.P00)).mean()
            ) / 4
        else:  # Cross
            return (
                abs(self.get_np(self.S0)).mean()
                + abs(self.get_np(self.C01)).mean()
                + abs(self.get_np(self.C10)).mean()
                + abs(self.get_np(self.C11)).mean()
                + abs(self.get_np(self.P00)).mean()
            ) / 4

    def initdx(self, norient):
        idx1 = np.zeros([norient * norient], dtype="int")
        for i in range(norient):
            idx1[i * norient : (i + 1) * norient] = (
                np.arange(norient) + i
            ) % norient + i * norient

        idx2 = np.zeros([norient * norient * norient], dtype="int")
        for i in range(norient):
            for j in range(norient):
                idx2[
                    i * norient * norient
                    + j * norient : i * norient * norient
                    + (j + 1) * norient
                ] = (
                    ((np.arange(norient) + i) % norient) * norient
                    + (np.arange(norient) + i + j) % norient
                    + np.arange(norient) * norient * norient
                )
        self.idx1 = self.backend.constant(idx1)
        self.idx2 = self.backend.constant(idx2)

    def sqrt(self):

        s1 = None
        c10 = None

        if self.S1 is not None:
            s1 = self.backend.bk_sqrt(self.S1)
        if self.C10 is not None:
            c10 = self.backend.bk_sqrt(self.C10)

        s0 = self.backend.bk_sqrt(self.S0)
        p00 = self.backend.bk_sqrt(self.P00)
        c01 = self.backend.bk_sqrt(self.C01)
        c11 = self.backend.bk_sqrt(self.C11)

        return scat_cov(
            s0, p00, c01, c11, s1=s1, c10=c10, backend=self.backend, use_1D=self.use_1D
        )

    def L1(self):

        s1 = None
        c10 = None

        if self.S1 is not None:
            s1 = self.backend.bk_L1(self.S1)
        if self.C10 is not None:
            c10 = self.backend.bk_L1(self.C10)

        s0 = self.backend.bk_L1(self.S0)
        p00 = self.backend.bk_L1(self.P00)
        c01 = self.backend.bk_L1(self.C01)
        c11 = self.backend.bk_L1(self.C11)

        return scat_cov(
            s0, p00, c01, c11, s1=s1, c10=c10, backend=self.backend, use_1D=self.use_1D
        )

    def square_comp(self):

        s1 = None
        c10 = None

        if self.S1 is not None:
            s1 = self.backend.bk_square_comp(self.S1)
        if self.C10 is not None:
            c10 = self.backend.bk_square_comp(self.C10)

        s0 = self.backend.bk_square_comp(self.S0)
        p00 = self.backend.bk_square_comp(self.P00)
        c01 = self.backend.bk_square_comp(self.C01)
        c11 = self.backend.bk_square_comp(self.C11)

        return scat_cov(
            s0, p00, c01, c11, s1=s1, c10=c10, backend=self.backend, use_1D=self.use_1D
        )

    def iso_mean(self, repeat=False):
        shape = list(self.P00.shape)
        norient = shape[3]

        S1 = self.S1
        if self.S1 is not None:
            S1 = self.backend.bk_reduce_mean(self.S1, 3)
            if repeat:
                S1 = self.backend.bk_reshape(
                    self.backend.bk_repeat(S1, norient, 2), self.S1.shape
                )
        P00 = self.backend.bk_reduce_mean(self.P00, 3)
        if repeat:
            P00 = self.backend.bk_reshape(
                self.backend.bk_repeat(P00, norient, 2), self.P00.shape
            )

        C01 = self.C01

        if norient not in self.backend._iso_orient:
            self.backend.calc_iso_orient(norient)

        shape = list(self.C01.shape)
        if self.C01 is not None:
            if self.backend.bk_is_complex(self.C01):
                lmat = self.backend._iso_orient_C[norient]
                lmat_T = self.backend._iso_orient_C_T[norient]
            else:
                lmat = self.backend._iso_orient[norient]
                lmat_T = self.backend._iso_orient_T[norient]

            C01 = self.backend.bk_reshape(
                self.backend.backend.matmul(
                    self.backend.bk_reshape(
                        self.C01, [shape[0] * shape[1] * shape[2], norient * norient]
                    ),
                    lmat,
                ),
                [shape[0], shape[1], shape[2], norient],
            )
            if repeat:
                C01 = self.backend.bk_reshape(
                    self.backend.backend.matmul(
                        self.backend.bk_reshape(
                            C01, [shape[0] * shape[1] * shape[2], norient]
                        ),
                        lmat_T,
                    ),
                    [shape[0], shape[1], shape[2], norient, norient],
                )

        C10 = self.C10
        if self.C10 is not None:
            if self.backend.bk_is_complex(self.C10):
                lmat = self.backend._iso_orient_C[norient]
                lmat_T = self.backend._iso_orient_C_T[norient]
            else:
                lmat = self.backend._iso_orient[norient]
                lmat_T = self.backend._iso_orient_T[norient]

            C10 = self.backend.bk_reshape(
                self.backend.backend.matmul(
                    self.backend.bk_reshape(
                        self.C10, [shape[0] * shape[1] * shape[2], norient * norient]
                    ),
                    lmat,
                ),
                [shape[0], shape[1], shape[2], norient],
            )
            if repeat:
                C10 = self.backend.bk_reshape(
                    self.backend.backend.matmul(
                        self.backend.bk_reshape(
                            C10, [shape[0] * shape[1] * shape[2], norient]
                        ),
                        lmat_T,
                    ),
                    [shape[0], shape[1], shape[2], norient, norient],
                )

        C11 = self.C11
        if self.C11 is not None:
            if self.backend.bk_is_complex(self.C11):
                lmat = self.backend._iso_orient_C[norient]
                lmat_T = self.backend._iso_orient_C_T[norient]
            else:
                lmat = self.backend._iso_orient[norient]
                lmat_T = self.backend._iso_orient_T[norient]

            shape = list(self.C11.shape)
            C11 = self.backend.bk_reshape(
                self.backend.backend.matmul(
                    self.backend.bk_reshape(
                        self.C11,
                        [shape[0] * shape[1] * shape[2] * norient, norient * norient],
                    ),
                    lmat,
                ),
                [shape[0], shape[1], shape[2], norient, norient],
            )
            C11 = self.backend.bk_reduce_mean(C11, 3)
            if repeat:
                C11 = self.backend.bk_reshape(
                    self.backend.bk_repeat(
                        self.backend.bk_reshape(
                            C11, [shape[0] * shape[1] * shape[2], norient]
                        ),
                        norient,
                        axis=0,
                    ),
                    [shape[0] * shape[1] * shape[2] * norient, norient],
                )
                C11 = self.backend.bk_reshape(
                    self.backend.backend.matmul(C11, lmat_T),
                    [shape[0], shape[1], shape[2], norient, norient, norient],
                )

        return scat_cov(
            self.S0,
            P00,
            C01,
            C11,
            s1=S1,
            c10=C10,
            backend=self.backend,
            use_1D=self.use_1D,
        )

    def fft_ang(self, nharm=1, imaginary=False):
        shape = list(self.P00.shape)
        norient = shape[3]

        if (norient, nharm) not in self.backend._fft_1_orient:
            self.backend.calc_fft_orient(norient, nharm, imaginary)

        nout = 1 + nharm
        if imaginary:
            nout = 1 + nharm * 2

        S1 = self.S1
        if self.S1 is not None:
            if self.backend.bk_is_complex(self.S1):
                lmat = self.backend._fft_1_orient_C[(norient, nharm, imaginary)]
            else:
                lmat = self.backend._fft_1_orient[(norient, nharm, imaginary)]
            S1 = self.backend.bk_reshape(
                self.backend.backend.matmul(
                    self.backend.bk_reshape(
                        self.S1, [shape[0] * shape[1] * shape[2], norient]
                    ),
                    lmat,
                ),
                [shape[0], shape[1], shape[2], nout],
            )

        if self.backend.bk_is_complex(self.P00):
            lmat = self.backend._fft_1_orient_C[(norient, nharm, imaginary)]
        else:
            lmat = self.backend._fft_1_orient[(norient, nharm, imaginary)]

        P00 = self.backend.bk_reshape(
            self.backend.backend.matmul(
                self.backend.bk_reshape(
                    self.P00, [shape[0] * shape[1] * shape[2], norient]
                ),
                lmat,
            ),
            [shape[0], shape[1], shape[2], nout],
        )

        C01 = self.C01
        shape = list(self.C01.shape)
        if self.C01 is not None:
            if self.backend.bk_is_complex(self.C01):
                lmat = self.backend._fft_2_orient_C[(norient, nharm, imaginary)]
            else:
                lmat = self.backend._fft_2_orient[(norient, nharm, imaginary)]

            C01 = self.backend.bk_reshape(
                self.backend.backend.matmul(
                    self.backend.bk_reshape(
                        self.C01, [shape[0] * shape[1] * shape[2], norient * norient]
                    ),
                    lmat,
                ),
                [shape[0], shape[1], shape[2], nout, nout],
            )

        C10 = self.C10
        if self.C10 is not None:
            if self.backend.bk_is_complex(self.C10):
                lmat = self.backend._fft_2_orient_C[(norient, nharm, imaginary)]
            else:
                lmat = self.backend._fft_2_orient[(norient, nharm, imaginary)]

            C10 = self.backend.bk_reshape(
                self.backend.backend.matmul(
                    self.backend.bk_reshape(
                        self.C10, [shape[0] * shape[1] * shape[2], norient * norient]
                    ),
                    lmat,
                ),
                [shape[0], shape[1], shape[2], nout, nout],
            )

        C11 = self.C11
        if self.C11 is not None:
            if self.backend.bk_is_complex(self.C01):
                lmat = self.backend._fft_3_orient_C[(norient, nharm, imaginary)]
            else:
                lmat = self.backend._fft_3_orient[(norient, nharm, imaginary)]

            shape = list(self.C11.shape)
            C11 = self.backend.bk_reshape(
                self.backend.backend.matmul(
                    self.backend.bk_reshape(
                        self.C11,
                        [shape[0] * shape[1] * shape[2], norient * norient * norient],
                    ),
                    lmat,
                ),
                [shape[0], shape[1], shape[2], nout, nout, nout],
            )

        return scat_cov(
            self.S0,
            P00,
            C01,
            C11,
            s1=S1,
            c10=C10,
            backend=self.backend,
            use_1D=self.use_1D,
        )

    def iso_std(self, repeat=False):

        val = (self - self.iso_mean(repeat=True)).square_comp()
        return (val.iso_mean(repeat=repeat)).L1()

    def get_nscale(self):
        return self.P00.shape[2]

    def get_norient(self):
        return self.P00.shape[3]

    def add_data_from_log_slope(self, y, n, ds=3):
        if len(y) < ds:
            if len(y) == 1:
                return np.repeat(y[0], n)
            if len(y) == 2:
                a = np.polyfit(np.arange(2), np.log(y[0:2]), 1)
        else:
            a = np.polyfit(np.arange(ds), np.log(y[0:ds]), 1)
        return np.exp((np.arange(n) - 1 - n) * a[0] + a[1])

    def add_data_from_slope(self, y, n, ds=3):
        if len(y) < ds:
            if len(y) == 1:
                return np.repeat(y[0], n)
            if len(y) == 2:
                a = np.polyfit(np.arange(2), y[0:2], 1)
        else:
            a = np.polyfit(np.arange(ds), y[0:ds], 1)
        return (np.arange(n) - 1 - n) * a[0] + a[1]

    def up_grade(self, nscale, ds=3):
        noff = nscale - self.P00.shape[2]
        if noff == 0:
            return scat_cov(
                (self.S0),
                (self.P00),
                (self.C01),
                (self.C11),
                s1=self.S1,
                c10=self.C10,
                backend=self.backend,
                use_1D=self.use_1D,
            )

        inscale = self.P00.shape[2]
        p00 = np.zeros(
            [self.P00.shape[0], self.P00.shape[1], nscale, self.P00.shape[3]],
            dtype="complex",
        )
        if self.BACKEND == "numpy":
            p00[:, :, noff:, :] = self.P00
        else:
            p00[:, :, noff:, :] = self.P00.numpy()
        for i in range(self.P00.shape[0]):
            for j in range(self.P00.shape[1]):
                for k in range(self.P00.shape[3]):
                    p00[i, j, 0:noff, k] = self.add_data_from_log_slope(
                        p00[i, j, noff:, k], noff, ds=ds
                    )

        s1 = np.zeros([self.S1.shape[0], self.S1.shape[1], nscale, self.S1.shape[3]])
        if self.BACKEND == "numpy":
            s1[:, :, noff:, :] = self.S1
        else:
            s1[:, :, noff:, :] = self.S1.numpy()
        for i in range(self.S1.shape[0]):
            for j in range(self.S1.shape[1]):
                for k in range(self.S1.shape[3]):
                    s1[i, j, 0:noff, k] = self.add_data_from_log_slope(
                        s1[i, j, noff:, k], noff, ds=ds
                    )

        nout = 0
        for i in range(1, nscale):
            nout = nout + i

        c01 = np.zeros(
            [
                self.C01.shape[0],
                self.C01.shape[1],
                nout,
                self.C01.shape[3],
                self.C01.shape[4],
            ],
            dtype="complex",
        )

        jo1 = np.zeros([nout])
        jo2 = np.zeros([nout])

        n = 0
        for i in range(1, nscale):
            jo1[n : n + i] = np.arange(i)
            jo2[n : n + i] = i
            n = n + i

        j1 = np.zeros([self.C01.shape[2]])
        j2 = np.zeros([self.C01.shape[2]])

        n = 0
        for i in range(1, self.P00.shape[2]):
            j1[n : n + i] = np.arange(i)
            j2[n : n + i] = i
            n = n + i

        for i in range(self.C01.shape[0]):
            for j in range(self.C01.shape[1]):
                for k in range(self.C01.shape[3]):
                    for l_orient in range(self.C01.shape[4]):
                        for ij in range(noff + 1, nscale):
                            idx = np.where(jo2 == ij)[0]
                            if self.BACKEND == "numpy":
                                c01[i, j, idx[noff:], k, l_orient] = self.C01[
                                    i, j, j2 == ij - noff, k, l_orient
                                ]
                                c01[i, j, idx[:noff], k, l_orient] = (
                                    self.add_data_from_slope(
                                        self.C01[i, j, j2 == ij - noff, k, l_orient],
                                        noff,
                                        ds=ds,
                                    )
                                )
                            else:
                                c01[i, j, idx[noff:], k, l_orient] = self.C01.numpy()[
                                    i, j, j2 == ij - noff, k, l_orient
                                ]
                                c01[i, j, idx[:noff], k, l_orient] = (
                                    self.add_data_from_slope(
                                        self.C01.numpy()[
                                            i, j, j2 == ij - noff, k, l_orient
                                        ],
                                        noff,
                                        ds=ds,
                                    )
                                )

                        for ij in range(nscale):
                            idx = np.where(jo1 == ij)[0]
                            if idx.shape[0] > noff:
                                c01[i, j, idx[:noff], k, l_orient] = (
                                    self.add_data_from_slope(
                                        c01[i, j, idx[noff:], k, l_orient], noff, ds=ds
                                    )
                                )
                            else:
                                c01[i, j, idx, k, l_orient] = np.mean(
                                    c01[i, j, jo1 == ij - 1, k, l_orient]
                                )

        nout = 0
        for j3 in range(nscale):
            for j2 in range(0, j3):
                for j1 in range(0, j2):
                    nout = nout + 1

        c11 = np.zeros(
            [
                self.C11.shape[0],
                self.C11.shape[1],
                nout,
                self.C11.shape[3],
                self.C11.shape[4],
                self.C11.shape[5],
            ],
            dtype="complex",
        )

        jo1 = np.zeros([nout])
        jo2 = np.zeros([nout])
        jo3 = np.zeros([nout])

        nout = 0
        for j3 in range(nscale):
            for j2 in range(0, j3):
                for j1 in range(0, j2):
                    jo1[nout] = j1
                    jo2[nout] = j2
                    jo3[nout] = j3
                    nout = nout + 1

        ncross = self.C11.shape[2]
        jj1 = np.zeros([ncross])
        jj2 = np.zeros([ncross])
        jj3 = np.zeros([ncross])

        n = 0
        for j3 in range(inscale):
            for j2 in range(0, j3):
                for j1 in range(0, j2):
                    jj1[n] = j1
                    jj2[n] = j2
                    jj3[n] = j3
                    n = n + 1

        n = 0
        for j3 in range(nscale):
            for j2 in range(j3):
                idx = np.where((jj3 == j3) * (jj2 == j2))[0]
                if idx.shape[0] > 0:
                    idx2 = np.where((jo3 == j3 + noff) * (jo2 == j2 + noff))[0]
                    for i in range(self.C11.shape[0]):
                        for j in range(self.C11.shape[1]):
                            for k in range(self.C11.shape[3]):
                                for l_orient in range(self.C11.shape[4]):
                                    for m in range(self.C11.shape[5]):
                                        if self.BACKEND == "numpy":
                                            c11[i, j, idx2[noff:], k, l_orient, m] = (
                                                self.C11[i, j, idx, k, l_orient, m]
                                            )
                                            c11[i, j, idx2[:noff], k, l_orient, m] = (
                                                self.add_data_from_log_slope(
                                                    self.C11[i, j, idx, k, l_orient, m],
                                                    noff,
                                                    ds=ds,
                                                )
                                            )
                                        else:
                                            c11[
                                                i, j, idx2[noff:], k, l_orient, m
                                            ] = self.C11.numpy()[
                                                i, j, idx, k, l_orient, m
                                            ]
                                            c11[i, j, idx2[:noff], k, l_orient, m] = (
                                                self.add_data_from_log_slope(
                                                    self.C11.numpy()[
                                                        i, j, idx, k, l_orient, m
                                                    ],
                                                    noff,
                                                    ds=ds,
                                                )
                                            )

        idx = np.where(abs(c11[0, 0, :, 0, 0, 0]) == 0)[0]
        for iii in idx:
            iii1 = np.where(
                (jo1 == jo1[iii] + 1) * (jo2 == jo2[iii] + 1) * (jo3 == jo3[iii] + 1)
            )[0]
            iii2 = np.where(
                (jo1 == jo1[iii] + 2) * (jo2 == jo2[iii] + 2) * (jo3 == jo3[iii] + 2)
            )[0]
            if iii2.shape[0] > 0:
                for i in range(self.C11.shape[0]):
                    for j in range(self.C11.shape[1]):
                        for k in range(self.C11.shape[3]):
                            for l_orient in range(self.C11.shape[4]):
                                for m in range(self.C11.shape[5]):
                                    c11[i, j, iii, k, l_orient, m] = (
                                        self.add_data_from_slope(
                                            c11[i, j, [iii1, iii2], k, l_orient, m],
                                            1,
                                            ds=2,
                                        )[0]
                                    )

        idx = np.where(abs(c11[0, 0, :, 0, 0, 0]) == 0)[0]
        for iii in idx:
            iii1 = np.where(
                (jo1 == jo1[iii]) * (jo2 == jo2[iii]) * (jo3 == jo3[iii] - 1)
            )[0]
            iii2 = np.where(
                (jo1 == jo1[iii]) * (jo2 == jo2[iii]) * (jo3 == jo3[iii] - 2)
            )[0]
            if iii2.shape[0] > 0:
                for i in range(self.C11.shape[0]):
                    for j in range(self.C11.shape[1]):
                        for k in range(self.C11.shape[3]):
                            for l_orient in range(self.C11.shape[4]):
                                for m in range(self.C11.shape[5]):
                                    c11[i, j, iii, k, l_orient, m] = (
                                        self.add_data_from_slope(
                                            c11[i, j, [iii1, iii2], k, l_orient, m],
                                            1,
                                            ds=2,
                                        )[0]
                                    )

        return scat_cov(
            self.S0,
            (p00),
            (c01),
            (c11),
            s1=(s1),
            backend=self.backend,
            use_1D=self.use_1D,
        )


class funct(FOC.FoCUS):

    def fill(self, im, nullval=hp.UNSEEN):
        if self.use_2D:
            return self.fill_2d(im, nullval=nullval)
        if self.use_1D:
            return self.fill_1d(im, nullval=nullval)
        return self.fill_healpy(im, nullval=nullval)

    def moments(self, list_scat):
        if isinstance(list_scat, foscat.scat_cov.scat_cov):
            mS0 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_mean(list_scat.S0, 0), 0
            )
            mP00 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_mean(list_scat.P00, 0), 0
            )
            mC01 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_mean(list_scat.C01, 0), 0
            )
            mC11 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_mean(list_scat.C11, 0), 0
            )
            sS0 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_std(list_scat.S0, 0), 0
            )
            sP00 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_std(list_scat.P00, 0), 0
            )
            sC01 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_std(list_scat.C01, 0), 0
            )
            sC11 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_std(list_scat.C11, 0), 0
            )

            if list_scat.S1 is not None:
                mS1 = self.backend.bk_expand_dims(
                    self.backend.bk_reduce_mean(list_scat.S1, 0), 0
                )
                sS1 = self.backend.bk_expand_dims(
                    self.backend.bk_reduce_std(list_scat.S1, 0), 0
                )
            else:
                mS1 = None
                sS1 = None
            if list_scat.C10 is not None:
                mC10 = self.backend.bk_expand_dims(
                    self.backend.bk_reduce_mean(list_scat.C10, 0), 0
                )
                sC10 = self.backend.bk_expand_dims(
                    self.backend.bk_reduce_std(list_scat.C10, 0), 0
                )
            else:
                mC10 = None
                sC10 = None
        else:
            S0 = None
            for k in list_scat:
                tmp = list_scat[k]
                if self.BACKEND == "numpy":
                    nS0 = np.expand_dims(tmp.S0, 0)
                    nP00 = np.expand_dims(tmp.P00, 0)
                    nC01 = np.expand_dims(tmp.C01, 0)
                    nC11 = np.expand_dims(tmp.C11, 0)
                    if tmp.C10 is not None:
                        nC10 = np.expand_dims(tmp.C10, 0)
                    if tmp.S1 is not None:
                        nS1 = np.expand_dims(tmp.S1, 0)
                else:
                    nS0 = np.expand_dims(tmp.S0.numpy(), 0)
                    nP00 = np.expand_dims(tmp.P00.numpy(), 0)
                    nC01 = np.expand_dims(tmp.C01.numpy(), 0)
                    nC11 = np.expand_dims(tmp.C11.numpy(), 0)
                    if tmp.C10 is not None:
                        nC10 = np.expand_dims(tmp.C10.numpy(), 0)
                    if tmp.S1 is not None:
                        nS1 = np.expand_dims(tmp.S1.numpy(), 0)

                if S0 is None:
                    S0 = nS0
                    P00 = nP00
                    C01 = nC01
                    C11 = nC11
                    if tmp.C10 is not None:
                        C10 = nC10
                    if tmp.S1 is not None:
                        S1 = nS1
                else:
                    S0 = np.concatenate([S0, nS0], 0)
                    P00 = np.concatenate([P00, nP00], 0)
                    C01 = np.concatenate([C01, nC01], 0)
                    C11 = np.concatenate([C11, nC11], 0)
                    if tmp.C10 is not None:
                        C10 = np.concatenate([C10, nC10], 0)
                    if tmp.S1 is not None:
                        S1 = np.concatenate([S1, nS1], 0)
            sS0 = np.std(S0, 0)
            sP00 = np.std(P00, 0)
            sC01 = np.std(C01, 0)
            sC11 = np.std(C11, 0)
            mS0 = np.mean(S0, 0)
            mP00 = np.mean(P00, 0)
            mC01 = np.mean(C01, 0)
            mC11 = np.mean(C11, 0)
            if tmp.C10 is not None:
                sC10 = np.std(C10, 0)
                mC10 = np.mean(C10, 0)
            else:
                sC10 = None
                mC10 = None

            if tmp.S1 is not None:
                sS1 = np.std(S1, 0)
                mS1 = np.mean(S1, 0)
            else:
                sS1 = None
                mS1 = None

        return scat_cov(
            mS0,
            mP00,
            mC01,
            mC11,
            s1=mS1,
            c10=mC10,
            backend=self.backend,
            use_1D=self.use_1D,
        ), scat_cov(
            sS0,
            sP00,
            sC01,
            sC11,
            s1=sS1,
            c10=sC10,
            backend=self.backend,
            use_1D=self.use_1D,
        )

    # compute local direction to make the statistical analysis more efficient
    def stat_cfft(self, im, image2=None, upscale=False, smooth_scale=0):
        tmp = im
        if image2 is not None:
            tmpi2 = image2
        if upscale:
            l_nside = int(np.sqrt(tmp.shape[1] // 12))
            tmp = self.up_grade(tmp, l_nside * 2, axis=1)
            if image2 is not None:
                tmpi2 = self.up_grade(tmpi2, l_nside * 2, axis=1)

        l_nside = int(np.sqrt(tmp.shape[1] // 12))
        nscale = int(np.log(l_nside) / np.log(2))
        cmat = {}
        cmat2 = {}
        for k in range(nscale):
            sim = self.backend.bk_abs(self.convol(tmp, axis=1))
            if image2 is not None:
                sim = self.backend.bk_real(
                    self.backend.bk_L1(
                        self.convol(tmp, axis=1)
                        * self.backend.bk_conjugate(self.convol(tmpi2, axis=1))
                    )
                )
            else:
                sim = self.backend.bk_abs(self.convol(tmp, axis=1))

            cc = self.backend.bk_reduce_mean(sim[:, :, 0] - sim[:, :, 2], 0)
            ss = self.backend.bk_reduce_mean(sim[:, :, 1] - sim[:, :, 3], 0)
            for m in range(smooth_scale):
                if cc.shape[0] > 12:
                    cc = self.ud_grade_2(self.smooth(cc))
                    ss = self.ud_grade_2(self.smooth(ss))
            if cc.shape[0] != tmp.shape[0]:
                ll_nside = int(np.sqrt(tmp.shape[1] // 12))
                cc = self.up_grade(cc, ll_nside)
                ss = self.up_grade(ss, ll_nside)

            if self.BACKEND == "numpy":
                phase = np.fmod(np.arctan2(ss, cc) + 2 * np.pi, 2 * np.pi)
            else:
                phase = np.fmod(
                    np.arctan2(ss.numpy(), cc.numpy()) + 2 * np.pi, 2 * np.pi
                )

            iph = (4 * phase / (2 * np.pi)).astype("int")
            alpha = 4 * phase / (2 * np.pi) - iph
            mat = np.zeros([sim.shape[1], 4 * 4])
            lidx = np.arange(sim.shape[1])
            for l_orient in range(4):
                mat[lidx, 4 * ((l_orient + iph) % 4) + l_orient] = 1.0 - alpha
                mat[lidx, 4 * ((l_orient + iph + 1) % 4) + l_orient] = alpha

            cmat[k] = self.backend.bk_cast(mat.astype("complex64"))

            mat2 = np.zeros([k + 1, sim.shape[1], 4, 4 * 4])

            for k2 in range(k + 1):
                tmp2 = self.backend.bk_repeat(sim, 4, axis=-1)
                sim2 = self.backend.bk_reduce_sum(
                    self.backend.bk_reshape(
                        mat.reshape(1, mat.shape[0], 16) * tmp2,
                        [sim.shape[0], cmat[k].shape[0], 4, 4],
                    ),
                    2,
                )
                sim2 = self.backend.bk_abs(self.convol(sim2, axis=1))

                cc = self.smooth(
                    self.backend.bk_reduce_mean(sim2[:, :, 0] - sim2[:, :, 2], 0)
                )
                ss = self.smooth(
                    self.backend.bk_reduce_mean(sim2[:, :, 1] - sim2[:, :, 3], 0)
                )
                for m in range(smooth_scale):
                    if cc.shape[0] > 12:
                        cc = self.ud_grade_2(self.smooth(cc))
                        ss = self.ud_grade_2(self.smooth(ss))
                if cc.shape[0] != sim.shape[1]:
                    ll_nside = int(np.sqrt(sim.shape[1] // 12))
                    cc = self.up_grade(cc, ll_nside)
                    ss = self.up_grade(ss, ll_nside)

                if self.BACKEND == "numpy":
                    phase = np.fmod(np.arctan2(ss, cc) + 2 * np.pi, 2 * np.pi)
                else:
                    phase = np.fmod(
                        np.arctan2(ss.numpy(), cc.numpy()) + 2 * np.pi, 2 * np.pi
                    )
                """
                for k in range(4):
                    hp.mollview(np.fmod(phase+np.pi,2*np.pi),cmap='jet',nest=True,hold=False,sub=(2,2,1+k))
                plt.show()
                return None
                """
                iph = (4 * phase / (2 * np.pi)).astype("int")
                alpha = 4 * phase / (2 * np.pi) - iph
                lidx = np.arange(sim.shape[1])
                for m in range(4):
                    for l_orient in range(4):
                        mat2[
                            k2, lidx, m, 4 * ((l_orient + iph[:, m]) % 4) + l_orient
                        ] = (1.0 - alpha[:, m])
                        mat2[
                            k2, lidx, m, 4 * ((l_orient + iph[:, m] + 1) % 4) + l_orient
                        ] = alpha[:, m]

            cmat2[k] = self.backend.bk_cast(mat2.astype("complex64"))
            """
            tmp=self.backend.bk_repeat(sim[0],4,axis=1)
            sim2=self.backend.bk_reduce_sum(self.backend.bk_reshape(mat*tmp,[12*nside**2,4,4]),1)

            cc2=(sim2[:,0]-sim2[:,2])
            ss2=(sim2[:,1]-sim2[:,3])
            phase2=np.fmod(np.arctan2(ss2.numpy(),cc2.numpy())+2*np.pi,2*np.pi)

            plt.figure()
            hp.mollview(phase,cmap='jet',nest=True,hold=False,sub=(2,2,1))
            hp.mollview(np.fmod(phase2+np.pi,2*np.pi),cmap='jet',nest=True,hold=False,sub=(2,2,2))
            plt.figure()
            for k in range(4):
                hp.mollview((sim[0,:,k]).numpy().real,cmap='jet',nest=True,hold=False,sub=(2,4,1+k),min=-10,max=10)
                hp.mollview((sim2[:,k]).numpy().real,cmap='jet',nest=True,hold=False,sub=(2,4,5+k),min=-10,max=10)

            plt.show()
            """

            if k < l_nside - 1:
                tmp = self.ud_grade_2(tmp, axis=1)
                if image2 is not None:
                    tmpi2 = self.ud_grade_2(tmpi2, axis=1)
        return cmat, cmat2

    def div_norm(self, complex_value, float_value):
        return self.backend.bk_complex(
            self.backend.bk_real(complex_value) / float_value,
            self.backend.bk_imag(complex_value) / float_value,
        )

    def eval(
        self,
        image1,
        image2=None,
        mask=None,
        norm=None,
        Auto=True,
        calc_var=False,
        cmat=None,
        cmat2=None,
    ):
        """
        Calculates the scattering correlations for a batch of images. Mean are done over pixels.
        mean of modulus:
                        S1 = <|I * Psi_j3|>
             Normalization : take the log
        power spectrum:
                        P00 = <|I * Psi_j3|^2>
            Normalization : take the log
        orig. x modulus:
                        C01 = < (I * Psi)_j3 x (|I * Psi_j2| * Psi_j3)^* >
             Normalization : divide by (P00_j2 * P00_j3)^0.5
        modulus x modulus:
                        C11 = <(|I * psi1| * psi3)(|I * psi2| * psi3)^*>
             Normalization : divide by (P00_j1 * P00_j2)^0.5
        Parameters
        ----------
        image1: tensor
            Image on which we compute the scattering coefficients [Nbatch, Npix, 1, 1]
        image2: tensor
            Second image. If not None, we compute cross-scattering covariance coefficients.
        mask:
        norm: None or str
            If None no normalization is applied, if 'auto' normalize by the reference P00,
            if 'self' normalize by the current P00.
        all_cross: False or True
            If False compute all the coefficient even the Imaginary part,
            If True return only the terms computable in the auto case.
        Returns
        -------
        S1, P00, C01, C11 normalized
        """
        return_data = self.return_data
        # Check input consistency
        if image2 is not None:
            if list(image1.shape) != list(image2.shape):
                print(
                    "The two input image should have the same size to eval Scattering Covariance"
                )
                return None
        if mask is not None:
            if list(image1.shape) != list(mask.shape)[1:]:
                print(
                    "The LAST COLUMN of the mask should have the same size ",
                    mask.shape,
                    "than the input image ",
                    image1.shape,
                    "to eval Scattering Covariance",
                )
                return None
        if self.use_2D and len(image1.shape) < 2:
            print(
                "To work with 2D scattering transform, two dimension is needed, input map has only on dimension"
            )
            return None

        ### AUTO OR CROSS
        cross = False
        if image2 is not None:
            cross = True
            all_cross = Auto
        else:
            all_cross = False

        ### PARAMETERS
        axis = 1
        # determine jmax and nside corresponding to the input map
        im_shape = image1.shape
        if self.use_2D:
            if len(image1.shape) == 2:
                nside = np.min([im_shape[0], im_shape[1]])
                npix = im_shape[0] * im_shape[1]  # Number of pixels
                x1 = im_shape[0]
                x2 = im_shape[1]
            else:
                nside = np.min([im_shape[1], im_shape[2]])
                npix = im_shape[1] * im_shape[2]  # Number of pixels
                x1 = im_shape[1]
                x2 = im_shape[2]
            J = int(np.log(nside - self.KERNELSZ) / np.log(2))  # Number of j scales
        elif self.use_1D:
            if len(image1.shape) == 2:
                npix = int(im_shape[1])  # Number of pixels
            else:
                npix = int(im_shape[0])  # Number of pixels

            nside = int(npix)

            J = int(np.log(nside) / np.log(2))  # Number of j scales
        else:
            if len(image1.shape) == 2:
                npix = int(im_shape[1])  # Number of pixels
            else:
                npix = int(im_shape[0])  # Number of pixels

            nside = int(np.sqrt(npix // 12))

            J = int(np.log(nside) / np.log(2))  # Number of j scales

        Jmax = J - self.OSTEP  # Number of steps for the loop on scales

        ### LOCAL VARIABLES (IMAGES and MASK)
        if len(image1.shape) == 1 or (len(image1.shape) == 2 and self.use_2D):
            I1 = self.backend.bk_cast(
                self.backend.bk_expand_dims(image1, 0)
            )  # Local image1 [Nbatch, Npix]
            if cross:
                I2 = self.backend.bk_cast(
                    self.backend.bk_expand_dims(image2, 0)
                )  # Local image2 [Nbatch, Npix]
        else:
            I1 = self.backend.bk_cast(image1)  # Local image1 [Nbatch, Npix]
            if cross:
                I2 = self.backend.bk_cast(image2)  # Local image2 [Nbatch, Npix]

        if mask is None:
            if self.use_2D:
                vmask = self.backend.bk_ones([1, x1, x2], dtype=self.all_type)
            else:
                vmask = self.backend.bk_ones([1, npix], dtype=self.all_type)
        else:
            vmask = self.backend.bk_cast(mask)  # [Nmask, Npix]

        if self.KERNELSZ > 3:
            # if the kernel size is bigger than 3 increase the binning before smoothing
            if self.use_2D:
                vmask = self.up_grade(
                    vmask, I1.shape[axis] * 2, axis=1, nouty=I1.shape[axis + 1] * 2
                )
                I1 = self.up_grade(
                    I1, I1.shape[axis] * 2, axis=axis, nouty=I1.shape[axis + 1] * 2
                )
                if cross:
                    I2 = self.up_grade(
                        I2, I2.shape[axis] * 2, axis=axis, nouty=I2.shape[axis + 1] * 2
                    )
            elif self.use_1D:
                vmask = self.up_grade(vmask, I1.shape[axis] * 2, axis=1)
                I1 = self.up_grade(I1, I1.shape[axis] * 2, axis=axis)
                if cross:
                    I2 = self.up_grade(I2, I2.shape[axis] * 2, axis=axis)
            else:
                I1 = self.up_grade(I1, nside * 2, axis=axis)
                vmask = self.up_grade(vmask, nside * 2, axis=1)
                if cross:
                    I2 = self.up_grade(I2, nside * 2, axis=axis)

            if self.KERNELSZ > 5:
                # if the kernel size is bigger than 3 increase the binning before smoothing
                if self.use_2D:
                    vmask = self.up_grade(
                        vmask, I1.shape[axis] * 2, axis=1, nouty=I1.shape[axis + 1] * 2
                    )
                    I1 = self.up_grade(
                        I1, I1.shape[axis] * 2, axis=axis, nouty=I1.shape[axis + 1] * 2
                    )
                    if cross:
                        I2 = self.up_grade(
                            I2,
                            I2.shape[axis] * 2,
                            axis=axis,
                            nouty=I2.shape[axis + 1] * 2,
                        )
                elif self.use_1D:
                    vmask = self.up_grade(vmask, I1.shape[axis] * 4, axis=1)
                    I1 = self.up_grade(I1, I1.shape[axis] * 4, axis=axis)
                    if cross:
                        I2 = self.up_grade(I2, I2.shape[axis] * 4, axis=axis)
                else:
                    I1 = self.up_grade(I1, nside * 4, axis=axis)
                    vmask = self.up_grade(vmask, nside * 4, axis=1)
                    if cross:
                        I2 = self.up_grade(I2, nside * 4, axis=axis)

        # Normalize the masks because they have different pixel numbers
        # vmask /= self.backend.bk_reduce_sum(vmask, axis=1)[:, None]  # [Nmask, Npix]

        ### INITIALIZATION
        # Coefficients
        S1, P00, C01, C11, C10 = None, None, None, None, None

        off_P0 = -2
        off_C01 = -3
        off_C11 = -4
        if self.use_1D:
            off_P0 = -1
            off_C01 = -1
            off_C11 = -1

        # Dictionaries for C01 computation
        M1_dic = {}  # M stands for Module M1 = |I1 * Psi|
        if cross:
            M2_dic = {}

        # P00 for normalization
        cond_init_P1_dic = (norm == "self") or (
            (norm == "auto") and (self.P1_dic is None)
        )
        if norm is None:
            pass
        elif cond_init_P1_dic:
            P1_dic = {}
            if cross:
                P2_dic = {}
        elif (norm == "auto") and (self.P1_dic is not None):
            P1_dic = self.P1_dic
            if cross:
                P2_dic = self.P2_dic

        if return_data:
            s0 = I1
        else:
            if not cross:
                s0, l_vs0 = self.masked_mean(I1, vmask, axis=1, calc_var=True)
            else:
                s0, l_vs0 = self.masked_mean(
                    self.backend.bk_L1(I1 * I2), vmask, axis=1, calc_var=True
                )
            vs0 = self.backend.bk_concat([l_vs0, l_vs0], 1)
            s0 = self.backend.bk_concat([s0, l_vs0], 1)

        #### COMPUTE S1, P00, C01 and C11
        nside_j3 = nside  # NSIDE start (nside_j3 = nside / 2^j3)
        for j3 in range(Jmax):
            if return_data:
                if C01 is None:
                    C01 = {}
                C01[j3] = None

                if C10 is None:
                    C10 = {}
                C10[j3] = None

                if C11 is None:
                    C11 = {}
                C11[j3] = None

            ####### S1 and P00
            ### Make the convolution I1 * Psi_j3
            conv1 = self.convol(I1, axis=1)  # [Nbatch, Npix_j3, Norient3]

            if cmat is not None:
                tmp2 = self.backend.bk_repeat(conv1, 4, axis=-1)
                conv1 = self.backend.bk_reduce_sum(
                    self.backend.bk_reshape(
                        cmat[j3] * tmp2, [1, cmat[j3].shape[0], 4, 4]
                    ),
                    2,
                )

            ### Take the module M1 = |I1 * Psi_j3|
            M1_square = conv1 * self.backend.bk_conjugate(
                conv1
            )  # [Nbatch, Npix_j3, Norient3]
            M1 = self.backend.bk_L1(M1_square)  # [Nbatch, Npix_j3, Norient3]
            # Store M1_j3 in a dictionary
            M1_dic[j3] = M1

            if not cross:  # Auto
                M1_square = self.backend.bk_real(M1_square)

                ### P00_auto = < M1^2 >_pix
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                if return_data:
                    p00 = M1_square
                else:
                    if calc_var:
                        p00, vp00 = self.masked_mean(
                            M1_square, vmask, axis=1, rank=j3, calc_var=True
                        )
                    else:
                        p00 = self.masked_mean(M1_square, vmask, axis=1, rank=j3)

                if cond_init_P1_dic:
                    # We fill P1_dic with P00 for normalisation of C01 and C11
                    P1_dic[j3] = self.backend.bk_real(p00)  # [Nbatch, Nmask, Norient3]

                # We store P00_auto to return it [Nbatch, Nmask, NP00, Norient3]
                if return_data:
                    if P00 is None:
                        P00 = {}
                    P00[j3] = p00
                else:
                    if norm == "auto":  # Normalize P00
                        p00 /= P1_dic[j3]
                    if P00 is None:
                        P00 = self.backend.bk_expand_dims(
                            p00, off_P0
                        )  # Add a dimension for NP00
                        if calc_var:
                            VP00 = self.backend.bk_expand_dims(
                                vp00, off_P0
                            )  # Add a dimension for NP00
                    else:
                        P00 = self.backend.bk_concat(
                            [P00, self.backend.bk_expand_dims(p00, off_P0)], axis=2
                        )
                        if calc_var:
                            VP00 = self.backend.bk_concat(
                                [VP00, self.backend.bk_expand_dims(vp00, off_P0)],
                                axis=2,
                            )

                #### S1_auto computation
                ### Image 1 : S1 = < M1 >_pix
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                if return_data:
                    s1 = M1
                else:
                    if calc_var:
                        s1, vs1 = self.masked_mean(
                            M1, vmask, axis=1, rank=j3, calc_var=True
                        )  # [Nbatch, Nmask, Norient3]
                    else:
                        s1 = self.masked_mean(
                            M1, vmask, axis=1, rank=j3
                        )  # [Nbatch, Nmask, Norient3]

                if return_data:
                    if S1 is None:
                        S1 = {}
                    S1[j3] = s1
                else:
                    ### Normalize S1
                    if norm is not None:
                        self.div_norm(s1, (P1_dic[j3]) ** 0.5)
                    ### We store S1 for image1  [Nbatch, Nmask, NS1, Norient3]
                    if S1 is None:
                        S1 = self.backend.bk_expand_dims(
                            s1, off_P0
                        )  # Add a dimension for NS1
                        if calc_var:
                            VS1 = self.backend.bk_expand_dims(
                                vs1, off_P0
                            )  # Add a dimension for NS1
                    else:
                        S1 = self.backend.bk_concat(
                            [S1, self.backend.bk_expand_dims(s1, off_P0)], axis=2
                        )
                        if calc_var:
                            VS1 = self.backend.bk_concat(
                                [VS1, self.backend.bk_expand_dims(vs1, off_P0)], axis=2
                            )

            else:  # Cross
                ### Make the convolution I2 * Psi_j3
                conv2 = self.convol(I2, axis=1)  # [Nbatch, Npix_j3, Norient3]
                if cmat is not None:
                    tmp2 = self.backend.bk_repeat(conv2, 4, axis=-1)
                    conv2 = self.backend.bk_reduce_sum(
                        self.backend.bk_reshape(
                            cmat[j3] * tmp2, [1, cmat[j3].shape[0], 4, 4]
                        ),
                        2,
                    )
                ### Take the module M2 = |I2 * Psi_j3|
                M2_square = conv2 * self.backend.bk_conjugate(
                    conv2
                )  # [Nbatch, Npix_j3, Norient3]
                M2 = self.backend.bk_L1(M2_square)  # [Nbatch, Npix_j3, Norient3]
                # Store M2_j3 in a dictionary
                M2_dic[j3] = M2

                ### P00_auto = < M2^2 >_pix
                # Not returned, only for normalization
                if cond_init_P1_dic:
                    # Apply the mask [Nmask, Npix_j3] and average over pixels
                    if return_data:
                        p1 = M1_square
                        p2 = M2_square
                    else:
                        if calc_var:
                            p1, vp1 = self.masked_mean(
                                M1_square, vmask, axis=1, rank=j3, calc_var=True
                            )  # [Nbatch, Nmask, Norient3]
                            p2, vp2 = self.masked_mean(
                                M2_square, vmask, axis=1, rank=j3, calc_var=True
                            )  # [Nbatch, Nmask, Norient3]
                        else:
                            p1 = self.masked_mean(
                                M1_square, vmask, axis=1, rank=j3
                            )  # [Nbatch, Nmask, Norient3]
                            p2 = self.masked_mean(
                                M2_square, vmask, axis=1, rank=j3
                            )  # [Nbatch, Nmask, Norient3]
                    # We fill P1_dic with P00 for normalisation of C01 and C11
                    P1_dic[j3] = self.backend.bk_real(p1)  # [Nbatch, Nmask, Norient3]
                    P2_dic[j3] = self.backend.bk_real(p2)  # [Nbatch, Nmask, Norient3]

                ### P00_cross = < (I1 * Psi_j3) (I2 * Psi_j3)^* >_pix
                # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
                p00 = conv1 * self.backend.bk_conjugate(conv2)
                MX = self.backend.bk_L1(p00)
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                if return_data:
                    p00 = p00
                else:
                    if calc_var:
                        p00, vp00 = self.masked_mean(
                            p00, vmask, axis=1, rank=j3, calc_var=True
                        )
                    else:
                        p00 = self.masked_mean(p00, vmask, axis=1, rank=j3)

                if return_data:
                    if P00 is None:
                        P00 = {}
                    P00[j3] = p00
                else:
                    ### Normalize P00_cross
                    if norm == "auto":
                        p00 /= (P1_dic[j3] * P2_dic[j3]) ** 0.5

                    ### Store P00_cross as complex [Nbatch, Nmask, NP00, Norient3]
                    if not all_cross:
                        p00 = self.backend.bk_real(p00)

                    if P00 is None:
                        P00 = self.backend.bk_expand_dims(
                            p00, off_P0
                        )  # Add a dimension for NP00
                        if calc_var:
                            VP00 = self.backend.bk_expand_dims(
                                vp00, off_P0
                            )  # Add a dimension for NP00
                    else:
                        P00 = self.backend.bk_concat(
                            [P00, self.backend.bk_expand_dims(p00, off_P0)], axis=2
                        )
                        if calc_var:
                            VP00 = self.backend.bk_concat(
                                [VP00, self.backend.bk_expand_dims(vp00, off_P0)],
                                axis=2,
                            )

                #### S1_auto computation
                ### Image 1 : S1 = < M1 >_pix
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                if return_data:
                    s1 = MX
                else:
                    if calc_var:
                        s1, vs1 = self.masked_mean(
                            MX, vmask, axis=1, rank=j3, calc_var=True
                        )  # [Nbatch, Nmask, Norient3]
                    else:
                        s1 = self.masked_mean(
                            MX, vmask, axis=1, rank=j3
                        )  # [Nbatch, Nmask, Norient3]
                if return_data:
                    if S1 is None:
                        S1 = {}
                    S1[j3] = s1
                else:
                    ### Normalize S1
                    if norm is not None:
                        self.div_norm(s1, (P1_dic[j3]) ** 0.5)
                    ### We store S1 for image1  [Nbatch, Nmask, NS1, Norient3]
                    if S1 is None:
                        S1 = self.backend.bk_expand_dims(
                            s1, off_P0
                        )  # Add a dimension for NS1
                        if calc_var:
                            VS1 = self.backend.bk_expand_dims(
                                vs1, off_P0
                            )  # Add a dimension for NS1
                    else:
                        S1 = self.backend.bk_concat(
                            [S1, self.backend.bk_expand_dims(s1, off_P0)], axis=2
                        )
                        if calc_var:
                            VS1 = self.backend.bk_concat(
                                [VS1, self.backend.bk_expand_dims(vs1, off_P0)], axis=2
                            )

            # Initialize dictionaries for |I1*Psi_j| * Psi_j3
            M1convPsi_dic = {}
            if cross:
                # Initialize dictionaries for |I2*Psi_j| * Psi_j3
                M2convPsi_dic = {}

            ###### C01
            for j2 in range(0, j3 + 1):  # j2 <= j3
                if return_data:
                    if C11[j3] is None:
                        C11[j3] = {}
                    C11[j3][j2] = None

                ### C01_auto = < (I1 * Psi)_j3 x (|I1 * Psi_j2| * Psi_j3)^* >_pix
                if not cross:
                    if calc_var:
                        c01, vc01 = self._compute_C01(
                            j2,
                            j3,
                            conv1,
                            vmask,
                            M1_dic,
                            M1convPsi_dic,
                            calc_var=True,
                            cmat2=cmat2,
                        )  # [Nbatch, Nmask, Norient3, Norient2]
                    else:
                        c01 = self._compute_C01(
                            j2,
                            j3,
                            conv1,
                            vmask,
                            M1_dic,
                            M1convPsi_dic,
                            return_data=return_data,
                            cmat2=cmat2,
                        )  # [Nbatch, Nmask, Norient3, Norient2]

                    if return_data:
                        if C01[j3] is None:
                            C01[j3] = {}
                        C01[j3][j2] = c01
                    else:
                        ### Normalize C01 with P00_j [Nbatch, Nmask, Norient_j]
                        if norm is not None:
                            self.div_norm(
                                c01,
                                (
                                    self.backend.bk_expand_dims(P1_dic[j2], off_P0)
                                    * self.backend.bk_expand_dims(P1_dic[j3], -1)
                                )
                                ** 0.5,
                            )  # [Nbatch, Nmask, Norient3, Norient2]

                        ### Store C01 as a complex [Nbatch, Nmask, NC01, Norient3, Norient2]
                        if C01 is None:
                            C01 = self.backend.bk_expand_dims(
                                c01, off_C01
                            )  # Add a dimension for NC01
                            if calc_var:
                                VC01 = self.backend.bk_expand_dims(
                                    vc01, off_C01
                                )  # Add a dimension for NC01
                        else:
                            C01 = self.backend.bk_concat(
                                [C01, self.backend.bk_expand_dims(c01, off_C01)], axis=2
                            )  # Add a dimension for NC01
                            if calc_var:
                                VC01 = self.backend.bk_concat(
                                    [VC01, self.backend.bk_expand_dims(vc01, off_C01)],
                                    axis=2,
                                )  # Add a dimension for NC01

                ### C01_cross = < (I1 * Psi)_j3 x (|I2 * Psi_j2| * Psi_j3)^* >_pix
                ### C10_cross = < (I2 * Psi)_j3 x (|I1 * Psi_j2| * Psi_j3)^* >_pix
                else:
                    if calc_var:
                        c01, vc01 = self._compute_C01(
                            j2,
                            j3,
                            conv1,
                            vmask,
                            M2_dic,
                            M2convPsi_dic,
                            calc_var=True,
                            cmat2=cmat2,
                        )
                        c10, vc10 = self._compute_C01(
                            j2,
                            j3,
                            conv2,
                            vmask,
                            M1_dic,
                            M1convPsi_dic,
                            calc_var=True,
                            cmat2=cmat2,
                        )
                    else:
                        c01 = self._compute_C01(
                            j2,
                            j3,
                            conv1,
                            vmask,
                            M2_dic,
                            M2convPsi_dic,
                            return_data=return_data,
                            cmat2=cmat2,
                        )
                        c10 = self._compute_C01(
                            j2,
                            j3,
                            conv2,
                            vmask,
                            M1_dic,
                            M1convPsi_dic,
                            return_data=return_data,
                            cmat2=cmat2,
                        )

                    if return_data:
                        if C01[j3] is None:
                            C01[j3] = {}
                            C10[j3] = {}
                        C01[j3][j2] = c01
                        C10[j3][j2] = c10
                    else:
                        ### Normalize C01 and C10 with P00_j [Nbatch, Nmask, Norient_j]
                        if norm is not None:
                            self.div_norm(
                                c01,
                                (
                                    self.backend.bk_expand_dims(P2_dic[j2], off_P0)
                                    * self.backend.bk_expand_dims(P1_dic[j3], -1)
                                )
                                ** 0.5,
                            )  # [Nbatch, Nmask, Norient3, Norient2]
                            self.div_norm(
                                c10,
                                (
                                    self.backend.bk_expand_dims(P1_dic[j2], off_P0)
                                    * self.backend.bk_expand_dims(P2_dic[j3], -1)
                                )
                                ** 0.5,
                            )  # [Nbatch, Nmask, Norient3, Norient2]

                        ### Store C01 and C10 as a complex [Nbatch, Nmask, NC01, Norient3, Norient2]
                        if C01 is None:
                            C01 = self.backend.bk_expand_dims(
                                c01, off_C01
                            )  # Add a dimension for NC01
                            if calc_var:
                                VC01 = self.backend.bk_expand_dims(
                                    vc01, off_C01
                                )  # Add a dimension for NC01
                        else:
                            C01 = self.backend.bk_concat(
                                [C01, self.backend.bk_expand_dims(c01, off_C01)], axis=2
                            )  # Add a dimension for NC01
                            if calc_var:
                                VC01 = self.backend.bk_concat(
                                    [VC01, self.backend.bk_expand_dims(vc01, off_C01)],
                                    axis=2,
                                )  # Add a dimension for NC01
                        if C10 is None:
                            C10 = self.backend.bk_expand_dims(
                                c10, off_C01
                            )  # Add a dimension for NC01
                            if calc_var:
                                VC10 = self.backend.bk_expand_dims(
                                    vc10, off_C01
                                )  # Add a dimension for NC01
                        else:
                            C10 = self.backend.bk_concat(
                                [C10, self.backend.bk_expand_dims(c10, off_C01)], axis=2
                            )  # Add a dimension for NC01
                            if calc_var:
                                VC10 = self.backend.bk_concat(
                                    [VC10, self.backend.bk_expand_dims(vc10, off_C01)],
                                    axis=2,
                                )  # Add a dimension for NC01

                ##### C11
                for j1 in range(0, j2 + 1):  # j1 <= j2
                    ### C11_auto = <(|I1 * psi1| * psi3)(|I1 * psi2| * psi3)^*>
                    if not cross:
                        if calc_var:
                            c11, vc11 = self._compute_C11(
                                j1,
                                j2,
                                vmask,
                                M1convPsi_dic,
                                M2convPsi_dic=None,
                                calc_var=True,
                            )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        else:
                            c11 = self._compute_C11(
                                j1,
                                j2,
                                vmask,
                                M1convPsi_dic,
                                M2convPsi_dic=None,
                                return_data=return_data,
                            )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]

                        if return_data:
                            if C11[j3][j2] is None:
                                C11[j3][j2] = {}
                            C11[j3][j2][j1] = c11
                        else:
                            ### Normalize C11 with P00_j [Nbatch, Nmask, Norient_j]
                            if norm is not None:
                                self.div_norm(
                                    c11,
                                    (
                                        self.backend.bk_expand_dims(
                                            self.backend.bk_expand_dims(
                                                P1_dic[j1], off_P0
                                            ),
                                            off_P0,
                                        )
                                        * self.backend.bk_expand_dims(
                                            self.backend.bk_expand_dims(
                                                P1_dic[j2], off_P0
                                            ),
                                            -1,
                                        )
                                    )
                                    ** 0.5,
                                )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                            ### Store C11 as a complex [Nbatch, Nmask, NC11, Norient3, Norient2, Norient1]
                            if C11 is None:
                                C11 = self.backend.bk_expand_dims(
                                    c11, off_C11
                                )  # Add a dimension for NC11
                                if calc_var:
                                    VC11 = self.backend.bk_expand_dims(
                                        vc11, off_C11
                                    )  # Add a dimension for NC11
                            else:
                                C11 = self.backend.bk_concat(
                                    [C11, self.backend.bk_expand_dims(c11, off_C11)],
                                    axis=2,
                                )  # Add a dimension for NC11
                                if calc_var:
                                    VC11 = self.backend.bk_concat(
                                        [
                                            VC11,
                                            self.backend.bk_expand_dims(vc11, off_C11),
                                        ],
                                        axis=2,
                                    )  # Add a dimension for NC11

                        ### C11_cross = <(|I1 * psi1| * psi3)(|I2 * psi2| * psi3)^*>
                    else:
                        if calc_var:
                            c11, vc11 = self._compute_C11(
                                j1,
                                j2,
                                vmask,
                                M1convPsi_dic,
                                M2convPsi_dic=M2convPsi_dic,
                                calc_var=True,
                            )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        else:
                            c11 = self._compute_C11(
                                j1,
                                j2,
                                vmask,
                                M1convPsi_dic,
                                M2convPsi_dic=M2convPsi_dic,
                                return_data=return_data,
                            )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]

                        if return_data:
                            if C11[j3][j2] is None:
                                C11[j3][j2] = {}
                            C11[j3][j2][j1] = c11
                        else:
                            ### Normalize C11 with P00_j [Nbatch, Nmask, Norient_j]
                            if norm is not None:
                                self.div_norm(
                                    c11,
                                    (
                                        self.backend.bk_expand_dims(
                                            self.backend.bk_expand_dims(
                                                P1_dic[j1], off_P0
                                            ),
                                            off_P0,
                                        )
                                        * self.backend.bk_expand_dims(
                                            self.backend.bk_expand_dims(
                                                P2_dic[j2], off_P0
                                            ),
                                            -1,
                                        )
                                    )
                                    ** 0.5,
                                )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                            ### Store C11 as a complex [Nbatch, Nmask, NC11, Norient3, Norient2, Norient1]
                            if C11 is None:
                                C11 = self.backend.bk_expand_dims(
                                    c11, off_C11
                                )  # Add a dimension for NC11
                                if calc_var:
                                    VC11 = self.backend.bk_expand_dims(
                                        vc11, off_C11
                                    )  # Add a dimension for NC11
                            else:
                                C11 = self.backend.bk_concat(
                                    [C11, self.backend.bk_expand_dims(c11, off_C11)],
                                    axis=2,
                                )  # Add a dimension for NC11
                                if calc_var:
                                    VC11 = self.backend.bk_concat(
                                        [
                                            VC11,
                                            self.backend.bk_expand_dims(vc11, off_C11),
                                        ],
                                        axis=2,
                                    )  # Add a dimension for NC11

            ###### Reshape for next iteration on j3
            ### Image I1,
            # downscale the I1 [Nbatch, Npix_j3]
            if j3 != Jmax - 1:
                I1_smooth = self.smooth(I1, axis=1)
                I1 = self.ud_grade_2(I1_smooth, axis=1)

                ### Image I2
                if cross:
                    I2_smooth = self.smooth(I2, axis=1)
                    I2 = self.ud_grade_2(I2_smooth, axis=1)

                ### Modules
                for j2 in range(0, j3 + 1):  # j2 =< j3
                    ### Dictionary M1_dic[j2]
                    M1_smooth = self.smooth(
                        M1_dic[j2], axis=1
                    )  # [Nbatch, Npix_j3, Norient3]
                    M1_dic[j2] = self.ud_grade_2(
                        M1_smooth, axis=1
                    )  # [Nbatch, Npix_j3, Norient3]

                    ### Dictionary M2_dic[j2]
                    if cross:
                        M2_smooth = self.smooth(
                            M2_dic[j2], axis=1
                        )  # [Nbatch, Npix_j3, Norient3]
                        M2_dic[j2] = self.ud_grade_2(
                            M2_smooth, axis=1
                        )  # [Nbatch, Npix_j3, Norient3]
                ### Mask
                vmask = self.ud_grade_2(vmask, axis=1)

                if self.mask_thres is not None:
                    vmask = self.backend.bk_threshold(vmask, self.mask_thres)

                ### NSIDE_j3
                nside_j3 = nside_j3 // 2

        ### Store P1_dic and P2_dic in self
        if (norm == "auto") and (self.P1_dic is None):
            self.P1_dic = P1_dic
            if cross:
                self.P2_dic = P2_dic

        if calc_var:
            if not cross:
                return scat_cov(
                    s0, P00, C01, C11, s1=S1, backend=self.backend, use_1D=self.use_1D
                ), scat_cov(
                    vs0,
                    VP00,
                    VC01,
                    VC11,
                    s1=VS1,
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
            else:
                return scat_cov(
                    s0,
                    P00,
                    C01,
                    C11,
                    s1=S1,
                    c10=C10,
                    backend=self.backend,
                    use_1D=self.use_1D,
                ), scat_cov(
                    vs0,
                    VP00,
                    VC01,
                    VC11,
                    s1=VS1,
                    c10=VC10,
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
        else:
            if not cross:
                return scat_cov(
                    s0, P00, C01, C11, s1=S1, backend=self.backend, use_1D=self.use_1D
                )
            else:
                return scat_cov(
                    s0,
                    P00,
                    C01,
                    C11,
                    s1=S1,
                    c10=C10,
                    backend=self.backend,
                    use_1D=self.use_1D,
                )

    def clean_norm(self):
        self.P1_dic = None
        self.P2_dic = None
        return

    def _compute_C01(
        self,
        j2,
        j3,
        conv,
        vmask,
        M_dic,
        MconvPsi_dic,
        calc_var=False,
        return_data=False,
        cmat2=None,
    ):
        """
        Compute the C01 coefficients (auto or cross)
        C01 = < (Ia * Psi)_j3 x (|Ib * Psi_j2| * Psi_j3)^* >_pix
        Parameters
        ----------
        Returns
        -------
        cc01, sc01: real and imag parts of C01 coeff
        """
        ### Compute |I1 * Psi_j2| * Psi_j3 = M1_j2 * Psi_j3
        # Warning: M1_dic[j2] is already at j3 resolution [Nbatch, Npix_j3, Norient3]
        MconvPsi = self.convol(
            M_dic[j2], axis=1
        )  # [Nbatch, Npix_j3, Norient3, Norient2]
        if cmat2 is not None:
            tmp2 = self.backend.bk_repeat(MconvPsi, 4, axis=-1)
            MconvPsi = self.backend.bk_reduce_sum(
                self.backend.bk_reshape(
                    cmat2[j3][j2] * tmp2, [1, cmat2[j3].shape[1], 4, 4, 4]
                ),
                3,
            )

        # Store it so we can use it in C11 computation
        MconvPsi_dic[j2] = MconvPsi  # [Nbatch, Npix_j3, Norient3, Norient2]

        ### Compute the product (I2 * Psi)_j3 x (M1_j2 * Psi_j3)^*
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        # cconv, sconv are [Nbatch, Npix_j3, Norient3]
        if self.use_1D:
            c01 = conv * self.backend.bk_conjugate(MconvPsi)
        else:
            c01 = self.backend.bk_expand_dims(conv, -1) * self.backend.bk_conjugate(
                MconvPsi
            )  # [Nbatch, Npix_j3, Norient3, Norient2]

        ### Apply the mask [Nmask, Npix_j3] and sum over pixels
        if return_data:
            return c01
        else:
            if calc_var:
                c01, vc01 = self.masked_mean(
                    c01, vmask, axis=1, rank=j2, calc_var=True
                )  # [Nbatch, Nmask, Norient3, Norient2]
                return c01, vc01
            else:
                c01 = self.masked_mean(
                    c01, vmask, axis=1, rank=j2
                )  # [Nbatch, Nmask, Norient3, Norient2]
            return c01

    def _compute_C11(
        self,
        j1,
        j2,
        vmask,
        M1convPsi_dic,
        M2convPsi_dic=None,
        calc_var=False,
        return_data=False,
    ):
        #### Simplify notations
        M1 = M1convPsi_dic[j1]  # [Nbatch, Npix_j3, Norient3, Norient1]

        # Auto or Cross coefficients
        if M2convPsi_dic is None:  # Auto
            M2 = M1convPsi_dic[j2]  # [Nbatch, Npix_j3, Norient3, Norient2]
        else:  # Cross
            M2 = M2convPsi_dic[j2]

        ### Compute the product (|I1 * Psi_j1| * Psi_j3)(|I2 * Psi_j2| * Psi_j3)
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        if self.use_1D:
            c11 = M1 * self.backend.bk_conjugate(M2)
        else:
            c11 = self.backend.bk_expand_dims(M1, -2) * self.backend.bk_conjugate(
                self.backend.bk_expand_dims(M2, -1)
            )  # [Nbatch, Npix_j3, Norient3, Norient2, Norient1]

        ### Apply the mask and sum over pixels
        if return_data:
            return c11
        else:
            if calc_var:
                c11, vc11 = self.masked_mean(
                    c11, vmask, axis=1, rank=j2, calc_var=True
                )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                return c11, vc11
            else:
                c11 = self.masked_mean(
                    c11, vmask, axis=1, rank=j2
                )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                return c11

    def square(self, x):
        if isinstance(x, scat_cov):
            if x.S1 is None:
                return scat_cov(
                    self.backend.bk_square(self.backend.bk_abs(x.S0)),
                    self.backend.bk_square(self.backend.bk_abs(x.P00)),
                    self.backend.bk_square(self.backend.bk_abs(x.C01)),
                    self.backend.bk_square(self.backend.bk_abs(x.C11)),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
            else:
                return scat_cov(
                    self.backend.bk_square(self.backend.bk_abs(x.S0)),
                    self.backend.bk_square(self.backend.bk_abs(x.P00)),
                    self.backend.bk_square(self.backend.bk_abs(x.C01)),
                    self.backend.bk_square(self.backend.bk_abs(x.C11)),
                    s1=self.backend.bk_square(self.backend.bk_abs(x.S1)),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
        else:
            return self.backend.bk_abs(self.backend.bk_square(x))

    def sqrt(self, x):
        if isinstance(x, scat_cov):
            if x.S1 is None:
                return scat_cov(
                    self.backend.bk_sqrt(self.backend.bk_abs(x.S0)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.P00)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.C01)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.C11)),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
            else:
                return scat_cov(
                    self.backend.bk_sqrt(self.backend.bk_abs(x.S0)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.P00)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.C01)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.C11)),
                    s1=self.backend.bk_sqrt(self.backend.bk_abs(x.S1)),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
        else:
            return self.backend.bk_abs(self.backend.bk_sqrt(x))

    def reduce_mean(self, x):
        if isinstance(x, scat_cov):
            if x.S1 is None:
                result = (
                    self.backend.bk_reduce_mean(self.backend.bk_abs(x.S0))
                    + self.backend.bk_reduce_mean(self.backend.bk_abs(x.P00))
                    + self.backend.bk_reduce_mean(self.backend.bk_abs(x.C01))
                    + self.backend.bk_reduce_mean(self.backend.bk_abs(x.C11))
                ) / 3
            else:
                result = (
                    self.backend.bk_reduce_mean(self.backend.bk_abs(x.S0))
                    + self.backend.bk_reduce_mean(self.backend.bk_abs(x.P00))
                    + self.backend.bk_reduce_mean(self.backend.bk_abs(x.S1))
                    + self.backend.bk_reduce_mean(self.backend.bk_abs(x.C01))
                    + self.backend.bk_reduce_mean(self.backend.bk_abs(x.C11))
                ) / 4
        else:
            return self.backend.bk_reduce_mean(x)
        return result

    def reduce_distance(self, x, y, sigma=None):

        if isinstance(x, scat_cov):
            if sigma is None:
                result = self.diff_data(y.S0, x.S0, is_complex=False)
                if x.S1 is not None:
                    result += self.diff_data(y.S1, x.S1)
                if x.C10 is not None:
                    result += self.diff_data(y.C10, x.C10)
                result += self.diff_data(y.P00, x.P00)
                result += self.diff_data(y.C01, x.C01)
                result += self.diff_data(y.C11, x.C11)
            else:
                result = self.diff_data(y.S0, x.S0, is_complex=False, sigma=sigma.S0)
                if x.S1 is not None:
                    result += self.diff_data(y.S1, x.S1, sigma=sigma.S1)
                if x.C10 is not None:
                    result += self.diff_data(y.C10, x.C10, sigma=sigma.C10)
                result += self.diff_data(y.P00, x.P00, sigma=sigma.P00)
                result += self.diff_data(y.C01, x.C01, sigma=sigma.C01)
                result += self.diff_data(y.C11, x.C11, sigma=sigma.C11)
            nval = (
                self.backend.bk_size(x.S0)
                + self.backend.bk_size(x.P00)
                + self.backend.bk_size(x.C01)
                + self.backend.bk_size(x.C11)
            )
            if x.S1 is not None:
                nval += self.backend.bk_size(x.S1)
            if x.C10 is not None:
                nval += self.backend.bk_size(x.C10)
            result /= self.backend.bk_cast(nval)
        else:
            return self.backend.bk_reduce_sum(x)
        return result

    def reduce_sum(self, x):

        if isinstance(x, scat_cov):
            if x.S1 is None:
                result = (
                    self.backend.bk_reduce_sum(x.S0)
                    + self.backend.bk_reduce_sum(x.P00)
                    + self.backend.bk_reduce_sum(x.C01)
                    + self.backend.bk_reduce_sum(x.C11)
                )
            else:
                result = (
                    self.backend.bk_reduce_sum(x.S0)
                    + self.backend.bk_reduce_sum(x.P00)
                    + self.backend.bk_reduce_sum(x.S1)
                    + self.backend.bk_reduce_sum(x.C01)
                    + self.backend.bk_reduce_sum(x.C11)
                )
        else:
            return self.backend.bk_reduce_sum(x)
        return result

    def ldiff(self, sig, x):

        if x.S1 is None:
            if x.C10 is not None:
                return scat_cov(
                    x.domult(sig.S0, x.S0) * x.domult(sig.S0, x.S0),
                    x.domult(sig.P00, x.P00) * x.domult(sig.P00, x.P00),
                    x.domult(sig.C01, x.C01) * x.domult(sig.C01, x.C01),
                    x.domult(sig.C11, x.C11) * x.domult(sig.C11, x.C11),
                    C10=x.domult(sig.C10, x.C10) * x.domult(sig.C10, x.C10),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
            else:
                return scat_cov(
                    x.domult(sig.S0, x.S0) * x.domult(sig.S0, x.S0),
                    x.domult(sig.P00, x.P00) * x.domult(sig.P00, x.P00),
                    x.domult(sig.C01, x.C01) * x.domult(sig.C01, x.C01),
                    x.domult(sig.C11, x.C11) * x.domult(sig.C11, x.C11),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
        else:
            if x.C10 is None:
                return scat_cov(
                    x.domult(sig.S0, x.S0) * x.domult(sig.S0, x.S0),
                    x.domult(sig.P00, x.P00) * x.domult(sig.P00, x.P00),
                    x.domult(sig.C01, x.C01) * x.domult(sig.C01, x.C01),
                    x.domult(sig.C11, x.C11) * x.domult(sig.C11, x.C11),
                    S1=x.domult(sig.S1, x.S1) * x.domult(sig.S1, x.S1),
                    C10=x.domult(sig.C10, x.C10) * x.domult(sig.C10, x.C10),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
            else:
                return scat_cov(
                    x.domult(sig.S0, x.S0) * x.domult(sig.S0, x.S0),
                    x.domult(sig.P00, x.P00) * x.domult(sig.P00, x.P00),
                    x.domult(sig.C01, x.C01) * x.domult(sig.C01, x.C01),
                    x.domult(sig.C11, x.C11) * x.domult(sig.C11, x.C11),
                    S1=x.domult(sig.S1, x.S1) * x.domult(sig.S1, x.S1),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )

    def log(self, x):
        if isinstance(x, scat_cov):

            if x.S1 is None:
                result = (
                    self.backend.bk_log(x.S0)
                    + self.backend.bk_log(x.P00)
                    + self.backend.bk_log(x.C01)
                    + self.backend.bk_log(x.C11)
                )
            else:
                result = (
                    self.backend.bk_log(x.S0)
                    + self.backend.bk_log(x.P00)
                    + self.backend.bk_log(x.S1)
                    + self.backend.bk_log(x.C01)
                    + self.backend.bk_log(x.C11)
                )
        else:
            return self.backend.bk_log(x)

        return result

    #    # ---------------------------------------------−---------
    #    def std(self, list_of_sc):
    #        n = len(list_of_sc)
    #        res = list_of_sc[0]
    #        res2 = list_of_sc[0] * list_of_sc[0]
    #        for k in range(1, n):
    #            res = res + list_of_sc[k]
    #            res2 = res2 + list_of_sc[k] * list_of_sc[k]
    #
    #        if res.S1 is None:
    #            if res.C10 is not None:
    #                return scat_cov(
    #                    res.domult(sig.S0, res.S0) * res.domult(sig.S0, res.S0),
    #                    res.domult(sig.P00, res.P00) * res.domult(sig.P00, res.P00),
    #                    res.domult(sig.C01, res.C01) * res.domult(sig.C01, res.C01),
    #                    res.domult(sig.C11, res.C11) * res.domult(sig.C11, res.C11),
    #                    C10=res.domult(sig.C10, res.C10) * res.domult(sig.C10, res.C10),
    #                    backend=self.backend,
    #                    use_1D=self.use_1D,
    #                )
    #            else:
    #                return scat_cov(
    #                    res.domult(sig.S0, res.S0) * res.domult(sig.S0, res.S0),
    #                    res.domult(sig.P00, res.P00) * res.domult(sig.P00, res.P00),
    #                    res.domult(sig.C01, res.C01) * res.domult(sig.C01, res.C01),
    #                    res.domult(sig.C11, res.C11) * res.domult(sig.C11, res.C11),
    #                    backend=self.backend,
    #                    use_1D=self.use_1D,
    #                )
    #        else:
    #            if res.C10 is None:
    #                return scat_cov(
    #                    res.domult(sig.S0, res.S0) * res.domult(sig.S0, res.S0),
    #                    res.domult(sig.P00, res.P00) * res.domult(sig.P00, res.P00),
    #                    res.domult(sig.C01, res.C01) * res.domult(sig.C01, res.C01),
    #                    res.domult(sig.C11, res.C11) * res.domult(sig.C11, res.C11),
    #                    S1=res.domult(sig.S1, res.S1) * res.domult(sig.S1, res.S1),
    #                    C10=res.domult(sig.C10, res.C10) * res.domult(sig.C10, res.C10),
    #                    backend=self.backend,
    #                )
    #            else:
    #                return scat_cov(
    #                    res.domult(sig.P00, res.P00) * res.domult(sig.P00, res.P00),
    #                    res.domult(sig.S1, res.S1) * res.domult(sig.S1, res.S1),
    #                    res.domult(sig.C01, res.C01) * res.domult(sig.C01, res.C01),
    #                    res.domult(sig.C11, res.C11) * res.domult(sig.C11, res.C11),
    #                    backend=self.backend,
    #                    use_1D=self.use_1D,
    #                )
    #        return self.NORIENT

    @tf_function
    def eval_comp_fast(
        self,
        image1,
        image2=None,
        mask=None,
        norm=None,
        Auto=True,
        cmat=None,
        cmat2=None,
    ):

        res = self.eval(
            image1, image2=image2, mask=mask, Auto=Auto, cmat=cmat, cmat2=cmat2
        )
        return res.S0, res.P00, res.S1, res.C01, res.C11, res.C10

    def eval_fast(
        self,
        image1,
        image2=None,
        mask=None,
        norm=None,
        Auto=True,
        cmat=None,
        cmat2=None,
    ):
        s0, p0, s1, c01, c11, c10 = self.eval_comp_fast(
            image1, image2=image2, mask=mask, Auto=Auto, cmat=cmat, cmat2=cmat2
        )
        return scat_cov(
            s0, p0, c01, c11, s1=s1, c10=c10, backend=self.backend, use_1D=self.use_1D
        )
