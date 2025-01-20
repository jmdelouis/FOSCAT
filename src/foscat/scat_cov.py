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
        self, s0, s2, s3, s4, s1=None, s3p=None, backend=None, use_1D=False
    ):
        self.S0 = s0
        self.S2 = s2
        self.S3 = s3
        self.S4 = s4
        self.S1 = s1
        self.S3P = s3p
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
        if self.S3P is None:
            s3p = None
        else:
            s3p = self.S3P.numpy()

        return scat_cov(
            (self.S0.numpy()),
            (self.S2.numpy()),
            (self.S3.numpy()),
            (self.S4.numpy()),
            s1=s1,
            s3p=s3p,
            backend=self.backend,
            use_1D=self.use_1D,
        )

    def constant(self):

        if self.S1 is None:
            s1 = None
        else:
            s1 = self.backend.constant(self.S1)
        if self.S3P is None:
            s3p = None
        else:
            s3p = self.backend.constant(self.S3P)

        return scat_cov(
            self.backend.constant(self.S0),
            self.backend.constant(self.S2),
            self.backend.constant(self.S3),
            self.backend.constant(self.S4),
            s1=s1,
            s3p=s3p,
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
                        self.S2,
                        [self.S1.shape[0], self.S1.shape[1] * self.S1.shape[2]],
                    )
                ),
                self.conv2complex(
                    self.backend.bk_reshape(
                        self.S3,
                        [self.S3.shape[0], self.S3.shape[1] * self.S3.shape[2]],
                    )
                ),
            ]
            if self.S3P is not None:
                tmp = tmp + [
                    self.conv2complex(
                        self.backend.bk_reshape(
                            self.S3P,
                            [self.S3.shape[0], self.S3.shape[1] * self.S3.shape[2]],
                        )
                    )
                ]

            tmp = tmp + [
                self.conv2complex(
                    self.backend.bk_reshape(
                        self.S4,
                        [self.S3.shape[0], self.S4.shape[1] * self.S4.shape[2]],
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
                    self.S2,
                    [
                        self.S1.shape[0],
                        self.S1.shape[1] * self.S1.shape[2] * self.S1.shape[3],
                    ],
                )
            ),
            self.conv2complex(
                self.backend.bk_reshape(
                    self.S3,
                    [
                        self.S3.shape[0],
                        self.S3.shape[1]
                        * self.S3.shape[2]
                        * self.S3.shape[3]
                        * self.S3.shape[4],
                    ],
                )
            ),
        ]
        if self.S3P is not None:
            tmp = tmp + [
                self.conv2complex(
                    self.backend.bk_reshape(
                        self.S3P,
                        [
                            self.S3.shape[0],
                            self.S3.shape[1]
                            * self.S3.shape[2]
                            * self.S3.shape[3]
                            * self.S3.shape[4],
                        ],
                    )
                )
            ]

        tmp = tmp + [
            self.conv2complex(
                self.backend.bk_reshape(
                    self.S4,
                    [
                        self.S3.shape[0],
                        self.S4.shape[1]
                        * self.S4.shape[2]
                        * self.S4.shape[3]
                        * self.S4.shape[4]
                        * self.S4.shape[5],
                    ],
                )
            )
        ]

        return self.backend.bk_concat(tmp, 1)

    # ---------------------------------------------−---------
    def flattenMask(self):
        if isinstance(self.S2, np.ndarray):
            if self.S1 is None:
                if self.S3P is None:
                    tmp = np.concatenate(
                        [
                            self.S0[0].flatten(),
                            self.S2[0].flatten(),
                            self.S3[0].flatten(),
                            self.S4[0].flatten(),
                        ],
                        0,
                    )
                else:
                    tmp = np.concatenate(
                        [
                            self.S0[0].flatten(),
                            self.S2[0].flatten(),
                            self.S3[0].flatten(),
                            self.S3P[0].flatten(),
                            self.S4[0].flatten(),
                        ],
                        0,
                    )
            else:
                if self.S3P is None:
                    tmp = np.concatenate(
                        [
                            self.S0[0].flatten(),
                            self.S1[0].flatten(),
                            self.S2[0].flatten(),
                            self.S3[0].flatten(),
                            self.S4[0].flatten(),
                        ],
                        0,
                    )
                else:
                    tmp = np.concatenate(
                        [
                            self.S0[0].flatten(),
                            self.S1[0].flatten(),
                            self.S2[0].flatten(),
                            self.S3[0].flatten(),
                            self.S3P[0].flatten(),
                            self.S4[0].flatten(),
                        ],
                        0,
                    )
            tmp = np.expand_dims(tmp, 0)

            for k in range(1, self.S2.shape[0]):
                if self.S1 is None:
                    if self.S3P is None:
                        ltmp = np.concatenate(
                            [
                                self.S0[k].flatten(),
                                self.S2[k].flatten(),
                                self.S3[k].flatten(),
                                self.S4[k].flatten(),
                            ],
                            0,
                        )
                    else:
                        ltmp = np.concatenate(
                            [
                                self.S0[k].flatten(),
                                self.S2[k].flatten(),
                                self.S3[k].flatten(),
                                self.S3P[k].flatten(),
                                self.S4[k].flatten(),
                            ],
                            0,
                        )
                else:
                    if self.S3P is None:
                        ltmp = np.concatenate(
                            [
                                self.S0[k].flatten(),
                                self.S1[k].flatten(),
                                self.S2[k].flatten(),
                                self.S3[k].flatten(),
                                self.S4[k].flatten(),
                            ],
                            0,
                        )
                    else:
                        ltmp = np.concatenate(
                            [
                                self.S0[k].flatten(),
                                self.S1[k].flatten(),
                                self.S2[k].flatten(),
                                self.S3[k].flatten(),
                                self.S3P[k].flatten(),
                                self.S4[k].flatten(),
                            ],
                            0,
                        )

                tmp = np.concatenate([tmp, np.expand_dims(ltmp, 0)], 0)

            return tmp
        else:
            if self.S1 is None:
                if self.S3P is None:
                    tmp = self.backend.bk_concat(
                        [
                            self.backend.bk_flattenR(self.S0[0]),
                            self.backend.bk_flattenR(self.S2[0]),
                            self.backend.bk_flattenR(self.S3[0]),
                            self.backend.bk_flattenR(self.S4[0]),
                        ],
                        0,
                    )
                else:
                    tmp = self.backend.bk_concat(
                        [
                            self.backend.bk_flattenR(self.S0[0]),
                            self.backend.bk_flattenR(self.S2[0]),
                            self.backend.bk_flattenR(self.S3[0]),
                            self.backend.bk_flattenR(self.S3P[0]),
                            self.backend.bk_flattenR(self.S4[0]),
                        ],
                        0,
                    )
            else:
                if self.S3P is None:
                    tmp = self.backend.bk_concat(
                        [
                            self.backend.bk_flattenR(self.S0[0]),
                            self.backend.bk_flattenR(self.S1[0]),
                            self.backend.bk_flattenR(self.S2[0]),
                            self.backend.bk_flattenR(self.S3[0]),
                            self.backend.bk_flattenR(self.S4[0]),
                        ],
                        0,
                    )
                else:
                    tmp = self.backend.bk_concat(
                        [
                            self.backend.bk_flattenR(self.S0[0]),
                            self.backend.bk_flattenR(self.S1[0]),
                            self.backend.bk_flattenR(self.S2[0]),
                            self.backend.bk_flattenR(self.S3[0]),
                            self.backend.bk_flattenR(self.S3P[0]),
                            self.backend.bk_flattenR(self.S4[0]),
                        ],
                        0,
                    )
            tmp = self.backend.bk_expand_dims(tmp, 0)

            for k in range(1, self.S2.shape[0]):
                if self.S1 is None:
                    if self.S3P is None:
                        ltmp = self.backend.bk_concat(
                            [
                                self.backend.bk_flattenR(self.S0[k]),
                                self.backend.bk_flattenR(self.S2[k]),
                                self.backend.bk_flattenR(self.S3[k]),
                                self.backend.bk_flattenR(self.S4[k]),
                            ],
                            0,
                        )
                    else:
                        ltmp = self.backend.bk_concat(
                            [
                                self.backend.bk_flattenR(self.S0[k]),
                                self.backend.bk_flattenR(self.S2[k]),
                                self.backend.bk_flattenR(self.S3[k]),
                                self.backend.bk_flattenR(self.S3P[k]),
                                self.backend.bk_flattenR(self.S4[k]),
                            ],
                            0,
                        )
                else:
                    if self.S3P is None:
                        ltmp = self.backend.bk_concat(
                            [
                                self.backend.bk_flattenR(self.S0[k]),
                                self.backend.bk_flattenR(self.S1[k]),
                                self.backend.bk_flattenR(self.S2[k]),
                                self.backend.bk_flattenR(self.S3[k]),
                                self.backend.bk_flattenR(self.S4[k]),
                            ],
                            0,
                        )
                    else:
                        ltmp = self.backend.bk_concat(
                            [
                                self.backend.bk_flattenR(self.S0[k]),
                                self.backend.bk_flattenR(self.S1[k]),
                                self.backend.bk_flattenR(self.S2[k]),
                                self.backend.bk_flattenR(self.S3[k]),
                                self.backend.bk_flattenR(self.S3P[k]),
                                self.backend.bk_flattenR(self.S4[k]),
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

    def get_S2(self):
        return self.S2

    def reset_S2(self):
        self.S2 = 0 * self.S2

    def get_S3(self):
        return self.S3

    def get_S3P(self):
        return self.S3P

    def get_S4(self):
        return self.S4

    def get_j_idx(self):
        shape = list(self.S2.shape)
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

    def get_js4_idx(self):
        shape = list(self.S2.shape)
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

        if self.S3P is None:
            s3p = None
        else:
            if isinstance(other, scat_cov):
                if other.S3P is None:
                    s3p = None
                else:
                    s3p = self.doadd(self.S3P, other.S3P)
            else:
                s3p = self.S3P + other

        if self.S4 is None:
            s4 = None
        else:
            if isinstance(other, scat_cov):
                if other.S4 is None:
                    s4 = None
                else:
                    s4 = self.doadd(self.S4, other.S4)
            else:
                s4 = self.S4 + other

        if isinstance(other, scat_cov):
            return scat_cov(
                self.doadd(self.S0, other.S0),
                self.doadd(self.S2, other.S2),
                (self.S3 + other.S3),
                s4,
                s1=s1,
                s3p=s3p,
                backend=self.backend,
                use_1D=self.use_1D,
            )
        else:
            return scat_cov(
                (self.S0 + other),
                (self.S2 + other),
                (self.S3 + other),
                s4,
                s1=s1,
                s3p=s3p,
                backend=self.backend,
                use_1D=self.use_1D,
            )

    def relu(self):

        if self.S1 is None:
            s1 = None
        else:
            s1 = self.backend.bk_relu(self.S1)

        if self.S3P is None:
            s3p = None
        else:
            s3p = self.backend.bk_relu(self.s3p)

        if self.S4 is None:
            s4 = None
        else:
            s4 = self.backend.bk_relu(self.s4)

        return scat_cov(
            self.backend.bk_relu(self.S0),
            self.backend.bk_relu(self.S2),
            self.backend.bk_relu(self.S3),
            s4,
            s1=s1,
            s3p=s3p,
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

        if self.S3P is None:
            s3p = None
        else:
            if isinstance(other, scat_cov):
                if other.S3P is None:
                    s3p = None
                else:
                    s3p = self.dodiv(self.S3P, other.S3P)
            else:
                s3p = self.dodiv(self.S3P, other)

        if self.S4 is None:
            s4 = None
        else:
            if isinstance(other, scat_cov):
                if other.S4 is None:
                    s4 = None
                else:
                    s4 = self.dodiv(self.S4, other.S4)
            else:
                s4 = self.S4 / other

        if isinstance(other, scat_cov):
            return scat_cov(
                self.dodiv(self.S0, other.S0),
                self.dodiv(self.S2, other.S2),
                self.dodiv(self.S3, other.S3),
                s4,
                s1=s1,
                s3p=s3p,
                backend=self.backend,
                use_1D=self.use_1D,
            )
        else:
            return scat_cov(
                (self.S0 / other),
                (self.S2 / other),
                (self.S3 / other),
                s4,
                s1=s1,
                s3p=s3p,
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

        if self.S3P is None:
            s3p = None
        else:
            if isinstance(other, scat_cov):
                s3p = self.dodiv(other.S3P, self.S3P)
            else:
                s3p = other / self.S3P

        if self.S4 is None:
            s4 = None
        else:
            if isinstance(other, scat_cov):
                if other.S4 is None:
                    s4 = None
                else:
                    s4 = self.dodiv(other.S4, self.S4)
            else:
                s4 = other / self.S4

        if isinstance(other, scat_cov):
            return scat_cov(
                self.dodiv(other.S0, self.S0),
                self.dodiv(other.S2, self.S2),
                (other.S3 / self.S3),
                s4,
                s1=s1,
                s3p=s3p,
                backend=self.backend,
                use_1D=self.use_1D,
            )
        else:
            return scat_cov(
                (other / self.S0),
                (other / self.S2),
                (other / self.S3),
                (other / self.S4),
                s1=s1,
                s3p=s3p,
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

        if self.S3P is None:
            s3p = None
        else:
            if isinstance(other, scat_cov):
                if other.S3P is None:
                    s3p = None
                else:
                    s3p = self.domin(other.S3P, self.S3P)
            else:
                s3p = other - self.S3P

        if self.S4 is None:
            s4 = None
        else:
            if isinstance(other, scat_cov):
                if other.S4 is None:
                    s4 = None
                else:
                    s4 = self.domin(other.S4, self.S4)
            else:
                s4 = other - self.S4

        if isinstance(other, scat_cov):
            return scat_cov(
                self.domin(other.S0, self.S0),
                self.domin(other.S2, self.S2),
                (other.S3 - self.S3),
                s4,
                s1=s1,
                s3p=s3p,
                backend=self.backend,
                use_1D=self.use_1D,
            )
        else:
            return scat_cov(
                (other - self.S0),
                (other - self.S2),
                (other - self.S3),
                s4,
                s1=s1,
                s3p=s3p,
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

        if self.S3P is None:
            s3p = None
        else:
            if isinstance(other, scat_cov):
                if other.S3P is None:
                    s3p = None
                else:
                    s3p = self.domin(self.S3P, other.S3P)
            else:
                s3p = self.S3P - other

        if self.S4 is None:
            s4 = None
        else:
            if isinstance(other, scat_cov):
                if other.S4 is None:
                    s4 = None
                else:
                    s4 = self.domin(self.S4, other.S4)
            else:
                s4 = self.S4 - other

        if isinstance(other, scat_cov):
            return scat_cov(
                self.domin(self.S0, other.S0),
                self.domin(self.S2, other.S2),
                (self.S3 - other.S3),
                s4,
                s1=s1,
                s3p=s3p,
                backend=self.backend,
                use_1D=self.use_1D,
            )
        else:
            return scat_cov(
                (self.S0 - other),
                (self.S2 - other),
                (self.S3 - other),
                s4,
                s1=s1,
                s3p=s3p,
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

        if self.S3P is None:
            s3p = None
        else:
            if isinstance(other, scat_cov):
                if other.S3P is None:
                    s3p = None
                else:
                    s3p = self.domult(self.S3P, other.S3P)
            else:
                s3p = self.S3P * other

        if self.S4 is None:
            s4 = None
        else:
            if isinstance(other, scat_cov):
                if other.S4 is None:
                    s4 = None
                else:
                    s4 = self.domult(self.S4, other.S4)
            else:
                s4 = self.S4 * other

        if isinstance(other, scat_cov):
            return scat_cov(
                self.domult(self.S0, other.S0),
                self.domult(self.S2, other.S2),
                self.domult(self.S3, other.S3),
                s4,
                s1=s1,
                s3p=s3p,
                backend=self.backend,
                use_1D=self.use_1D,
            )
        else:
            return scat_cov(
                (self.S0 * other),
                (self.S2 * other),
                (self.S3 * other),
                s4,
                s1=s1,
                s3p=s3p,
                backend=self.backend,
                use_1D=self.use_1D,
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    # ---------------------------------------------−---------
    def interp(self, nscale, extend=True, constant=False):

        if nscale + 2 > self.S2.shape[2]:
            print(
                "Can not *interp* %d with a statistic described over %d"
                % (nscale, self.S2.shape[2])
            )
            return scat_cov(
                self.S2,
                self.S3,
                self.S4,
                s1=self.S1,
                s3p=self.S3P,
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
            s2 = self.S2
        else:
            s2 = self.S2.numpy()

        for k in range(nscale):
            if constant:
                if self.S1 is not None:
                    s1[:, :, nscale - 1 - k, :] = s1[:, :, nscale - k, :]
                s2[:, :, nscale - 1 - k, :] = s2[:, :, nscale - k, :]
            else:
                if self.S1 is not None:
                    s1[:, :, nscale - 1 - k, :] = np.exp(
                        2 * np.log(s1[:, :, nscale - k, :])
                        - np.log(s1[:, :, nscale + 1 - k, :])
                    )
                s2[:, :, nscale - 1 - k, :] = np.exp(
                    2 * np.log(s2[:, :, nscale - k, :])
                    - np.log(s2[:, :, nscale + 1 - k, :])
                )

        j1, j2 = self.get_j_idx()

        if self.S3P is not None:
            if self.BACKEND == "numpy":
                s3p = self.S3P
            else:
                s3p = self.S3P.numpy()
        else:
            s3p = self.S3P
        if self.BACKEND == "numpy":
            s3 = self.S3
        else:
            s3 = self.S3.numpy()

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
                    s3p[:, :, i0] = s3p[:, :, i1]
                    s3[:, :, i0] = s3[:, :, i1]
                else:
                    s3p[:, :, i0] = np.exp(
                        2 * np.log(s3p[:, :, i1]) - np.log(s3p[:, :, i2])
                    )
                    s3[:, :, i0] = np.exp(
                        2 * np.log(s3[:, :, i1]) - np.log(s3[:, :, i2])
                    )

        if self.BACKEND == "numpy":
            s4 = self.S4
        else:
            s4 = self.S4.numpy()

        j1, j2, j3 = self.get_js4_idx()

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
                    s4[:, :, i0] = s4[:, :, i1]
                else:
                    s4[:, :, i0] = np.exp(
                        2 * np.log(s4[:, :, i1]) - np.log(s4[:, :, i2])
                    )

        if s1 is not None:
            s1 = self.backend.constant(s1)
        if s3p is not None:
            s3p = self.backend.constant(s3p)

        return scat_cov(
            self.S0,
            self.backend.constant(s2),
            self.backend.constant(s3),
            self.backend.constant(s4),
            s1=s1,
            s3p=s3p,
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
        tmp = abs(self.get_np(self.S2))
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
                                label=r"%s $S_2$" % (name),
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
                            label=r"%s $S_2$" % (name),
                            lw=lw,
                        )
                    else:
                        plt.plot(tmp[i1, i2, :], color=color, lw=lw)
        plt.yscale("log")
        plt.ylabel("$S_2$")
        plt.xlabel(r"$j_{1}$")
        plt.legend(frameon=0)

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
            plt.legend(frameon=0)
            if norm:
                plt.ylabel(r"$\frac{S_1}{\sqrt{S_2}}$")
            else:
                plt.ylabel("$S_1$")
            plt.xlabel(r"$j_{1}$")

        ax1 = plt.subplot(2, 2, 3)
        ax2 = ax1.twiny()
        n = 0
        tmp = abs(self.get_np(self.S3))
        if norm:
            lname = r"%s norm. $S_{3}$" % (name)
            ax1.set_ylabel(r"$\frac{S_3}{\sqrt{S_{2,j_1}S_{2,j_2}}}$")
        else:
            lname = r"%s $S_3$" % (name)
            ax1.set_ylabel(r"$S_3$")

        if self.S3P is not None:
            tmp = abs(self.get_np(self.S3))
            if norm:
                lname = r"%s norm. $\tilde{S}_{3}$" % (name)
                ax1.set_ylabel(r"$\frac{\tilde{S}_{3}}{\sqrt{S_{2,j_1}S_{2,j_2}}}$")
            else:
                lname = r"%s $\tilde{S}_{3}$" % (name)
                ax1.set_ylabel(r"$\tilde{S}_{3}$")

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
        j1, j2, j3 = self.get_js4_idx()
        ax2 = ax1.twiny()
        n = 1
        tmp = abs(self.get_np(self.S4))
        lname = r"%s $S_4$" % (name)
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
            ax1.set_ylabel(r"$\frac{S_4}{\sqrt{S_{2,j_1}S_{2,j_2}}}$")
        else:
            ax1.set_ylabel(r"$S_4$")

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
            self.get_np(self.S3P),
            self.get_np(self.S3),
            self.get_np(self.S4),
            self.get_np(self.S2),
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
            s3p=outlist[2],
            backend=self.backend,
            use_1D=self.use_1D,
        )

    def std(self):
        if self.S1 is not None:  # Auto
            return np.sqrt(
                (
                    (abs(self.get_np(self.S0)).std()) ** 2
                    + (abs(self.get_np(self.S1)).std()) ** 2
                    + (abs(self.get_np(self.S3)).std()) ** 2
                    + (abs(self.get_np(self.S4)).std()) ** 2
                    + (abs(self.get_np(self.S2)).std()) ** 2
                )
                / 4
            )
        else:  # Cross
            return np.sqrt(
                (
                    (abs(self.get_np(self.S0)).std()) ** 2
                    + (abs(self.get_np(self.S3)).std()) ** 2
                    + (abs(self.get_np(self.S3P)).std()) ** 2
                    + (abs(self.get_np(self.S4)).std()) ** 2
                    + (abs(self.get_np(self.S2)).std()) ** 2
                )
                / 4
            )

    def mean(self):
        if self.S1 is not None:  # Auto
            return (
                abs(self.get_np(self.S0)).mean()
                + abs(self.get_np(self.S1)).mean()
                + abs(self.get_np(self.S3)).mean()
                + abs(self.get_np(self.S4)).mean()
                + abs(self.get_np(self.S2)).mean()
            ) / 4
        else:  # Cross
            return (
                abs(self.get_np(self.S0)).mean()
                + abs(self.get_np(self.S3)).mean()
                + abs(self.get_np(self.S3P)).mean()
                + abs(self.get_np(self.S4)).mean()
                + abs(self.get_np(self.S2)).mean()
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
        s3p = None

        if self.S1 is not None:
            s1 = self.backend.bk_sqrt(self.S1)
        if self.S3P is not None:
            s3p = self.backend.bk_sqrt(self.S3P)

        s0 = self.backend.bk_sqrt(self.S0)
        s2 = self.backend.bk_sqrt(self.S2)
        s3 = self.backend.bk_sqrt(self.S3)
        s4 = self.backend.bk_sqrt(self.S4)

        return scat_cov(
            s0, s2, s3, s4, s1=s1, s3p=s3p, backend=self.backend, use_1D=self.use_1D
        )

    def L1(self):

        s1 = None
        s3p = None

        if self.S1 is not None:
            s1 = self.backend.bk_L1(self.S1)
        if self.S3P is not None:
            s3p = self.backend.bk_L1(self.S3P)

        s0 = self.backend.bk_L1(self.S0)
        s2 = self.backend.bk_L1(self.S2)
        s3 = self.backend.bk_L1(self.S3)
        s4 = self.backend.bk_L1(self.S4)

        return scat_cov(
            s0, s2, s3, s4, s1=s1, s3p=s3p, backend=self.backend, use_1D=self.use_1D
        )

    def square_comp(self):

        s1 = None
        s3p = None

        if self.S1 is not None:
            s1 = self.backend.bk_square_comp(self.S1)
        if self.S3P is not None:
            s3p = self.backend.bk_square_comp(self.S3P)

        s0 = self.backend.bk_square_comp(self.S0)
        s2 = self.backend.bk_square_comp(self.S2)
        s3 = self.backend.bk_square_comp(self.S3)
        s4 = self.backend.bk_square_comp(self.S4)

        return scat_cov(
            s0, s2, s3, s4, s1=s1, s3p=s3p, backend=self.backend, use_1D=self.use_1D
        )

    def iso_mean(self, repeat=False):
        shape = list(self.S2.shape)
        norient = shape[3]

        S1 = self.S1
        if self.S1 is not None:
            S1 = self.backend.bk_reduce_mean(self.S1, 3)
            if repeat:
                S1 = self.backend.bk_reshape(
                    self.backend.bk_repeat(S1, norient, 2), self.S1.shape
                )
        S2 = self.backend.bk_reduce_mean(self.S2, 3)
        if repeat:
            S2 = self.backend.bk_reshape(
                self.backend.bk_repeat(S2, norient, 2), self.S2.shape
            )

        S3 = self.S3

        if norient not in self.backend._iso_orient:
            self.backend.calc_iso_orient(norient)

        shape = list(self.S3.shape)
        if self.S3 is not None:
            if self.backend.bk_is_complex(self.S3):
                lmat = self.backend._iso_orient_C[norient]
                lmat_T = self.backend._iso_orient_C_T[norient]
            else:
                lmat = self.backend._iso_orient[norient]
                lmat_T = self.backend._iso_orient_T[norient]

            S3 = self.backend.bk_reshape(
                self.backend.backend.matmul(
                    self.backend.bk_reshape(
                        self.S3, [shape[0] * shape[1] * shape[2], norient * norient]
                    ),
                    lmat,
                ),
                [shape[0], shape[1], shape[2], norient],
            )
            if repeat:
                S3 = self.backend.bk_reshape(
                    self.backend.backend.matmul(
                        self.backend.bk_reshape(
                            S3, [shape[0] * shape[1] * shape[2], norient]
                        ),
                        lmat_T,
                    ),
                    [shape[0], shape[1], shape[2], norient, norient],
                )

        S3P = self.S3P
        if self.S3P is not None:
            if self.backend.bk_is_complex(self.S3P):
                lmat = self.backend._iso_orient_C[norient]
                lmat_T = self.backend._iso_orient_C_T[norient]
            else:
                lmat = self.backend._iso_orient[norient]
                lmat_T = self.backend._iso_orient_T[norient]

            S3P = self.backend.bk_reshape(
                self.backend.backend.matmul(
                    self.backend.bk_reshape(
                        self.S3P, [shape[0] * shape[1] * shape[2], norient * norient]
                    ),
                    lmat,
                ),
                [shape[0], shape[1], shape[2], norient],
            )
            if repeat:
                S3P = self.backend.bk_reshape(
                    self.backend.backend.matmul(
                        self.backend.bk_reshape(
                            S3P, [shape[0] * shape[1] * shape[2], norient]
                        ),
                        lmat_T,
                    ),
                    [shape[0], shape[1], shape[2], norient, norient],
                )

        S4 = self.S4
        if self.S4 is not None:
            if self.backend.bk_is_complex(self.S4):
                lmat = self.backend._iso_orient_C[norient]
                lmat_T = self.backend._iso_orient_C_T[norient]
            else:
                lmat = self.backend._iso_orient[norient]
                lmat_T = self.backend._iso_orient_T[norient]

            shape = list(self.S4.shape)
            S4 = self.backend.bk_reshape(
                self.backend.backend.matmul(
                    self.backend.bk_reshape(
                        self.S4,
                        [shape[0] * shape[1] * shape[2] * norient, norient * norient],
                    ),
                    lmat,
                ),
                [shape[0], shape[1], shape[2], norient, norient],
            )
            S4 = self.backend.bk_reduce_mean(S4, 3)
            if repeat:
                S4 = self.backend.bk_reshape(
                    self.backend.bk_repeat(
                        self.backend.bk_reshape(
                            S4, [shape[0] * shape[1] * shape[2], norient]
                        ),
                        norient,
                        axis=0,
                    ),
                    [shape[0] * shape[1] * shape[2] * norient, norient],
                )
                S4 = self.backend.bk_reshape(
                    self.backend.backend.matmul(S4, lmat_T),
                    [shape[0], shape[1], shape[2], norient, norient, norient],
                )

        return scat_cov(
            self.S0,
            S2,
            S3,
            S4,
            s1=S1,
            s3p=S3P,
            backend=self.backend,
            use_1D=self.use_1D,
        )

    def fft_ang(self, nharm=1, imaginary=False):
        shape = list(self.S2.shape)
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

        if self.backend.bk_is_complex(self.S2):
            lmat = self.backend._fft_1_orient_C[(norient, nharm, imaginary)]
        else:
            lmat = self.backend._fft_1_orient[(norient, nharm, imaginary)]

        S2 = self.backend.bk_reshape(
            self.backend.backend.matmul(
                self.backend.bk_reshape(
                    self.S2, [shape[0] * shape[1] * shape[2], norient]
                ),
                lmat,
            ),
            [shape[0], shape[1], shape[2], nout],
        )

        S3 = self.S3
        shape = list(self.S3.shape)
        if self.S3 is not None:
            if self.backend.bk_is_complex(self.S3):
                lmat = self.backend._fft_2_orient_C[(norient, nharm, imaginary)]
            else:
                lmat = self.backend._fft_2_orient[(norient, nharm, imaginary)]

            S3 = self.backend.bk_reshape(
                self.backend.backend.matmul(
                    self.backend.bk_reshape(
                        self.S3, [shape[0] * shape[1] * shape[2], norient * norient]
                    ),
                    lmat,
                ),
                [shape[0], shape[1], shape[2], nout, nout],
            )

        S3P = self.S3P
        if self.S3P is not None:
            if self.backend.bk_is_complex(self.S3P):
                lmat = self.backend._fft_2_orient_C[(norient, nharm, imaginary)]
            else:
                lmat = self.backend._fft_2_orient[(norient, nharm, imaginary)]

            S3P = self.backend.bk_reshape(
                self.backend.backend.matmul(
                    self.backend.bk_reshape(
                        self.S3P, [shape[0] * shape[1] * shape[2], norient * norient]
                    ),
                    lmat,
                ),
                [shape[0], shape[1], shape[2], nout, nout],
            )

        S4 = self.S4
        if self.S4 is not None:
            if self.backend.bk_is_complex(self.S3):
                lmat = self.backend._fft_3_orient_C[(norient, nharm, imaginary)]
            else:
                lmat = self.backend._fft_3_orient[(norient, nharm, imaginary)]

            shape = list(self.S4.shape)
            S4 = self.backend.bk_reshape(
                self.backend.backend.matmul(
                    self.backend.bk_reshape(
                        self.S4,
                        [shape[0] * shape[1] * shape[2], norient * norient * norient],
                    ),
                    lmat,
                ),
                [shape[0], shape[1], shape[2], nout, nout, nout],
            )

        return scat_cov(
            self.S0,
            S2,
            S3,
            S4,
            s1=S1,
            s3p=S3P,
            backend=self.backend,
            use_1D=self.use_1D,
        )

    def iso_std(self, repeat=False):

        val = (self - self.iso_mean(repeat=True)).square_comp()
        return (val.iso_mean(repeat=repeat)).L1()

    def get_nscale(self):
        return self.S2.shape[2]

    def get_norient(self):
        return self.S2.shape[3]

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
        noff = nscale - self.S2.shape[2]
        if noff == 0:
            return scat_cov(
                (self.S0),
                (self.S2),
                (self.S3),
                (self.S4),
                s1=self.S1,
                s3p=self.S3P,
                backend=self.backend,
                use_1D=self.use_1D,
            )

        inscale = self.S2.shape[2]
        s2 = np.zeros(
            [self.S2.shape[0], self.S2.shape[1], nscale, self.S2.shape[3]],
            dtype="complex",
        )
        if self.BACKEND == "numpy":
            s2[:, :, noff:, :] = self.S2
        else:
            s2[:, :, noff:, :] = self.S2.numpy()
        for i in range(self.S2.shape[0]):
            for j in range(self.S2.shape[1]):
                for k in range(self.S2.shape[3]):
                    s2[i, j, 0:noff, k] = self.add_data_from_log_slope(
                        s2[i, j, noff:, k], noff, ds=ds
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

        s3 = np.zeros(
            [
                self.S3.shape[0],
                self.S3.shape[1],
                nout,
                self.S3.shape[3],
                self.S3.shape[4],
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

        j1 = np.zeros([self.S3.shape[2]])
        j2 = np.zeros([self.S3.shape[2]])

        n = 0
        for i in range(1, self.S2.shape[2]):
            j1[n : n + i] = np.arange(i)
            j2[n : n + i] = i
            n = n + i

        for i in range(self.S3.shape[0]):
            for j in range(self.S3.shape[1]):
                for k in range(self.S3.shape[3]):
                    for l_orient in range(self.S3.shape[4]):
                        for ij in range(noff + 1, nscale):
                            idx = np.where(jo2 == ij)[0]
                            if self.BACKEND == "numpy":
                                s3[i, j, idx[noff:], k, l_orient] = self.S3[
                                    i, j, j2 == ij - noff, k, l_orient
                                ]
                                s3[i, j, idx[:noff], k, l_orient] = (
                                    self.add_data_from_slope(
                                        self.S3[i, j, j2 == ij - noff, k, l_orient],
                                        noff,
                                        ds=ds,
                                    )
                                )
                            else:
                                s3[i, j, idx[noff:], k, l_orient] = self.S3.numpy()[
                                    i, j, j2 == ij - noff, k, l_orient
                                ]
                                s3[i, j, idx[:noff], k, l_orient] = (
                                    self.add_data_from_slope(
                                        self.S3.numpy()[
                                            i, j, j2 == ij - noff, k, l_orient
                                        ],
                                        noff,
                                        ds=ds,
                                    )
                                )

                        for ij in range(nscale):
                            idx = np.where(jo1 == ij)[0]
                            if idx.shape[0] > noff:
                                s3[i, j, idx[:noff], k, l_orient] = (
                                    self.add_data_from_slope(
                                        s3[i, j, idx[noff:], k, l_orient], noff, ds=ds
                                    )
                                )
                            else:
                                s3[i, j, idx, k, l_orient] = np.mean(
                                    s3[i, j, jo1 == ij - 1, k, l_orient]
                                )

        nout = 0
        for j3 in range(nscale):
            for j2 in range(0, j3):
                for j1 in range(0, j2):
                    nout = nout + 1

        s4 = np.zeros(
            [
                self.S4.shape[0],
                self.S4.shape[1],
                nout,
                self.S4.shape[3],
                self.S4.shape[4],
                self.S4.shape[5],
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

        ncross = self.S4.shape[2]
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
                    for i in range(self.S4.shape[0]):
                        for j in range(self.S4.shape[1]):
                            for k in range(self.S4.shape[3]):
                                for l_orient in range(self.S4.shape[4]):
                                    for m in range(self.S4.shape[5]):
                                        if self.BACKEND == "numpy":
                                            s4[i, j, idx2[noff:], k, l_orient, m] = (
                                                self.S4[i, j, idx, k, l_orient, m]
                                            )
                                            s4[i, j, idx2[:noff], k, l_orient, m] = (
                                                self.add_data_from_log_slope(
                                                    self.S4[i, j, idx, k, l_orient, m],
                                                    noff,
                                                    ds=ds,
                                                )
                                            )
                                        else:
                                            s4[
                                                i, j, idx2[noff:], k, l_orient, m
                                            ] = self.S4.numpy()[
                                                i, j, idx, k, l_orient, m
                                            ]
                                            s4[i, j, idx2[:noff], k, l_orient, m] = (
                                                self.add_data_from_log_slope(
                                                    self.S4.numpy()[
                                                        i, j, idx, k, l_orient, m
                                                    ],
                                                    noff,
                                                    ds=ds,
                                                )
                                            )

        idx = np.where(abs(s4[0, 0, :, 0, 0, 0]) == 0)[0]
        for iii in idx:
            iii1 = np.where(
                (jo1 == jo1[iii] + 1) * (jo2 == jo2[iii] + 1) * (jo3 == jo3[iii] + 1)
            )[0]
            iii2 = np.where(
                (jo1 == jo1[iii] + 2) * (jo2 == jo2[iii] + 2) * (jo3 == jo3[iii] + 2)
            )[0]
            if iii2.shape[0] > 0:
                for i in range(self.S4.shape[0]):
                    for j in range(self.S4.shape[1]):
                        for k in range(self.S4.shape[3]):
                            for l_orient in range(self.S4.shape[4]):
                                for m in range(self.S4.shape[5]):
                                    s4[i, j, iii, k, l_orient, m] = (
                                        self.add_data_from_slope(
                                            s4[i, j, [iii1, iii2], k, l_orient, m],
                                            1,
                                            ds=2,
                                        )[0]
                                    )

        idx = np.where(abs(s4[0, 0, :, 0, 0, 0]) == 0)[0]
        for iii in idx:
            iii1 = np.where(
                (jo1 == jo1[iii]) * (jo2 == jo2[iii]) * (jo3 == jo3[iii] - 1)
            )[0]
            iii2 = np.where(
                (jo1 == jo1[iii]) * (jo2 == jo2[iii]) * (jo3 == jo3[iii] - 2)
            )[0]
            if iii2.shape[0] > 0:
                for i in range(self.S4.shape[0]):
                    for j in range(self.S4.shape[1]):
                        for k in range(self.S4.shape[3]):
                            for l_orient in range(self.S4.shape[4]):
                                for m in range(self.S4.shape[5]):
                                    s4[i, j, iii, k, l_orient, m] = (
                                        self.add_data_from_slope(
                                            s4[i, j, [iii1, iii2], k, l_orient, m],
                                            1,
                                            ds=2,
                                        )[0]
                                    )

        return scat_cov(
            self.S0,
            (s2),
            (s3),
            (s4),
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
            mS2 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_mean(list_scat.S2, 0), 0
            )
            mS3 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_mean(list_scat.S3, 0), 0
            )
            mS4 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_mean(list_scat.S4, 0), 0
            )
            sS0 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_std(list_scat.S0, 0), 0
            )
            sS2 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_std(list_scat.S2, 0), 0
            )
            sS3 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_std(list_scat.S3, 0), 0
            )
            sS4 = self.backend.bk_expand_dims(
                self.backend.bk_reduce_std(list_scat.S4, 0), 0
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
            if list_scat.S3P is not None:
                mS3P = self.backend.bk_expand_dims(
                    self.backend.bk_reduce_mean(list_scat.S3P, 0), 0
                )
                sS3P = self.backend.bk_expand_dims(
                    self.backend.bk_reduce_std(list_scat.S3P, 0), 0
                )
            else:
                mS3P = None
                sS3P = None
        else:
            S0 = None
            for k in list_scat:
                tmp = list_scat[k]
                if self.BACKEND == "numpy":
                    nS0 = np.expand_dims(tmp.S0, 0)
                    nS2 = np.expand_dims(tmp.S2, 0)
                    nS3 = np.expand_dims(tmp.S3, 0)
                    nS4 = np.expand_dims(tmp.S4, 0)
                    if tmp.S3P is not None:
                        nS3P = np.expand_dims(tmp.S3P, 0)
                    if tmp.S1 is not None:
                        nS1 = np.expand_dims(tmp.S1, 0)
                else:
                    nS0 = np.expand_dims(tmp.S0.numpy(), 0)
                    nS2 = np.expand_dims(tmp.S2.numpy(), 0)
                    nS3 = np.expand_dims(tmp.S3.numpy(), 0)
                    nS4 = np.expand_dims(tmp.S4.numpy(), 0)
                    if tmp.S3P is not None:
                        nS3P = np.expand_dims(tmp.S3P.numpy(), 0)
                    if tmp.S1 is not None:
                        nS1 = np.expand_dims(tmp.S1.numpy(), 0)

                if S0 is None:
                    S0 = nS0
                    S2 = nS2
                    S3 = nS3
                    S4 = nS4
                    if tmp.S3P is not None:
                        S3P = nS3P
                    if tmp.S1 is not None:
                        S1 = nS1
                else:
                    S0 = np.concatenate([S0, nS0], 0)
                    S2 = np.concatenate([S2, nS2], 0)
                    S3 = np.concatenate([S3, nS3], 0)
                    S4 = np.concatenate([S4, nS4], 0)
                    if tmp.S3P is not None:
                        S3P = np.concatenate([S3P, nS3P], 0)
                    if tmp.S1 is not None:
                        S1 = np.concatenate([S1, nS1], 0)
            sS0 = np.std(S0, 0)
            sS2 = np.std(S2, 0)
            sS3 = np.std(S3, 0)
            sS4 = np.std(S4, 0)
            mS0 = np.mean(S0, 0)
            mS2 = np.mean(S2, 0)
            mS3 = np.mean(S3, 0)
            mS4 = np.mean(S4, 0)
            if tmp.S3P is not None:
                sS3P = np.std(S3P, 0)
                mS3P = np.mean(S3P, 0)
            else:
                sS3P = None
                mS3P = None

            if tmp.S1 is not None:
                sS1 = np.std(S1, 0)
                mS1 = np.mean(S1, 0)
            else:
                sS1 = None
                mS1 = None

        return scat_cov(
            mS0,
            mS2,
            mS3,
            mS4,
            s1=mS1,
            s3p=mS3P,
            backend=self.backend,
            use_1D=self.use_1D,
        ), scat_cov(
            sS0,
            sS2,
            sS3,
            sS4,
            s1=sS1,
            s3p=sS3P,
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
            out_nside=None
    ):
        """
        Calculates the scattering correlations for a batch of images. Mean are done over pixels.
        mean of modulus:
                        S1 = <|I * Psi_j3|>
             Normalization : take the log
        power spectrum:
                        S2 = <|I * Psi_j3|^2>
            Normalization : take the log
        orig. x modulus:
                        S3 = < (I * Psi)_j3 x (|I * Psi_j2| * Psi_j3)^* >
             Normalization : divide by (S2_j2 * S2_j3)^0.5
        modulus x modulus:
                        S4 = <(|I * psi1| * psi3)(|I * psi2| * psi3)^*>
             Normalization : divide by (S2_j1 * S2_j2)^0.5
        Parameters
        ----------
        image1: tensor
            Image on which we compute the scattering coefficients [Nbatch, Npix, 1, 1]
        image2: tensor
            Second image. If not None, we compute cross-scattering covariance coefficients.
        mask:
        norm: None or str
            If None no normalization is applied, if 'auto' normalize by the reference S2,
            if 'self' normalize by the current S2.
        all_cross: False or True
            If False compute all the coefficient even the Imaginary part,
            If True return only the terms computable in the auto case.
        Returns
        -------
        S1, S2, S3, S4 normalized
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
        S1, S2, S3, S4, S3P = None, None, None, None, None

        off_S2 = -2
        off_S3 = -3
        off_S4 = -4
        if self.use_1D:
            off_S2 = -1
            off_S3 = -1
            off_S4 = -1

        # Dictionaries for S3 computation
        M1_dic = {}  # M stands for Module M1 = |I1 * Psi|
        if cross:
            M2_dic = {}

        # S2 for normalization
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
            if out_nside is not None:
                s0 = self.backend.bk_reduce_mean(self.backend.bk_reshape(s0,[s0.shape[0],12*out_nside**2,(nside//out_nside)**2]),2)
        else:
            if not cross:
                s0, l_vs0 = self.masked_mean(I1, vmask, axis=1, calc_var=True)
            else:
                s0, l_vs0 = self.masked_mean(
                    self.backend.bk_L1(I1 * I2), vmask, axis=1, calc_var=True
                )
            vs0 = self.backend.bk_concat([l_vs0, l_vs0], 1)
            s0 = self.backend.bk_concat([s0, l_vs0], 1)

        #### COMPUTE S1, S2, S3 and S4
        nside_j3 = nside  # NSIDE start (nside_j3 = nside / 2^j3)
        for j3 in range(Jmax):
            if return_data:
                if S3 is None:
                    S3 = {}
                S3[j3] = None

                if S3P is None:
                    S3P = {}
                S3P[j3] = None

                if S4 is None:
                    S4 = {}
                S4[j3] = None

            ####### S1 and S2
            ### Make the convolution I1 * Psi_j3
            conv1 = self.convol(I1, axis=1)  # [Nbatch, Npix_j3, Norient3]

            if cmat is not None:
                tmp2 = self.backend.bk_repeat(conv1, 4, axis=-1)
                conv1 = self.backend.bk_reduce_sum(
                    self.backend.bk_reshape(
                        cmat[j3] * tmp2, [tmp2.shape[0], cmat[j3].shape[0], 4, 4]
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

                ### S2_auto = < M1^2 >_pix
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                if return_data:
                    s2 = M1_square
                else:
                    if calc_var:
                        s2, vs2 = self.masked_mean(
                            M1_square, vmask, axis=1, rank=j3, calc_var=True
                        )
                    else:
                        s2 = self.masked_mean(M1_square, vmask, axis=1, rank=j3)

                if cond_init_P1_dic:
                    # We fill P1_dic with S2 for normalisation of S3 and S4
                    P1_dic[j3] = self.backend.bk_real(s2)  # [Nbatch, Nmask, Norient3]

                # We store S2_auto to return it [Nbatch, Nmask, NS2, Norient3]
                if return_data:
                    if S2 is None:
                        S2 = {}
                    if out_nside is not None and out_nside<nside_j3:
                        s2 = self.backend.bk_reduce_mean(
                            self.backend.bk_reshape(s2,[s2.shape[0],
                                                        12*out_nside**2,
                                                        (nside_j3//out_nside)**2,
                                                        s2.shape[2]]),2)
                    S2[j3] = s2
                else:
                    if norm == "auto":  # Normalize S2
                        s2 /= P1_dic[j3]
                    if S2 is None:
                        S2 = self.backend.bk_expand_dims(
                            s2, off_S2
                        )  # Add a dimension for NS2
                        if calc_var:
                            VS2 = self.backend.bk_expand_dims(
                                vs2, off_S2
                            )  # Add a dimension for NS2
                    else:
                        S2 = self.backend.bk_concat(
                            [S2, self.backend.bk_expand_dims(s2, off_S2)], axis=2
                        )
                        if calc_var:
                            VS2 = self.backend.bk_concat(
                                [VS2, self.backend.bk_expand_dims(vs2, off_S2)],
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
                    if out_nside is not None and out_nside<nside_j3:
                        s1 = self.backend.bk_reduce_mean(
                            self.backend.bk_reshape(s1,[s1.shape[0],
                                                        12*out_nside**2,
                                                        (nside_j3//out_nside)**2,
                                                        s1.shape[2]]),2)
                    S1[j3] = s1
                else:
                    ### Normalize S1
                    if norm is not None:
                        self.div_norm(s1, (P1_dic[j3]) ** 0.5)
                    ### We store S1 for image1  [Nbatch, Nmask, NS1, Norient3]
                    if S1 is None:
                        S1 = self.backend.bk_expand_dims(
                            s1, off_S2
                        )  # Add a dimension for NS1
                        if calc_var:
                            VS1 = self.backend.bk_expand_dims(
                                vs1, off_S2
                            )  # Add a dimension for NS1
                    else:
                        S1 = self.backend.bk_concat(
                            [S1, self.backend.bk_expand_dims(s1, off_S2)], axis=2
                        )
                        if calc_var:
                            VS1 = self.backend.bk_concat(
                                [VS1, self.backend.bk_expand_dims(vs1, off_S2)], axis=2
                            )

            else:  # Cross
                ### Make the convolution I2 * Psi_j3
                conv2 = self.convol(I2, axis=1)  # [Nbatch, Npix_j3, Norient3]
                if cmat is not None:
                    tmp2 = self.backend.bk_repeat(conv2, 4, axis=-1)
                    conv2 = self.backend.bk_reduce_sum(
                        self.backend.bk_reshape(
                            cmat[j3] * tmp2, [tmp2.shape[0], cmat[j3].shape[0], 4, 4]
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

                ### S2_auto = < M2^2 >_pix
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
                    # We fill P1_dic with S2 for normalisation of S3 and S4
                    P1_dic[j3] = self.backend.bk_real(p1)  # [Nbatch, Nmask, Norient3]
                    P2_dic[j3] = self.backend.bk_real(p2)  # [Nbatch, Nmask, Norient3]

                ### S2_cross = < (I1 * Psi_j3) (I2 * Psi_j3)^* >_pix
                # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
                s2 = conv1 * self.backend.bk_conjugate(conv2)
                MX = self.backend.bk_L1(s2)
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                if return_data:
                    s2 = s2
                else:
                    if calc_var:
                        s2, vs2 = self.masked_mean(
                            s2, vmask, axis=1, rank=j3, calc_var=True
                        )
                    else:
                        s2 = self.masked_mean(s2, vmask, axis=1, rank=j3)

                if return_data:
                    if S2 is None:
                        S2 = {}
                    if out_nside is not None and out_nside<nside_j3:
                        s2 = self.backend.bk_reduce_mean(
                            self.backend.bk_reshape(s2,[s2.shape[0],
                                                        12*out_nside**2,
                                                        (nside_j3//out_nside)**2,
                                                        s2.shape[2]]),2)
                    S2[j3] = s2
                else:
                    ### Normalize S2_cross
                    if norm == "auto":
                        s2 /= (P1_dic[j3] * P2_dic[j3]) ** 0.5

                    ### Store S2_cross as complex [Nbatch, Nmask, NS2, Norient3]
                    if not all_cross:
                        s2 = self.backend.bk_real(s2)

                    if S2 is None:
                        S2 = self.backend.bk_expand_dims(
                            s2, off_S2
                        )  # Add a dimension for NS2
                        if calc_var:
                            VS2 = self.backend.bk_expand_dims(
                                vs2, off_S2
                            )  # Add a dimension for NS2
                    else:
                        S2 = self.backend.bk_concat(
                            [S2, self.backend.bk_expand_dims(s2, off_S2)], axis=2
                        )
                        if calc_var:
                            VS2 = self.backend.bk_concat(
                                [VS2, self.backend.bk_expand_dims(vs2, off_S2)],
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
                    if out_nside is not None and out_nside<nside_j3:
                        s1 = self.backend.bk_reduce_mean(
                            self.backend.bk_reshape(s1,[s1.shape[0],
                                                        12*out_nside**2,
                                                        (nside_j3//out_nside)**2,
                                                        s1.shape[2]]),2)
                    S1[j3] = s1
                else:
                    ### Normalize S1
                    if norm is not None:
                        self.div_norm(s1, (P1_dic[j3]) ** 0.5)
                    ### We store S1 for image1  [Nbatch, Nmask, NS1, Norient3]
                    if S1 is None:
                        S1 = self.backend.bk_expand_dims(
                            s1, off_S2
                        )  # Add a dimension for NS1
                        if calc_var:
                            VS1 = self.backend.bk_expand_dims(
                                vs1, off_S2
                            )  # Add a dimension for NS1
                    else:
                        S1 = self.backend.bk_concat(
                            [S1, self.backend.bk_expand_dims(s1, off_S2)], axis=2
                        )
                        if calc_var:
                            VS1 = self.backend.bk_concat(
                                [VS1, self.backend.bk_expand_dims(vs1, off_S2)], axis=2
                            )

            # Initialize dictionaries for |I1*Psi_j| * Psi_j3
            M1convPsi_dic = {}
            if cross:
                # Initialize dictionaries for |I2*Psi_j| * Psi_j3
                M2convPsi_dic = {}

            ###### S3
            nside_j2=nside_j3
            for j2 in range(0, j3 + 1):  # j2 <= j3
                if return_data:
                    if S4[j3] is None:
                        S4[j3] = {}
                    S4[j3][j2] = None

                ### S3_auto = < (I1 * Psi)_j3 x (|I1 * Psi_j2| * Psi_j3)^* >_pix
                if not cross:
                    if calc_var:
                        s3, vs3 = self._compute_S3(
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
                        s3 = self._compute_S3(
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
                        if S3[j3] is None:
                            S3[j3] = {}
                        if out_nside is not None and out_nside<nside_j2:
                            s3 = self.backend.bk_reduce_mean(
                                self.backend.bk_reshape(s3,[s3.shape[0],
                                                            12*out_nside**2,
                                                            (nside_j2//out_nside)**2,
                                                            s3.shape[2],
                                                            s3.shape[3]]),2)
                        S3[j3][j2] = s3
                    else:
                        ### Normalize S3 with S2_j [Nbatch, Nmask, Norient_j]
                        if norm is not None:
                            self.div_norm(
                                s3,
                                (
                                    self.backend.bk_expand_dims(P1_dic[j2], off_S2)
                                    * self.backend.bk_expand_dims(P1_dic[j3], -1)
                                )
                                ** 0.5,
                            )  # [Nbatch, Nmask, Norient3, Norient2]

                        ### Store S3 as a complex [Nbatch, Nmask, NS3, Norient3, Norient2]
                        if S3 is None:
                            S3 = self.backend.bk_expand_dims(
                                s3, off_S3
                            )  # Add a dimension for NS3
                            if calc_var:
                                VS3 = self.backend.bk_expand_dims(
                                    vs3, off_S3
                                )  # Add a dimension for NS3
                        else:
                            S3 = self.backend.bk_concat(
                                [S3, self.backend.bk_expand_dims(s3, off_S3)], axis=2
                            )  # Add a dimension for NS3
                            if calc_var:
                                VS3 = self.backend.bk_concat(
                                    [VS3, self.backend.bk_expand_dims(vs3, off_S3)],
                                    axis=2,
                                )  # Add a dimension for NS3

                ### S3_cross = < (I1 * Psi)_j3 x (|I2 * Psi_j2| * Psi_j3)^* >_pix
                ### S3P_cross = < (I2 * Psi)_j3 x (|I1 * Psi_j2| * Psi_j3)^* >_pix
                else:
                    if calc_var:
                        s3, vs3 = self._compute_S3(
                            j2,
                            j3,
                            conv1,
                            vmask,
                            M2_dic,
                            M2convPsi_dic,
                            calc_var=True,
                            cmat2=cmat2,
                        )
                        s3p, vs3p = self._compute_S3(
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
                        s3 = self._compute_S3(
                            j2,
                            j3,
                            conv1,
                            vmask,
                            M2_dic,
                            M2convPsi_dic,
                            return_data=return_data,
                            cmat2=cmat2,
                        )
                        s3p = self._compute_S3(
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
                        if S3[j3] is None:
                            S3[j3] = {}
                            S3P[j3] = {}
                        if out_nside is not None and out_nside<nside_j2:
                            s3 = self.backend.bk_reduce_mean(
                                self.backend.bk_reshape(s3,[s3.shape[0],
                                                            12*out_nside**2,
                                                            (nside_j2//out_nside)**2,
                                                            s3.shape[2],
                                                            s3.shape[3]]),2)
                            s3p = self.backend.bk_reduce_mean(
                                self.backend.bk_reshape(s3p,[s3.shape[0],
                                                             12*out_nside**2,
                                                             (nside_j2//out_nside)**2,
                                                             s3.shape[2],
                                                             s3.shape[3]]),2)
                        S3[j3][j2] = s3
                        S3P[j3][j2] = s3p
                    else:
                        ### Normalize S3 and S3P with S2_j [Nbatch, Nmask, Norient_j]
                        if norm is not None:
                            self.div_norm(
                                s3,
                                (
                                    self.backend.bk_expand_dims(P2_dic[j2], off_S2)
                                    * self.backend.bk_expand_dims(P1_dic[j3], -1)
                                )
                                ** 0.5,
                            )  # [Nbatch, Nmask, Norient3, Norient2]
                            self.div_norm(
                                s3p,
                                (
                                    self.backend.bk_expand_dims(P1_dic[j2], off_S2)
                                    * self.backend.bk_expand_dims(P2_dic[j3], -1)
                                )
                                ** 0.5,
                            )  # [Nbatch, Nmask, Norient3, Norient2]

                        ### Store S3 and S3P as a complex [Nbatch, Nmask, NS3, Norient3, Norient2]
                        if S3 is None:
                            S3 = self.backend.bk_expand_dims(
                                s3, off_S3
                            )  # Add a dimension for NS3
                            if calc_var:
                                VS3 = self.backend.bk_expand_dims(
                                    vs3, off_S3
                                )  # Add a dimension for NS3
                        else:
                            S3 = self.backend.bk_concat(
                                [S3, self.backend.bk_expand_dims(s3, off_S3)], axis=2
                            )  # Add a dimension for NS3
                            if calc_var:
                                VS3 = self.backend.bk_concat(
                                    [VS3, self.backend.bk_expand_dims(vs3, off_S3)],
                                    axis=2,
                                )  # Add a dimension for NS3
                        if S3P is None:
                            S3P = self.backend.bk_expand_dims(
                                s3p, off_S3
                            )  # Add a dimension for NS3
                            if calc_var:
                                VS3P = self.backend.bk_expand_dims(
                                    vs3p, off_S3
                                )  # Add a dimension for NS3
                        else:
                            S3P = self.backend.bk_concat(
                                [S3P, self.backend.bk_expand_dims(s3p, off_S3)], axis=2
                            )  # Add a dimension for NS3
                            if calc_var:
                                VS3P = self.backend.bk_concat(
                                    [VS3P, self.backend.bk_expand_dims(vs3p, off_S3)],
                                    axis=2,
                                )  # Add a dimension for NS3

                ##### S4
                nside_j1=nside_j2
                for j1 in range(0, j2 + 1):  # j1 <= j2
                    ### S4_auto = <(|I1 * psi1| * psi3)(|I1 * psi2| * psi3)^*>
                    if not cross:
                        if calc_var:
                            s4, vs4 = self._compute_S4(
                                j1,
                                j2,
                                vmask,
                                M1convPsi_dic,
                                M2convPsi_dic=None,
                                calc_var=True,
                            )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        else:
                            s4 = self._compute_S4(
                                j1,
                                j2,
                                vmask,
                                M1convPsi_dic,
                                M2convPsi_dic=None,
                                return_data=return_data,
                            )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]

                        if return_data:
                            if S4[j3][j2] is None:
                                S4[j3][j2] = {}
                            if out_nside is not None and out_nside<nside_j1:
                                s4 = self.backend.bk_reduce_mean(
                                    self.backend.bk_reshape(s4,[s4.shape[0],
                                                                12*out_nside**2,
                                                                (nside_j1//out_nside)**2,
                                                                s4.shape[2],
                                                                s4.shape[3],
                                                                s4.shape[4]]),2)
                            S4[j3][j2][j1] = s4
                        else:
                            ### Normalize S4 with S2_j [Nbatch, Nmask, Norient_j]
                            if norm is not None:
                                self.div_norm(
                                    s4,
                                    (
                                        self.backend.bk_expand_dims(
                                            self.backend.bk_expand_dims(
                                                P1_dic[j1], off_S2
                                            ),
                                            off_S2,
                                        )
                                        * self.backend.bk_expand_dims(
                                            self.backend.bk_expand_dims(
                                                P1_dic[j2], off_S2
                                            ),
                                            -1,
                                        )
                                    )
                                    ** 0.5,
                                )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                            ### Store S4 as a complex [Nbatch, Nmask, NS4, Norient3, Norient2, Norient1]
                            if S4 is None:
                                S4 = self.backend.bk_expand_dims(
                                    s4, off_S4
                                )  # Add a dimension for NS4
                                if calc_var:
                                    VS4 = self.backend.bk_expand_dims(
                                        vs4, off_S4
                                    )  # Add a dimension for NS4
                            else:
                                S4 = self.backend.bk_concat(
                                    [S4, self.backend.bk_expand_dims(s4, off_S4)],
                                    axis=2,
                                )  # Add a dimension for NS4
                                if calc_var:
                                    VS4 = self.backend.bk_concat(
                                        [
                                            VS4,
                                            self.backend.bk_expand_dims(vs4, off_S4),
                                        ],
                                        axis=2,
                                    )  # Add a dimension for NS4

                        ### S4_cross = <(|I1 * psi1| * psi3)(|I2 * psi2| * psi3)^*>
                    else:
                        if calc_var:
                            s4, vs4 = self._compute_S4(
                                j1,
                                j2,
                                vmask,
                                M1convPsi_dic,
                                M2convPsi_dic=M2convPsi_dic,
                                calc_var=True,
                            )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        else:
                            s4 = self._compute_S4(
                                j1,
                                j2,
                                vmask,
                                M1convPsi_dic,
                                M2convPsi_dic=M2convPsi_dic,
                                return_data=return_data,
                            )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]

                        if return_data:
                            if S4[j3][j2] is None:
                                S4[j3][j2] = {}
                            if out_nside is not None and out_nside<nside_j1:
                                s4 = self.backend.bk_reduce_mean(
                                    self.backend.bk_reshape(s4,[s4.shape[0],
                                                                12*out_nside**2,
                                                                (nside_j1//out_nside)**2,
                                                                s4.shape[2],
                                                                s4.shape[3],
                                                                s4.shape[4]]),2)
                            S4[j3][j2][j1] = s4
                        else:
                            ### Normalize S4 with S2_j [Nbatch, Nmask, Norient_j]
                            if norm is not None:
                                self.div_norm(
                                    s4,
                                    (
                                        self.backend.bk_expand_dims(
                                            self.backend.bk_expand_dims(
                                                P1_dic[j1], off_S2
                                            ),
                                            off_S2,
                                        )
                                        * self.backend.bk_expand_dims(
                                            self.backend.bk_expand_dims(
                                                P2_dic[j2], off_S2
                                            ),
                                            -1,
                                        )
                                    )
                                    ** 0.5,
                                )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                            ### Store S4 as a complex [Nbatch, Nmask, NS4, Norient3, Norient2, Norient1]
                            if S4 is None:
                                S4 = self.backend.bk_expand_dims(
                                    s4, off_S4
                                )  # Add a dimension for NS4
                                if calc_var:
                                    VS4 = self.backend.bk_expand_dims(
                                        vs4, off_S4
                                    )  # Add a dimension for NS4
                            else:
                                S4 = self.backend.bk_concat(
                                    [S4, self.backend.bk_expand_dims(s4, off_S4)],
                                    axis=2,
                                )  # Add a dimension for NS4
                                if calc_var:
                                    VS4 = self.backend.bk_concat(
                                        [
                                            VS4,
                                            self.backend.bk_expand_dims(vs4, off_S4),
                                        ],
                                        axis=2,
                                    )  # Add a dimension for NS4
                            nside_j1=nside_j1 // 2
                        nside_j2=nside_j2 // 2
                        
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
                    s0, S2, S3, S4, s1=S1, backend=self.backend, use_1D=self.use_1D
                ), scat_cov(
                    vs0,
                    VS2,
                    VS3,
                    VS4,
                    s1=VS1,
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
            else:
                return scat_cov(
                    s0,
                    S2,
                    S3,
                    S4,
                    s1=S1,
                    s3p=S3P,
                    backend=self.backend,
                    use_1D=self.use_1D,
                ), scat_cov(
                    vs0,
                    VS2,
                    VS3,
                    VS4,
                    s1=VS1,
                    s3p=VS3P,
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
        else:
            if not cross:
                return scat_cov(
                    s0, S2, S3, S4, s1=S1, backend=self.backend, use_1D=self.use_1D
                )
            else:
                return scat_cov(
                    s0,
                    S2,
                    S3,
                    S4,
                    s1=S1,
                    s3p=S3P,
                    backend=self.backend,
                    use_1D=self.use_1D,
                )

    def clean_norm(self):
        self.P1_dic = None
        self.P2_dic = None
        return

    def _compute_S3(
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
        Compute the S3 coefficients (auto or cross)
        S3 = < (Ia * Psi)_j3 x (|Ib * Psi_j2| * Psi_j3)^* >_pix
        Parameters
        ----------
        Returns
        -------
        cs3, ss3: real and imag parts of S3 coeff
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
                    cmat2[j3][j2] * tmp2, [tmp2.shape[0], cmat2[j3].shape[1], 4, 4, 4]
                ),
                3,
            )

        # Store it so we can use it in S4 computation
        MconvPsi_dic[j2] = MconvPsi  # [Nbatch, Npix_j3, Norient3, Norient2]

        ### Compute the product (I2 * Psi)_j3 x (M1_j2 * Psi_j3)^*
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        # cconv, sconv are [Nbatch, Npix_j3, Norient3]
        if self.use_1D:
            s3 = conv * self.backend.bk_conjugate(MconvPsi)
        else:
            s3 = self.backend.bk_expand_dims(conv, -1) * self.backend.bk_conjugate(
                MconvPsi
            )  # [Nbatch, Npix_j3, Norient3, Norient2]

        ### Apply the mask [Nmask, Npix_j3] and sum over pixels
        if return_data:
            return s3
        else:
            if calc_var:
                s3, vs3 = self.masked_mean(
                    s3, vmask, axis=1, rank=j2, calc_var=True
                )  # [Nbatch, Nmask, Norient3, Norient2]
                return s3, vs3
            else:
                s3 = self.masked_mean(
                    s3, vmask, axis=1, rank=j2
                )  # [Nbatch, Nmask, Norient3, Norient2]
            return s3

    def _compute_S4(
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
            s4 = M1 * self.backend.bk_conjugate(M2)
        else:
            s4 = self.backend.bk_expand_dims(M1, -2) * self.backend.bk_conjugate(
                self.backend.bk_expand_dims(M2, -1)
            )  # [Nbatch, Npix_j3, Norient3, Norient2, Norient1]

        ### Apply the mask and sum over pixels
        if return_data:
            return s4
        else:
            if calc_var:
                s4, vs4 = self.masked_mean(
                    s4, vmask, axis=1, rank=j2, calc_var=True
                )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                return s4, vs4
            else:
                s4 = self.masked_mean(
                    s4, vmask, axis=1, rank=j2
                )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                return s4

    def square(self, x):
        if isinstance(x, scat_cov):
            if x.S1 is None:
                return scat_cov(
                    self.backend.bk_square(self.backend.bk_abs(x.S0)),
                    self.backend.bk_square(self.backend.bk_abs(x.S2)),
                    self.backend.bk_square(self.backend.bk_abs(x.S3)),
                    self.backend.bk_square(self.backend.bk_abs(x.S4)),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
            else:
                return scat_cov(
                    self.backend.bk_square(self.backend.bk_abs(x.S0)),
                    self.backend.bk_square(self.backend.bk_abs(x.S2)),
                    self.backend.bk_square(self.backend.bk_abs(x.S3)),
                    self.backend.bk_square(self.backend.bk_abs(x.S4)),
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
                    self.backend.bk_sqrt(self.backend.bk_abs(x.S2)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.S3)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.S4)),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
            else:
                return scat_cov(
                    self.backend.bk_sqrt(self.backend.bk_abs(x.S0)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.S2)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.S3)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.S4)),
                    s1=self.backend.bk_sqrt(self.backend.bk_abs(x.S1)),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
        else:
            return self.backend.bk_abs(self.backend.bk_sqrt(x))

    def reduce_mean(self, x):
        
        if isinstance(x, scat_cov):
            result = self.backend.bk_reduce_sum(self.backend.bk_abs(x.S0)) + \
                     self.backend.bk_reduce_sum(self.backend.bk_abs(x.S2)) + \
                     self.backend.bk_reduce_sum(self.backend.bk_abs(x.S3)) + \
                     self.backend.bk_reduce_sum(self.backend.bk_abs(x.S4))
                
            N = self.backend.bk_size(x.S0)+self.backend.bk_size(x.S2)+ \
                self.backend.bk_size(x.S3)+self.backend.bk_size(x.S4)
                
            if x.S1 is not None:
                result = result+self.backend.bk_reduce_sum(self.backend.bk_abs(x.S1))
                N = N + self.backend.bk_size(x.S1)
            if x.S3P is not None:
                result = result+self.backend.bk_reduce_sum(self.backend.bk_abs(x.S3P))
                N = N + self.backend.bk_size(x.S3P)
            return result/self.backend.bk_cast(N)
        else:
            return self.backend.bk_reduce_mean(x, axis=0)
                

    def reduce_mean_batch(self, x):
        
        if isinstance(x, scat_cov):
            
            sS0=self.backend.bk_reduce_mean(x.S0, axis=0)
            sS2=self.backend.bk_reduce_mean(x.S2, axis=0)
            sS3=self.backend.bk_reduce_mean(x.S3, axis=0)
            sS4=self.backend.bk_reduce_mean(x.S4, axis=0)
            sS1=None
            sS3P=None
            if x.S1 is not None:
                sS1 = self.backend.bk_reduce_mean(x.S1, axis=0)
            if x.S3P is not None:
                sS3P = self.backend.bk_reduce_mean(x.S3P, axis=0)
                
            result = scat_cov(
                sS0,
                sS2,
                sS3,
                sS4,
                s1=sS1,
                s3p=sS3P,
                backend=self.backend,
                use_1D=self.use_1D,
            )
            return result
        else:
            return self.backend.bk_reduce_mean(x, axis=0)
    
    def reduce_sum_batch(self, x):
        
        if isinstance(x, scat_cov):
            
            sS0=self.backend.bk_reduce_sum(x.S0, axis=0)
            sS2=self.backend.bk_reduce_sum(x.S2, axis=0)
            sS3=self.backend.bk_reduce_sum(x.S3, axis=0)
            sS4=self.backend.bk_reduce_sum(x.S4, axis=0)
            sS1=None
            sS3P=None
            if x.S1 is not None:
                sS1 = self.backend.bk_reduce_sum(x.S1, axis=0)
            if x.S3P is not None:
                sS3P = self.backend.bk_reduce_sum(x.S3P, axis=0)
                
            result = scat_cov(
                sS0,
                sS2,
                sS3,
                sS4,
                s1=sS1,
                s3p=sS3P,
                backend=self.backend,
                use_1D=self.use_1D,
            )
            return result
        else:
            return self.backend.bk_reduce_mean(x, axis=0)
    
    def reduce_distance(self, x, y, sigma=None):

        if isinstance(x, scat_cov):
            if sigma is None:
                result = self.diff_data(y.S0, x.S0, is_complex=False)
                if x.S1 is not None:
                    result += self.diff_data(y.S1, x.S1)
                if x.S3P is not None:
                    result += self.diff_data(y.S3P, x.S3P)
                result += self.diff_data(y.S2, x.S2)
                result += self.diff_data(y.S3, x.S3)
                result += self.diff_data(y.S4, x.S4)
            else:
                result = self.diff_data(y.S0, x.S0, is_complex=False, sigma=sigma.S0)
                if x.S1 is not None:
                    result += self.diff_data(y.S1, x.S1, sigma=sigma.S1)
                if x.S3P is not None:
                    result += self.diff_data(y.S3P, x.S3P, sigma=sigma.S3P)
                result += self.diff_data(y.S2, x.S2, sigma=sigma.S2)
                result += self.diff_data(y.S3, x.S3, sigma=sigma.S3)
                result += self.diff_data(y.S4, x.S4, sigma=sigma.S4)
            nval = (
                self.backend.bk_size(x.S0)
                + self.backend.bk_size(x.S2)
                + self.backend.bk_size(x.S3)
                + self.backend.bk_size(x.S4)
            )
            if x.S1 is not None:
                nval += self.backend.bk_size(x.S1)
            if x.S3P is not None:
                nval += self.backend.bk_size(x.S3P)
            result /= self.backend.bk_cast(nval)
            return result
        else:
            if sigma is None:
                tmp=x-y
            else:
                tmp=(x-y)/sigma
            # do abs in case of complex values
            return self.backend.bk_abs(self.backend.bk_reduce_mean(self.backend.bk_square(tmp)))

    def reduce_sum(self, x):

        if isinstance(x, scat_cov):
            if x.S1 is None:
                result = (
                    self.backend.bk_reduce_sum(x.S0)
                    + self.backend.bk_reduce_sum(x.S2)
                    + self.backend.bk_reduce_sum(x.S3)
                    + self.backend.bk_reduce_sum(x.S4)
                )
            else:
                result = (
                    self.backend.bk_reduce_sum(x.S0)
                    + self.backend.bk_reduce_sum(x.S2)
                    + self.backend.bk_reduce_sum(x.S1)
                    + self.backend.bk_reduce_sum(x.S3)
                    + self.backend.bk_reduce_sum(x.S4)
                )
        else:
            return self.backend.bk_reduce_sum(x)
        return result

    def ldiff(self, sig, x):

        if x.S1 is None:
            if x.S3P is not None:
                return scat_cov(
                    x.domult(sig.S0, x.S0) * x.domult(sig.S0, x.S0),
                    x.domult(sig.S2, x.S2) * x.domult(sig.S2, x.S2),
                    x.domult(sig.S3, x.S3) * x.domult(sig.S3, x.S3),
                    x.domult(sig.S4, x.S4) * x.domult(sig.S4, x.S4),
                    S3P=x.domult(sig.S3P, x.S3P) * x.domult(sig.S3P, x.S3P),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
            else:
                return scat_cov(
                    x.domult(sig.S0, x.S0) * x.domult(sig.S0, x.S0),
                    x.domult(sig.S2, x.S2) * x.domult(sig.S2, x.S2),
                    x.domult(sig.S3, x.S3) * x.domult(sig.S3, x.S3),
                    x.domult(sig.S4, x.S4) * x.domult(sig.S4, x.S4),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
        else:
            if x.S3P is None:
                return scat_cov(
                    x.domult(sig.S0, x.S0) * x.domult(sig.S0, x.S0),
                    x.domult(sig.S2, x.S2) * x.domult(sig.S2, x.S2),
                    x.domult(sig.S3, x.S3) * x.domult(sig.S3, x.S3),
                    x.domult(sig.S4, x.S4) * x.domult(sig.S4, x.S4),
                    S1=x.domult(sig.S1, x.S1) * x.domult(sig.S1, x.S1),
                    S3P=x.domult(sig.S3P, x.S3P) * x.domult(sig.S3P, x.S3P),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )
            else:
                return scat_cov(
                    x.domult(sig.S0, x.S0) * x.domult(sig.S0, x.S0),
                    x.domult(sig.S2, x.S2) * x.domult(sig.S2, x.S2),
                    x.domult(sig.S3, x.S3) * x.domult(sig.S3, x.S3),
                    x.domult(sig.S4, x.S4) * x.domult(sig.S4, x.S4),
                    S1=x.domult(sig.S1, x.S1) * x.domult(sig.S1, x.S1),
                    backend=self.backend,
                    use_1D=self.use_1D,
                )

    def log(self, x):
        if isinstance(x, scat_cov):

            if x.S1 is None:
                result = (
                    self.backend.bk_log(x.S0)
                    + self.backend.bk_log(x.S2)
                    + self.backend.bk_log(x.S3)
                    + self.backend.bk_log(x.S4)
                )
            else:
                result = (
                    self.backend.bk_log(x.S0)
                    + self.backend.bk_log(x.S2)
                    + self.backend.bk_log(x.S1)
                    + self.backend.bk_log(x.S3)
                    + self.backend.bk_log(x.S4)
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
    #            if res.S3P is not None:
    #                return scat_cov(
    #                    res.domult(sig.S0, res.S0) * res.domult(sig.S0, res.S0),
    #                    res.domult(sig.S2, res.S2) * res.domult(sig.S2, res.S2),
    #                    res.domult(sig.S3, res.S3) * res.domult(sig.S3, res.S3),
    #                    res.domult(sig.S4, res.S4) * res.domult(sig.S4, res.S4),
    #                    S3P=res.domult(sig.S3P, res.S3P) * res.domult(sig.S3P, res.S3P),
    #                    backend=self.backend,
    #                    use_1D=self.use_1D,
    #                )
    #            else:
    #                return scat_cov(
    #                    res.domult(sig.S0, res.S0) * res.domult(sig.S0, res.S0),
    #                    res.domult(sig.S2, res.S2) * res.domult(sig.S2, res.S2),
    #                    res.domult(sig.S3, res.S3) * res.domult(sig.S3, res.S3),
    #                    res.domult(sig.S4, res.S4) * res.domult(sig.S4, res.S4),
    #                    backend=self.backend,
    #                    use_1D=self.use_1D,
    #                )
    #        else:
    #            if res.S3P is None:
    #                return scat_cov(
    #                    res.domult(sig.S0, res.S0) * res.domult(sig.S0, res.S0),
    #                    res.domult(sig.S2, res.S2) * res.domult(sig.S2, res.S2),
    #                    res.domult(sig.S3, res.S3) * res.domult(sig.S3, res.S3),
    #                    res.domult(sig.S4, res.S4) * res.domult(sig.S4, res.S4),
    #                    S1=res.domult(sig.S1, res.S1) * res.domult(sig.S1, res.S1),
    #                    S3P=res.domult(sig.S3P, res.S3P) * res.domult(sig.S3P, res.S3P),
    #                    backend=self.backend,
    #                )
    #            else:
    #                return scat_cov(
    #                    res.domult(sig.S2, res.S2) * res.domult(sig.S2, res.S2),
    #                    res.domult(sig.S1, res.S1) * res.domult(sig.S1, res.S1),
    #                    res.domult(sig.S3, res.S3) * res.domult(sig.S3, res.S3),
    #                    res.domult(sig.S4, res.S4) * res.domult(sig.S4, res.S4),
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
        return res.S0, res.S2, res.S1, res.S3, res.S4, res.S3P

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
        s0, s2, s1, s3, s4, s3p = self.eval_comp_fast(
            image1, image2=image2, mask=mask, Auto=Auto, cmat=cmat, cmat2=cmat2
        )
        return scat_cov(
            s0, s2, s3, s4, s1=s1, s3p=s3p, backend=self.backend, use_1D=self.use_1D
        )
