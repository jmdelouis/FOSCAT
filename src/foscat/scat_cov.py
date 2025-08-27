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
    def __init__(self,
                 s0, s2, s3, s4,
                 s1=None,
                 s3p=None,
                 backend=None,
                 use_1D=False,
                 return_data=False
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
        if not return_data:
            self.numel = self.backend.bk_len(s0)+ \
                self.backend.bk_len(s1)+ \
                self.backend.bk_len(s2)+ \
                self.backend.bk_len(s3)+ \
                self.backend.bk_len(s4)+ \
                self.backend.bk_len(s3p)

    def numpy(self):
        if self.BACKEND == "numpy":
            return self

        if self.S1 is None:
            s1 = None
        else:
            s1 = self.backend.to_numpy(self.S1)
        if self.S3P is None:
            s3p = None
        else:
            s3p = self.backend.to_numpy(self.S3P)

        return scat_cov(
            self.backend.to_numpy(self.S0),
            self.backend.to_numpy(self.S2),
            self.backend.to_numpy(self.S3),
            self.backend.to_numpy(self.S4),
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
        if self.backend.bk_is_complex(val):
            return val
        else:
            return self.backend.bk_complex(val, 0 * val)
        return val

    # ---------------------------------------------−---------
    def flatten(self):
        if self.use_1D:
            tmp = [
                self.conv2complex(
                    self.backend.bk_reshape(self.S0, [self.S1.shape[0], self.S0.shape[1]])
                )
            ]
            
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

        tmp = [
            self.conv2complex(
                self.backend.bk_reshape(self.S0, [self.S1.shape[0], self.S0.shape[1]*self.S0.shape[2]])
            )
        ]
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
                        self.S4.shape[0],
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
                    s1 = self.doadd(self.S1, other.S1)
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
                self.doadd(self.S3, other.S3),
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
                s1 = self.dodiv(other.S1, self.S1)
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
                self.dodiv(other.S3, self.S3),
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
                    s1 = self.domin(other.S1, self.S1)
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
                self.domin(other.S3, self.S3),
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
                    s1 = self.domin(self.S1, other.S1)
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
                self.domin(self.S3, other.S3),
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
                    s1 = self.domult(self.S1, other.S1)
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
                return self.backend.to_numpy(x)
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
                abs(self.get_np(self.S0)).sum()
                + abs(self.get_np(self.S1)).sum()
                + abs(self.get_np(self.S3)).sum()
                + abs(self.get_np(self.S4)).sum()
                + abs(self.get_np(self.S2)).sum()
            ) / self.numel
        else:  # Cross
            return (
                abs(self.get_np(self.S0)).sum()
                + abs(self.get_np(self.S3)).sum()
                + abs(self.get_np(self.S3P)).sum()
                + abs(self.get_np(self.S4)).sum()
                + abs(self.get_np(self.S2)).sum()
            ) / self.numel

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
            s2[:, :, noff:, :] = self.backend.to_numpy(self.S2)
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
            s1[:, :, noff:, :] = self.backend.to_numpy(self.S1)
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
                                s3[i, j, idx[noff:], k, l_orient] = (
                                    self.backend.to_numpy(self.S3)[
                                        i, j, j2 == ij - noff, k, l_orient
                                    ]
                                )
                                s3[i, j, idx[:noff], k, l_orient] = (
                                    self.add_data_from_slope(
                                        self.backend.to_numpy(self.S3)[
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
                    nS0 = np.expand_dims(self.backend.to_numpy(tmp.S0), 0)
                    nS2 = np.expand_dims(self.backend.to_numpy(tmp.S2), 0)
                    nS3 = np.expand_dims(self.backend.to_numpy(tmp.S3), 0)
                    nS4 = np.expand_dims(self.backend.to_numpy(tmp.S4), 0)
                    if tmp.S3P is not None:
                        nS3P = np.expand_dims(self.backend.to_numpy(tmp.S3P), 0)
                    if tmp.S1 is not None:
                        nS1 = np.expand_dims(self.backend.to_numpy(tmp.S1), 0)

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
            sS0 = self.backend.bk_cast(np.std(S0, 0))
            sS2 = self.backend.bk_cast(np.std(S2, 0))
            sS3 = self.backend.bk_cast(np.std(S3, 0))
            sS4 = self.backend.bk_cast(np.std(S4, 0))
            mS0 = self.backend.bk_cast(np.mean(S0, 0))
            mS2 = self.backend.bk_cast(np.mean(S2, 0))
            mS3 = self.backend.bk_cast(np.mean(S3, 0))
            mS4 = self.backend.bk_cast(np.mean(S4, 0))
            if tmp.S3P is not None:
                sS3P = self.backend.bk_cast(np.std(S3P, 0))
                mS3P = self.backend.bk_cast(np.mean(S3P, 0))
            else:
                sS3P = None
                mS3P = None

            if tmp.S1 is not None:
                sS1 = self.backend.bk_cast(np.std(S1, 0))
                mS1 = self.backend.bk_cast(np.mean(S1, 0))
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
    def stat_cfft(self, im, image2=None, upscale=False, smooth_scale=0,spin=0):
        tmp = im
        if image2 is not None:
            tmpi2 = image2
        if upscale:
            l_nside = int(np.sqrt(tmp.shape[-1] // 12))
            tmp = self.up_grade(tmp, l_nside * 2)
            if image2 is not None:
                tmpi2 = self.up_grade(tmpi2, l_nside * 2)
        l_nside = int(np.sqrt(tmp.shape[-1] // 12))
        nscale = int(np.log(l_nside) / np.log(2))
        cmat = {}
        cmat2 = {}

        # Loop over scales
        for k in range(nscale):
            if image2 is not None:
                sim = self.backend.bk_real(
                    self.backend.bk_L1(
                        self.convol(tmp,spin=spin)
                        * self.backend.bk_conjugate(self.convol(tmpi2,spin=spin))
                    )
                )
            else:
                sim = self.backend.bk_abs(self.convol(tmp,spin=spin))

            # instead of difference between "opposite" channels use weighted average
            # of cosine and sine contributions using all channels
            if spin==0:
                angles = self.backend.bk_cast(
                    (2 * np.pi * np.arange(self.NORIENT)
                     / self.NORIENT).reshape(1,self.NORIENT,1)) # shape: (NORIENT,)
            else:
                angles = self.backend.bk_cast(
                    (2 * np.pi * np.arange(self.NORIENT)
                     / self.NORIENT).reshape(1,1,self.NORIENT,1)) # shape: (NORIENT,)

            # we use cosines and sines as weights for sim
            weighted_cos = self.backend.bk_reduce_mean(
                sim * self.backend.bk_cos(angles), axis=-2
            )
            weighted_sin = self.backend.bk_reduce_mean(
                sim * self.backend.bk_sin(angles), axis=-2
            )
            # For simplicity, take first element of the batch
            cc = weighted_cos[0]
            ss = weighted_sin[0]

            if smooth_scale > 0:
                for m in range(smooth_scale):
                    if cc.shape[0] > 12:
                        cc, _ = self.ud_grade_2(self.smooth(cc))
                        ss, _ = self.ud_grade_2(self.smooth(ss))

            if cc.shape[-1] != tmp.shape[-1]:
                ll_nside = int(np.sqrt(tmp.shape[-1] // 12))
                cc = self.up_grade(cc, ll_nside)
                ss = self.up_grade(ss, ll_nside)

            # compute local phase from weighted cos and sin (same as before)
            if self.BACKEND == "numpy":
                phase = np.fmod(np.arctan2(ss, cc) + 2 * np.pi, 2 * np.pi)
            else:
                phase = np.fmod(
                    np.arctan2(self.backend.to_numpy(ss), self.backend.to_numpy(cc))
                    + 2 * np.pi,
                    2 * np.pi,
                )

            # instead of linear interpolation cosine‐based interpolation
            phase_scaled = self.NORIENT * phase / (2 * np.pi)
            iph = np.floor(phase_scaled).astype("int")  # lower bin index
            delta = phase_scaled - iph  # fractional part in [0,1)
            # interpolation weights
            w0 = np.cos(delta * np.pi / 2) ** 2
            w1 = np.sin(delta * np.pi / 2) ** 2

            # build rotation matrix
            if spin==0:
                mat = np.zeros([self.NORIENT * self.NORIENT, sim.shape[-1]])
            else:
                mat = np.zeros([2,self.NORIENT * self.NORIENT, sim.shape[-1]])
            lidx = np.arange(sim.shape[-1])
            for ell in range(self.NORIENT):
                # Instead of simple linear weights, we use the cosine weights w0 and w1.
                col0 = self.NORIENT * ((ell + iph) % self.NORIENT) + ell
                col1 = self.NORIENT * ((ell + iph + 1) % self.NORIENT) + ell

                if spin==0:
                    mat[col0, lidx] = w0
                    mat[col1, lidx] = w1
                else:
                    mat[0,col0, lidx] = w0[0]
                    mat[0,col1, lidx] = w1[0]
                    mat[1,col0, lidx] = w0[1]
                    mat[1,col1, lidx] = w1[1]

            cmat[k] = self.backend.bk_cast(mat[None, ...].astype("complex64"))

            # do same modifications for mat2
            if spin==0:
                mat2 = np.zeros(
                    [k + 1, self.NORIENT * self.NORIENT, self.NORIENT, sim.shape[-1]]
                )
            else:
                mat2 = np.zeros(
                    [k + 1, 2, self.NORIENT * self.NORIENT, self.NORIENT, sim.shape[-1]]
                )

            for k2 in range(k + 1):

                tmp2 = self.backend.bk_expand_dims(sim,-2)
                if spin==0:
                    sim2 = self.backend.bk_reduce_sum(
                        self.backend.bk_reshape(
                            self.backend.bk_cast(
                                mat.reshape(1, self.NORIENT, self.NORIENT, mat.shape[-1])
                            )
                            * tmp2,
                            [sim.shape[0], self.NORIENT, self.NORIENT, mat.shape[-1]],
                        ),
                        1,
                    )
                else:
                    sim2 = self.backend.bk_reduce_sum(
                        self.backend.bk_reshape(
                            self.backend.bk_cast(
                                mat.reshape(1, 2, self.NORIENT, self.NORIENT, mat.shape[-1])
                            )
                            * tmp2,
                            [sim.shape[0], 2, self.NORIENT, self.NORIENT, mat.shape[-1]],
                        ),
                        2,
                    )

                sim2 = self.backend.bk_abs(self.convol(sim2))
                angles = self.backend.bk_reshape(angles, [1, self.NORIENT, 1, 1])
                weighted_cos2 = self.backend.bk_reduce_mean(
                    sim2 * self.backend.bk_cos(angles), axis=-3
                )
                weighted_sin2 = self.backend.bk_reduce_mean(
                    sim2 * self.backend.bk_sin(angles), axis=-3
                )

                cc2 = weighted_cos2[0]
                ss2 = weighted_sin2[0]

                if smooth_scale > 0:
                    for m in range(smooth_scale):
                        if cc2.shape[1] > 12:
                            cc2, _ = self.ud_grade_2(self.smooth(cc2))
                            ss2, _ = self.ud_grade_2(self.smooth(ss2))

                if cc2.shape[-1] != sim.shape[-1]:
                    ll_nside = int(np.sqrt(sim.shape[-1] // 12))
                    cc2 = self.up_grade(cc2, ll_nside)
                    ss2 = self.up_grade(ss2, ll_nside)

                if self.BACKEND == "numpy":
                    phase2 = np.fmod(np.arctan2(ss2, cc2) + 2 * np.pi, 2 * np.pi)
                else:
                    phase2 = np.fmod(
                        np.arctan2(
                            self.backend.to_numpy(ss2), self.backend.to_numpy(cc2)
                        )
                        + 2 * np.pi,
                        2 * np.pi,
                    )

                phase2_scaled = self.NORIENT * phase2 / (2 * np.pi)
                iph2 = np.floor(phase2_scaled).astype("int")
                delta2 = phase2_scaled - iph2
                w0_2 = np.cos(delta2 * np.pi / 2) ** 2
                w1_2 = np.sin(delta2 * np.pi / 2) ** 2
                lidx = np.arange(sim.shape[-1])

                if spin==0:
                    for m in range(self.NORIENT):
                        for ell in range(self.NORIENT):
                            col0 = self.NORIENT * ((ell + iph2[m]) % self.NORIENT) + ell
                            col1 = self.NORIENT * ((ell + iph2[m] + 1) % self.NORIENT) + ell
                            mat2[k2, col0, m, lidx] = w0_2[m, lidx]
                            mat2[k2, col1, m, lidx] = w1_2[m, lidx]
                else:
                    for sidx in range(2):
                        for m in range(self.NORIENT):
                            for ell in range(self.NORIENT):
                                col0 = self.NORIENT * ((ell + iph2[sidx,m]) % self.NORIENT) + ell
                                col1 = self.NORIENT * ((ell + iph2[sidx,m] + 1) % self.NORIENT) + ell
                                mat2[k2, sidx, col0, m, lidx] = w0_2[sidx,m, lidx]
                                mat2[k2, sidx, col1, m, lidx] = w1_2[sidx,m, lidx]

                cmat2[k] = self.backend.bk_cast(
                    mat2[0 : k + 1, None, ...].astype("complex64")
                )

            if k < l_nside - 1:
                tmp, _ = self.ud_grade_2(tmp)
                if image2 is not None:
                    tmpi2, _ = self.ud_grade_2(tmpi)
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
        calc_var=False,
        cmat=None,
        cmat2=None,
        Jmax=None,
        out_nside=None,
        edge=True,
        nside=None,
        cell_ids=None,
        spin=0
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
        spin : Integer
            If different from 0 compute spinned data (U,V to Divergence/Rotational spin==1) or (Q,U to E,B spin=2).
            This implies that the input data is 2*12*nside^2.
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
            if self.use_2D:
                if (
                    image1.shape[-2] != mask.shape[-1]
                    or image1.shape[-1] != mask.shape[-1]
                ):
                    print(
                        "The LAST 2 COLUMNs of the mask should have the same size ",
                        mask.shape,
                        "than the input image ",
                        image1.shape,
                        "to eval Scattering Covariance",
                    )
                    return None
            else:
                if image1.shape[-1] != mask.shape[-1]:
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
        l_nside = 2**32  # not initialize if 1D or 2D
        ### PARAMETERS
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
            if J == 0:
                print("Use of too small 2D domain does not work J_max=", J)
                return None
        elif self.use_1D:
            if len(image1.shape) == 2:
                npix = int(im_shape[1])  # Number of pixels
            else:
                npix = int(im_shape[0])  # Number of pixels

            nside = int(npix)

            J = int(np.log(nside) / np.log(2))  # Number of j scales
        else:
            npix=int(im_shape[-1])

            if nside is None:
                nside = int(np.sqrt(npix // 12))

            J = int(np.log(nside) / np.log(2))  # Number of j scales

        if (self.use_2D or self.use_1D) and self.KERNELSZ > 3:
            J -= 1
        if Jmax is None:
            Jmax = J  # Number of steps for the loop on scales
        if Jmax > J:
            print("==========\n\n")
            print(
                "The Jmax you requested is larger than the data size, which may cause problems while computing the scattering transform."
            )
            print("\n\n==========")

        ### LOCAL VARIABLES (IMAGES and MASK)
        if len(image1.shape) == 1 or (len(image1.shape) == 2 and self.use_2D) or (len(image1.shape) == 2 and spin>0):
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

        if self.KERNELSZ > 3 and not self.use_2D and cell_ids is None:
            # if the kernel size is bigger than 3 increase the binning before smoothing
            if self.use_2D:
                vmask = self.up_grade(
                    vmask, I1.shape[-2] * 2, nouty=I1.shape[-1] * 2,axis=-2
                )
                I1 = self.up_grade(
                    I1, I1.shape[-2] * 2, nouty=I1.shape[-1] * 2,axis=-2
                )
                if cross:
                    I2 = self.up_grade(
                        I2, I2.shape[-2] * 2, nouty=I2.shape[-1] * 2,axis=-2
                    )
            elif self.use_1D:
                vmask = self.up_grade(vmask, I1.shape[-1] * 2)
                I1 = self.up_grade(I1, I1.shape[-1] * 2)
                if cross:
                    I2 = self.up_grade(I2, I2.shape[-1] * 2)
                nside = nside * 2
            else:
                I1 = self.up_grade(I1, nside * 2)
                vmask = self.up_grade(vmask, nside * 2)
                if cross:
                    I2 = self.up_grade(I2, nside * 2)

                nside = nside * 2

        # Normalize the masks because they have different pixel numbers
        # vmask /= self.backend.bk_reduce_sum(vmask, axis=1)[:, None]  # [Nmask, Npix]

        ### INITIALIZATION
        # Coefficients
        if return_data:
            S1 = {}
            S2 = {}
            S3 = {}
            S3P = {}
            S4 = {}
        else:
            S1 = []
            S2 = []
            S3 = []
            S4 = []
            S3P = []
            VS1 = []
            VS2 = []
            VS3 = []
            VS3P = []
            VS4 = []

        off_S2 = -2
        off_S3 = -3
        off_S4 = -4
            
        if self.use_1D:
            off_S2 = -1
            off_S3 = -1
            off_S4 = -1

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
                s0 = self.backend.bk_reduce_mean(
                    self.backend.bk_reshape(
                        s0, [s0.shape[0], 12 * out_nside**2, (nside // out_nside) ** 2]
                    ),
                    2,
                )
        else:
            if not cross:
                s0, l_vs0 = self.masked_mean(I1,
                                             vmask,
                                             calc_var=True)
            else:
                s0, l_vs0 = self.masked_mean(
                    self.backend.bk_L1(I1 * I2),
                    vmask,
                    calc_var=True)
                
            vs0 = self.backend.bk_concat([l_vs0, l_vs0], -1)
            s0 = self.backend.bk_concat([s0, l_vs0], -1)
            if spin>0:
                vs0=self.backend.bk_reshape(vs0,[vs0.shape[0],vs0.shape[1],2,vs0.shape[2]//2])
                s0=self.backend.bk_reshape(s0,[s0.shape[0],s0.shape[1],2,s0.shape[2]//2])
        #### COMPUTE S1, S2, S3 and S4
        nside_j3 = nside  # NSIDE start (nside_j3 = nside / 2^j3)

        # a remettre comme avant
        M1_dic = {}
        M2_dic = {}

        cell_ids_j3 = cell_ids

        for j3 in range(Jmax):

            if edge:
                if self.mask_mask is None:
                    self.mask_mask = {}
                if self.use_2D:
                    if (vmask.shape[1], vmask.shape[2]) not in self.mask_mask:
                        mask_mask = np.zeros([1, vmask.shape[1], vmask.shape[2]])
                        mask_mask[
                            0,
                            self.KERNELSZ // 2 : -self.KERNELSZ // 2 + 1,
                            self.KERNELSZ // 2 : -self.KERNELSZ // 2 + 1,
                        ] = 1.0
                        self.mask_mask[(vmask.shape[1], vmask.shape[2])] = (
                            self.backend.bk_cast(mask_mask)
                        )
                    vmask = vmask * self.mask_mask[(vmask.shape[1], vmask.shape[2])]
                    # print(self.KERNELSZ//2,vmask,mask_mask)

                if self.use_1D:
                    if (vmask.shape[1]) not in self.mask_mask:
                        mask_mask = np.zeros([1, vmask.shape[1]])
                        mask_mask[0, self.KERNELSZ // 2 : -self.KERNELSZ // 2 + 1] = 1.0
                        self.mask_mask[(vmask.shape[1])] = self.backend.bk_cast(
                            mask_mask
                        )
                    vmask = vmask * self.mask_mask[(vmask.shape[1])]

            if return_data:
                S3[j3] = None
                S3P[j3] = None

                if S4 is None:
                    S4 = {}
                S4[j3] = None

            ####### S1 and S2
            ### Make the convolution I1 * Psi_j3
            conv1 = self.convol(
                I1, cell_ids=cell_ids_j3, nside=nside_j3,
                spin=spin
            )  # [Nbatch, Norient3 , Npix_j3]

            if cmat is not None:
                tmp2 = self.backend.bk_repeat(conv1, self.NORIENT, axis=-2)
                
                if spin==0:
                    conv1 = self.backend.bk_reduce_sum(
                        self.backend.bk_reshape(
                            cmat[j3] * tmp2,
                            [tmp2.shape[0], self.NORIENT, self.NORIENT, cmat[j3].shape[2]],
                        ),
                        1,
                    )
                else:
                    conv1 = self.backend.bk_reduce_sum(
                        self.backend.bk_reshape(
                            cmat[j3] * tmp2,
                            [tmp2.shape[0], 2,self.NORIENT, self.NORIENT, cmat[j3].shape[3]],
                        ),
                        2,
                    )

            ### Take the module M1 = |I1 * Psi_j3|
            M1_square = conv1 * self.backend.bk_conjugate(
                conv1
            )  # [Nbatch, Norient3, Npix_j3]

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
                            M1_square, vmask, rank=j3, calc_var=True
                        )
                    else:
                        s2 = self.masked_mean(M1_square, vmask, rank=j3)
                        
                if cond_init_P1_dic:
                    # We fill P1_dic with S2 for normalisation of S3 and S4
                    P1_dic[j3] = self.backend.bk_real(s2)  # [Nbatch, Nmask, Norient3]

                # We store S2_auto to return it [Nbatch, Nmask, NS2, Norient3]
                if return_data:
                    if S2 is None:
                        S2 = {}
                    if out_nside is not None and out_nside < nside_j3:
                        s2 = self.backend.bk_reduce_mean(
                            self.backend.bk_reshape(
                                s2,
                                [
                                    s2.shape[0],
                                    s2.shape[2],
                                    12 * out_nside**2,
                                    (nside_j3 // out_nside) ** 2,
                                ],
                            ),
                            2,
                        )
                    S2[j3] = s2
                else:
                    if norm == "auto":  # Normalize S2
                        s2 /= P1_dic[j3]

                    S2.append(
                        self.backend.bk_expand_dims(s2, off_S2)
                    )  # Add a dimension for NS2
                    if calc_var:
                        VS2.append(
                            self.backend.bk_expand_dims(vs2, off_S2)
                        )  # Add a dimension for NS2

                #### S1_auto computation
                ### Image 1 : S1 = < M1 >_pix
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                if return_data:
                    s1 = M1
                else:
                    if calc_var:
                        s1, vs1 = self.masked_mean(
                            M1, vmask, rank=j3, calc_var=True
                        )  # [Nbatch, Nmask, Norient3]
                    else:
                        s1 = self.masked_mean(
                            M1, vmask, rank=j3
                        )  # [Nbatch, Nmask, Norient3]

                if return_data:
                    if out_nside is not None and out_nside < nside_j3:
                        s1 = self.backend.bk_reduce_mean(
                            self.backend.bk_reshape(
                                s1,
                                [
                                    s1.shape[0],
                                    s1.shape[2],
                                    12 * out_nside**2,
                                    (nside_j3 // out_nside) ** 2,
                                ],
                            ),
                            2,
                        )
                    S1[j3] = s1
                else:
                    ### Normalize S1
                    if norm is not None:
                        self.div_norm(s1, (P1_dic[j3]) ** 0.5)
                    ### We store S1 for image1  [Nbatch, Nmask, NS1, Norient3]
                    S1.append(
                        self.backend.bk_expand_dims(s1, off_S2)
                    )  # Add a dimension for NS1
                    if calc_var:
                        VS1.append(
                            self.backend.bk_expand_dims(vs1, off_S2)
                        )  # Add a dimension for NS1

            else:  # Cross
                ### Make the convolution I2 * Psi_j3
                conv2 = self.convol(
                    I2,  cell_ids=cell_ids_j3, nside=nside_j3,
                    spin=spin
                )  # [Nbatch, Npix_j3, Norient3]
                if cmat is not None:
                    tmp2 = self.backend.bk_repeat(conv2, self.NORIENT, axis=-2)
                    if spin==0:
                        conv2 = self.backend.bk_reduce_sum(
                            self.backend.bk_reshape(
                                cmat[j3] * tmp2,
                                [
                                    tmp2.shape[0],
                                    self.NORIENT,
                                    self.NORIENT,
                                    cmat[j3].shape[2],
                                ],
                            ),
                            1,
                        )
                    else:
                        conv2 = self.backend.bk_reduce_sum(
                            self.backend.bk_reshape(
                                cmat[j3] * tmp2,
                                [
                                    tmp2.shape[0],
                                    2,
                                    self.NORIENT,
                                    self.NORIENT,
                                    cmat[j3].shape[3],
                                ],
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
                                M1_square, vmask,  rank=j3, calc_var=True
                            )  # [Nbatch, Nmask, Norient3]
                            p2, vp2 = self.masked_mean(
                                M2_square, vmask,  rank=j3, calc_var=True
                            )  # [Nbatch, Nmask, Norient3]
                        else:
                            p1 = self.masked_mean(
                                M1_square, vmask, rank=j3
                            )  # [Nbatch, Nmask, Norient3]
                            p2 = self.masked_mean(
                                M2_square, vmask, rank=j3
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
                            s2, vmask, rank=j3, calc_var=True
                        )
                    else:
                        s2 = self.masked_mean(s2, vmask, rank=j3)

                if return_data:
                    if out_nside is not None and out_nside < nside_j3:
                        s2 = self.backend.bk_reduce_mean(
                            self.backend.bk_reshape(
                                s2,
                                [
                                    s2.shape[0],
                                    s2.shape[2],
                                    12 * out_nside**2,
                                    (nside_j3 // out_nside) ** 2,
                                ],
                            ),
                            2,
                        )
                    S2[j3] = s2
                else:

                    ### Store S2_cross as complex [Nbatch, Nmask, NS2, Norient3]
                    s2 = self.backend.bk_real(s2)

                    ### Normalize S2_cross
                    if norm == "auto":
                        s2 /= (P1_dic[j3] * P2_dic[j3]) ** 0.5

                    S2.append(
                        self.backend.bk_expand_dims(s2, off_S2)
                    )  # Add a dimension for NS2
                    if calc_var:
                        VS2.append(
                            self.backend.bk_expand_dims(vs2, off_S2)
                        )  # Add a dimension for NS2

                #### S1_auto computation
                ### Image 1 : S1 = < M1 >_pix
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                if return_data:
                    s1 = MX
                else:
                    if calc_var:
                        s1, vs1 = self.masked_mean(
                            MX, vmask, rank=j3, calc_var=True
                        )  # [Nbatch, Nmask, Norient3]
                    else:
                        s1 = self.masked_mean(
                            MX, vmask, rank=j3
                        )  # [Nbatch, Nmask, Norient3]
                if return_data:
                    if out_nside is not None and out_nside < nside_j3:
                        s1 = self.backend.bk_reduce_mean(
                            self.backend.bk_reshape(
                                s1,
                                [
                                    s1.shape[0],
                                    s1.shape[2],
                                    12 * out_nside**2,
                                    (nside_j3 // out_nside) ** 2,
                                ],
                            ),
                            2,
                        )
                    S1[j3] = s1
                else:
                    ### Normalize S1
                    if norm is not None:
                        self.div_norm(s1, (P1_dic[j3]) ** 0.5)
                    ### We store S1 for image1  [Nbatch, Nmask, NS1, Norient3]
                    S1.append(
                        self.backend.bk_expand_dims(s1, off_S2)
                    )  # Add a dimension for NS1
                    if calc_var:
                        VS1.append(
                            self.backend.bk_expand_dims(vs1, off_S2)
                        )  # Add a dimension for NS1

            # Initialize dictionaries for |I1*Psi_j| * Psi_j3
            M1convPsi_dic = {}
            if cross:
                # Initialize dictionaries for |I2*Psi_j| * Psi_j3
                M2convPsi_dic = {}

            ###### S3
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
                            cell_ids=cell_ids_j3,
                            nside_j2=nside_j3,
                            spin=spin,
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
                            cell_ids=cell_ids_j3,
                            nside_j2=nside_j3,
                            spin=spin,
                        )  # [Nbatch, Nmask, Norient3, Norient2]

                    if return_data:
                        if S3[j3] is None:
                            S3[j3] = {}
                        if out_nside is not None and out_nside < nside_j3:
                            s3 = self.backend.bk_reduce_mean(
                                self.backend.bk_reshape(
                                    s3,
                                    [
                                        s3.shape[0],
                                        12 * out_nside**2,
                                        (nside_j3 // out_nside) ** 2,
                                        s3.shape[2],
                                        s3.shape[3],
                                    ],
                                ),
                                2,
                            )
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

                        # S3.append(self.backend.bk_reshape(s3,[s3.shape[0],s3.shape[1],
                        #                                      s3.shape[2]*s3.shape[3]]))
                        S3.append(
                            self.backend.bk_expand_dims(s3, off_S3)
                        )  # Add a dimension for NS3
                        if calc_var:
                            VS3.append(
                                self.backend.bk_expand_dims(vs3, off_S3)
                            )  # Add a dimension for NS3
                            # VS3.append(self.backend.bk_reshape(vs3,[s3.shape[0],s3.shape[1],
                            #                                  s3.shape[2]*s3.shape[3]]))

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
                            cell_ids=cell_ids_j3,
                            nside_j2=nside_j3,
                            spin=spin,
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
                            cell_ids=cell_ids_j3,
                            nside_j2=nside_j3,
                            spin=spin,
                        )
                    else:
                        s3p = self._compute_S3(
                            j2,
                            j3,
                            conv2,
                            vmask,
                            M1_dic,
                            M1convPsi_dic,
                            return_data=return_data,
                            cmat2=cmat2,
                            cell_ids=cell_ids_j3,
                            nside_j2=nside_j3,
                            spin=spin,
                        )
                        s3 = self._compute_S3(
                            j2,
                            j3,
                            conv1,
                            vmask,
                            M2_dic,
                            M2convPsi_dic,
                            return_data=return_data,
                            cmat2=cmat2,
                            cell_ids=cell_ids_j3,
                            nside_j2=nside_j3,
                            spin=spin,
                        )

                    if return_data:
                        if S3[j3] is None:
                            S3[j3] = {}
                            S3P[j3] = {}
                        if out_nside is not None and out_nside < nside_j3:
                            s3 = self.backend.bk_reduce_mean(
                                self.backend.bk_reshape(
                                    s3,
                                    [
                                        s3.shape[0],
                                        12 * out_nside**2,
                                        (nside_j3 // out_nside) ** 2,
                                        s3.shape[2],
                                        s3.shape[3],
                                    ],
                                ),
                                2,
                            )
                            s3p = self.backend.bk_reduce_mean(
                                self.backend.bk_reshape(
                                    s3p,
                                    [
                                        s3.shape[0],
                                        12 * out_nside**2,
                                        (nside_j3 // out_nside) ** 2,
                                        s3.shape[2],
                                        s3.shape[3],
                                    ],
                                ),
                                2,
                            )
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

                        # S3.append(self.backend.bk_reshape(s3,[s3.shape[0],s3.shape[1],
                        #                                      s3.shape[2]*s3.shape[3]]))
                        S3.append(
                            self.backend.bk_expand_dims(s3, off_S3)
                        )  # Add a dimension for NS3
                        if calc_var:
                            VS3.append(
                                self.backend.bk_expand_dims(vs3, off_S3)
                            )  # Add a dimension for NS3

                            # VS3.append(self.backend.bk_reshape(vs3,[s3.shape[0],s3.shape[1],
                            #                                  s3.shape[2]*s3.shape[3]]))

                        # S3P.append(self.backend.bk_reshape(s3p,[s3.shape[0],s3.shape[1],
                        #                                      s3.shape[2]*s3.shape[3]]))
                        S3P.append(
                            self.backend.bk_expand_dims(s3p, off_S3)
                        )  # Add a dimension for NS3
                        if calc_var:
                            VS3P.append(
                                self.backend.bk_expand_dims(vs3p, off_S3)
                            )  # Add a dimension for NS3
                            # VS3P.append(self.backend.bk_reshape(vs3p,[s3.shape[0],s3.shape[1],
                            #                                  s3.shape[2]*s3.shape[3]]))

                ##### S4
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
                            if out_nside is not None and out_nside < nside_j3:
                                s4 = self.backend.bk_reduce_mean(
                                    self.backend.bk_reshape(
                                        s4,
                                        [
                                            s4.shape[0],
                                            12 * out_nside**2,
                                            (nside_j3 // out_nside) ** 2,
                                            s4.shape[2],
                                            s4.shape[3],
                                            s4.shape[4],
                                        ],
                                    ),
                                    2,
                                )
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

                            # S4.append(self.backend.bk_reshape(s4,[s4.shape[0],s4.shape[1],
                            #                                  s4.shape[2]*s4.shape[3]*s4.shape[4]]))
                            S4.append(
                                self.backend.bk_expand_dims(s4, off_S4)
                            )  # Add a dimension for NS4
                            if calc_var:
                                # VS4.append(self.backend.bk_reshape(vs4,[s4.shape[0],s4.shape[1],
                                #                              s4.shape[2]*s4.shape[3]*s4.shape[4]]))
                                VS4.append(
                                    self.backend.bk_expand_dims(vs4, off_S4)
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
                            if out_nside is not None and out_nside < nside_j3:
                                s4 = self.backend.bk_reduce_mean(
                                    self.backend.bk_reshape(
                                        s4,
                                        [
                                            s4.shape[0],
                                            12 * out_nside**2,
                                            (nside_j3 // out_nside) ** 2,
                                            s4.shape[2],
                                            s4.shape[3],
                                            s4.shape[4],
                                        ],
                                    ),
                                    2,
                                )
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
                            # S4.append(self.backend.bk_reshape(s4,[s4.shape[0],s4.shape[1],
                            #                                  s4.shape[2]*s4.shape[3]*s4.shape[4]]))
                            S4.append(
                                self.backend.bk_expand_dims(s4, off_S4)
                            )  # Add a dimension for NS4
                            if calc_var:

                                # VS4.append(self.backend.bk_reshape(vs4,[s4.shape[0],s4.shape[1],
                                #                              s4.shape[2]*s4.shape[3]*s4.shape[4]]))
                                VS4.append(
                                    self.backend.bk_expand_dims(vs4, off_S4)
                                )  # Add a dimension for NS4

            ###### Reshape for next iteration on j3
            ### Image I1,
            # downscale the I1 [Nbatch, Npix_j3]
            if j3 != Jmax - 1:
                I1 = self.smooth(I1, cell_ids=cell_ids_j3, nside=nside_j3)
                I1, new_cell_ids_j3 = self.ud_grade_2(
                    I1, cell_ids=cell_ids_j3, nside=nside_j3
                )

                ### Image I2
                if cross:
                    I2 = self.smooth(I2,  cell_ids=cell_ids_j3, nside=nside_j3)
                    I2, new_cell_ids_j3 = self.ud_grade_2(
                        I2, cell_ids=cell_ids_j3, nside=nside_j3
                    )

                ### Modules
                for j2 in range(0, j3 + 1):  # j2 =< j3
                    ### Dictionary M1_dic[j2]
                    M1_smooth = self.smooth(
                        M1_dic[j2], cell_ids=cell_ids_j3, nside=nside_j3
                    )  # [Nbatch, Npix_j3, Norient3]
                    M1_dic[j2], new_cell_ids_j2 = self.ud_grade_2(
                        M1_smooth,  cell_ids=cell_ids_j3, nside=nside_j3
                    )  # [Nbatch, Npix_j3, Norient3]

                    ### Dictionary M2_dic[j2]
                    if cross:
                        M2_smooth = self.smooth(
                            M2_dic[j2],  cell_ids=cell_ids_j3, nside=nside_j3
                        )  # [Nbatch, Npix_j3, Norient3]
                        M2_dic[j2], new_cell_ids_j2 = self.ud_grade_2(
                            M2_smooth,  cell_ids=cell_ids_j3, nside=nside_j3
                        )  # [Nbatch, Npix_j3, Norient3]
                ### Mask
                vmask, new_cell_ids_j3 = self.ud_grade_2(
                    vmask, cell_ids=cell_ids_j3, nside=nside_j3
                )

                if self.mask_thres is not None:
                    vmask = self.backend.bk_threshold(vmask, self.mask_thres)

                ### NSIDE_j3
                nside_j3 = nside_j3 // 2
                cell_ids_j3 = new_cell_ids_j3

        ### Store P1_dic and P2_dic in self
        if (norm == "auto") and (self.P1_dic is None):
            self.P1_dic = P1_dic
            if cross:
                self.P2_dic = P2_dic
                
        if not return_data:
            if not self.use_1D:
                S1 = self.backend.bk_concat(S1, -2)
                S2 = self.backend.bk_concat(S2, -2)
                S3 = self.backend.bk_concat(S3, -3)
                S4 = self.backend.bk_concat(S4, -4)
                if cross:
                    S3P = self.backend.bk_concat(S3P, -3)
                if calc_var:
                    VS1 = self.backend.bk_concat(VS1, -2)
                    VS2 = self.backend.bk_concat(VS2, -2)
                    VS3 = self.backend.bk_concat(VS3, -3)
                    VS4 = self.backend.bk_concat(VS4, -4)
                    if cross:
                        VS3P = self.backend.bk_concat(VS3P, -3)
            else:
                S1 = self.backend.bk_concat(S1, -1)
                S2 = self.backend.bk_concat(S2, -1)
                S3 = self.backend.bk_concat(S3, -1)
                S4 = self.backend.bk_concat(S4, -1)
                if cross:
                    S3P = self.backend.bk_concat(S3P, -1)
                if calc_var:
                    VS1 = self.backend.bk_concat(VS1, -1)
                    VS2 = self.backend.bk_concat(VS2, -1)
                    VS3 = self.backend.bk_concat(VS3, -1)
                    VS4 = self.backend.bk_concat(VS4, -1)
                    if cross:
                        VS3P = self.backend.bk_concat(VS3P, -1)
        if calc_var:
            if not cross:
                return scat_cov(
                    s0, S2, S3, S4, s1=S1, backend=self.backend,
                    use_1D=self.use_1D,
                    return_data=self.return_data
                ), scat_cov(
                    vs0,
                    VS2,
                    VS3,
                    VS4,
                    s1=VS1,
                    backend=self.backend,
                    use_1D=self.use_1D,
                    return_data=self.return_data
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
                    return_data=self.return_data
                ), scat_cov(
                    vs0,
                    VS2,
                    VS3,
                    VS4,
                    s1=VS1,
                    s3p=VS3P,
                    backend=self.backend,
                    use_1D=self.use_1D,
                    return_data=self.return_data
                )
        else:
            if not cross:
                return scat_cov(
                    s0, S2, S3, S4,
                    s1=S1,
                    backend=self.backend,
                    use_1D=self.use_1D,
                    return_data=self.return_data
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
                    return_data=self.return_data
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
            cell_ids=None,
            nside_j2=None,
            spin=0,
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
        # Warning: M1_dic[j2] is already at j3 resolution [Nbatch, Norient3, Npix_j3]
        MconvPsi = self.convol(
            M_dic[j2], cell_ids=cell_ids, nside=nside_j2
        )  # [Nbatch,   Norient3, Norient2, Npix_j3]

        if cmat2 is not None:
            tmp2 = self.backend.bk_repeat(MconvPsi, self.NORIENT, axis=-3)
            if spin==0:
                MconvPsi = self.backend.bk_reduce_sum(
                    self.backend.bk_reshape(
                        cmat2[j3][j2] * tmp2,
                        [
                            tmp2.shape[0],
                            self.NORIENT,
                            self.NORIENT,
                            self.NORIENT,
                            cmat2[j3][j2].shape[3],
                        ],
                    ),
                    1,
                )
            else:
                MconvPsi = self.backend.bk_reduce_sum(
                    self.backend.bk_reshape(
                        cmat2[j3][j2] * tmp2,
                        [
                            tmp2.shape[0],
                            2,
                            self.NORIENT,
                            self.NORIENT,
                            self.NORIENT,
                            cmat2[j3][j2].shape[4],
                        ],
                    ),
                    2,
                )

        # Store it so we can use it in S4 computation
        MconvPsi_dic[j2] = MconvPsi  # [Nbatch, Norient3, Norient2, Npix_j3]

        ### Compute the product (I2 * Psi)_j3 x (M1_j2 * Psi_j3)^*
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        # cconv, sconv are [Nbatch, Norient3, Npix_j3]
        if self.use_1D:
            s3 = conv * self.backend.bk_conjugate(MconvPsi)
        elif self.use_2D:
            s3 = self.backend.bk_expand_dims(conv, -4)* self.backend.bk_conjugate(
                MconvPsi
            )  # [Nbatch, Norient3, Norient2, Npix_j3]
        else:
            s3 = self.backend.bk_expand_dims(conv, -3)* self.backend.bk_conjugate(
                MconvPsi
            )  # [Nbatch, Norient3, Norient2, Npix_j3]
        ### Apply the mask [Nmask, Npix_j3] and sum over pixels
        if return_data:
            return s3
        else:
            if calc_var:
                s3, vs3 = self.masked_mean(
                    s3, vmask, rank=j2, calc_var=True
                )  # [Nbatch, Nmask, Norient3, Norient2]
                return s3, vs3
            else:
                s3 = self.masked_mean(
                    s3, vmask, rank=j2
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
        M1 = M1convPsi_dic[j1]  # [Nbatch, Norient3, Norient1, Npix_j3]

        # Auto or Cross coefficients
        if M2convPsi_dic is None:  # Auto
            M2 = M1convPsi_dic[j2]  # [Nbatch, Norient3, Norient2, Npix_j3]
        else:  # Cross
            M2 = M2convPsi_dic[j2]

        ### Compute the product (|I1 * Psi_j1| * Psi_j3)(|I2 * Psi_j2| * Psi_j3)
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        if self.use_1D:
            s4 = M1 * self.backend.bk_conjugate(M2)
        else:
            s4 = self.backend.bk_expand_dims(M1, -4) * self.backend.bk_conjugate(
                self.backend.bk_expand_dims(M2, -3)
            )  # [Nbatch,  Norient3, Norient2, Norient1,Npix_j3]

        ### Apply the mask and sum over pixels
        if return_data:
            return s4
        else:
            if calc_var:
                s4, vs4 = self.masked_mean(
                    s4, vmask, rank=j2, calc_var=True
                )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                return s4, vs4
            else:
                s4 = self.masked_mean(
                    s4, vmask, rank=j2
                )  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                return s4

    def computer_filter(self, M, N, J, L):
        """
        This function is strongly inspire by the package https://github.com/SihaoCheng/scattering_transform
        Done by Sihao Cheng and Rudy Morel.
        """

        if N!=0:
            filter = np.zeros([J, L, M, N], dtype="complex64")

            slant = 4.0 / L

            for j in range(J):

                for ell in range(L):

                    theta = (int(L - L / 2 - 1) - ell) * np.pi / L
                    sigma = 0.8 * 2**j
                    xi = 3.0 / 4.0 * np.pi / 2**j

                    R = np.array(
                        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
                        np.float64,
                    )
                    R_inv = np.array(
                        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]],
                        np.float64,
                    )
                    D = np.array([[1, 0], [0, slant * slant]])
                    curv = np.matmul(R, np.matmul(D, R_inv)) / (2 * sigma * sigma)

                    gab = np.zeros((M, N), np.complex128)
                    xx = np.empty((2, 2, M, N))
                    yy = np.empty((2, 2, M, N))

                    for ii, ex in enumerate([-1, 0]):
                        for jj, ey in enumerate([-1, 0]):
                            xx[ii, jj], yy[ii, jj] = np.mgrid[
                                ex * M : M + ex * M, ey * N : N + ey * N
                            ]

                    arg = -(
                        curv[0, 0] * xx * xx
                        + (curv[0, 1] + curv[1, 0]) * xx * yy
                        + curv[1, 1] * yy * yy
                    )
                    argi = arg + 1.0j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))

                    gabi = np.exp(argi).sum((0, 1))
                    gab = np.exp(arg).sum((0, 1))

                    norm_factor = 2 * np.pi * sigma * sigma / slant

                    gab = gab / norm_factor

                    gabi = gabi / norm_factor

                    K = gabi.sum() / gab.sum()

                    # Apply the Gaussian
                    filter[j, ell] = np.fft.fft2(gabi - K * gab)
                    filter[j, ell, 0, 0] = 0.0

            return self.backend.bk_cast(filter)
        else:
            filter = np.zeros([J, L, M], dtype="complex64")
            #TODO
            print('filter for 1D not yet available')
            exit(0)
            slant = 4.0 / L

            for j in range(J):

                for ell in range(L):

                    theta = (int(L - L / 2 - 1) - ell) * np.pi / L
                    sigma = 0.8 * 2**j
                    xi = 3.0 / 4.0 * np.pi / 2**j

                    R = np.array(
                        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
                        np.float64,
                    )
                    R_inv = np.array(
                        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]],
                        np.float64,
                    )
                    D = np.array([[1, 0], [0, slant * slant]])
                    curv = np.matmul(R, np.matmul(D, R_inv)) / (2 * sigma * sigma)

                    gab = np.zeros((M), np.complex128)
                    xx = np.empty((M))

                    for ii, ex in enumerate([-1, 0]):
                        for jj, ey in enumerate([-1, 0]):
                            xx[ii, jj], yy[ii, jj] = np.mgrid[
                                ex * M : M + ex * M, ey * N : N + ey * N
                            ]

                    arg = -(
                        curv[0, 0] * xx * xx
                        + (curv[0, 1] + curv[1, 0]) * xx * yy
                        + curv[1, 1] * yy * yy
                    )
                    argi = arg + 1.0j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))

                    gabi = np.exp(argi).sum((0, 1))
                    gab = np.exp(arg).sum((0, 1))

                    norm_factor = 2 * np.pi * sigma * sigma / slant

                    gab = gab / norm_factor

                    gabi = gabi / norm_factor

                    K = gabi.sum() / gab.sum()

                    # Apply the Gaussian
                    filter[j, ell] = np.fft.fft2(gabi - K * gab)
                    filter[j, ell, 0, 0] = 0.0

            return self.backend.bk_cast(filter)

    # ------------------------------------------------------------------------------------------
    #
    # utility functions
    #
    # ------------------------------------------------------------------------------------------
    def cut_high_k_off(self, data_f, dx, dy):
        """
        This function is strongly inspire by the package https://github.com/SihaoCheng/scattering_transform
        Done by Sihao Cheng and Rudy Morel.
        """

        if self.backend.BACKEND == "torch":
            if_xodd = data_f.shape[-2] % 2 == 1
            if_yodd = data_f.shape[-1] % 2 == 1
            result = self.backend.backend.cat(
                (
                    self.backend.backend.cat(
                        (
                            data_f[..., : dx + if_xodd, : dy + if_yodd],
                            data_f[..., -dx:, : dy + if_yodd],
                        ),
                        -2,
                    ),
                    self.backend.backend.cat(
                        (data_f[..., : dx + if_xodd, -dy:], data_f[..., -dx:, -dy:]), -2
                    ),
                ),
                -1,
            )
            return result
        else:
            # Check if the last two dimensions are odd
            if_xodd = self.backend.backend.cast(
                self.backend.backend.shape(data_f)[-2] % 2 == 1,
                self.backend.backend.int32,
            )
            if_yodd = self.backend.backend.cast(
                self.backend.backend.shape(data_f)[-1] % 2 == 1,
                self.backend.backend.int32,
            )

            # Extract four regions
            top_left = data_f[..., : dx + if_xodd, : dy + if_yodd]
            top_right = data_f[..., -dx:, : dy + if_yodd]
            bottom_left = data_f[..., : dx + if_xodd, -dy:]
            bottom_right = data_f[..., -dx:, -dy:]

            # Concatenate along the last two dimensions
            top = self.backend.backend.concat([top_left, top_right], axis=-2)
            bottom = self.backend.backend.concat([bottom_left, bottom_right], axis=-2)
            result = self.backend.backend.concat([top, bottom], axis=-1)

            return result

    # ---------------------------------------------------------------------------
    #
    # utility functions for computing scattering coef and covariance
    #
    # ---------------------------------------------------------------------------

    def get_dxdy(self, j, M, N):
        """
        This function is strongly inspire by the package https://github.com/SihaoCheng/scattering_transform
        Done by Sihao Cheng and Rudy Morel.
        """
        dx = int(max(8, min(np.ceil(M / 2**j), M // 2)))
        dy = int(max(8, min(np.ceil(N / 2**j), N // 2)))
        return dx, dy

    def get_edge_masks(self, M, N, J, d0=1, in_mask=None, edge_dx=None, edge_dy=None):
        """
        This function is strongly inspire by the package https://github.com/SihaoCheng/scattering_transform
        Done by Sihao Cheng and Rudy Morel.
        """
        edge_masks = np.empty((J, M, N))

        X, Y = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
        if in_mask is not None:
            from scipy.ndimage import binary_erosion

        if in_mask is not None:
            if in_mask.shape[0] != M or in_mask.shape[0] != N:
                l_mask = in_mask.reshape(
                    M, in_mask.shape[0] // M, N, in_mask.shape[1] // N
                )
                l_mask = (
                    np.sum(np.sum(l_mask, 1), 2)
                    * (M * N)
                    / (in_mask.shape[0] * in_mask.shape[1])
                )
            else:
                l_mask = in_mask

        if edge_dx is None:
            for j in range(J):
                edge_dx = min(M // 4, 2**j * d0)
                edge_dy = min(N // 4, 2**j * d0)

                edge_masks[j] = (
                    (X >= edge_dx)
                    * (X < M - edge_dx)
                    * (Y >= edge_dy)
                    * (Y < N - edge_dy)
                )
                if in_mask is not None:
                    l_mask = binary_erosion(
                        l_mask, iterations=1 + np.max([edge_dx, edge_dy])
                    )
                    edge_masks[j] *= l_mask

            edge_masks = edge_masks[:, None, :, :]

            edge_masks = edge_masks / edge_masks.mean((-2, -1))[:, :, None, None]
        else:
            edge_masks = (
                (X >= edge_dx) * (X < M - edge_dx) * (Y >= edge_dy) * (Y < N - edge_dy)
            )
            if in_mask is not None:
                l_mask = binary_erosion(
                    l_mask, iterations=1 + np.max([edge_dx, edge_dy])
                )
                edge_masks *= l_mask

            edge_masks = edge_masks / edge_masks.mean((-2, -1))

        return self.backend.bk_cast(edge_masks)

    # ---------------------------------------------------------------------------
    #
    # scattering cov
    #
    # ---------------------------------------------------------------------------
    def scattering_cov(
            self,
            data,
            data2=None,
            Jmax=None,
            if_large_batch=False,
            S4_criteria=None,
            use_ref=False,
            normalization="S2",
            edge=False,
            in_mask=None,
            pseudo_coef=1,
            get_variance=False,
            ref_sigma=None,
            iso_ang=False,
            return_table=False,
    ):
        """
        Calculates the scattering correlations for a batch of images, including:

        This function is strongly inspire by the package https://github.com/SihaoCheng/scattering_transform
        Done by Sihao Cheng and Rudy Morel.

        orig. x orig.:
                        P00 = <(I * psi)(I * psi)*> = L2(I * psi)^2
        orig. x modulus:
                        C01 = <(I * psi2)(|I * psi1| * psi2)*> / factor
            when normalization == 'P00', factor = L2(I * psi2) * L2(I * psi1)
            when normalization == 'P11', factor = L2(I * psi2) * L2(|I * psi1| * psi2)
        modulus x modulus:
                        C11_pre_norm = <(|I * psi1| * psi3)(|I * psi2| * psi3)>
                        C11 = C11_pre_norm / factor
            when normalization == 'P00', factor = L2(I * psi1) * L2(I * psi2)
            when normalization == 'P11', factor = L2(|I * psi1| * psi3) * L2(|I * psi2| * psi3)
        modulus x modulus (auto):
                        P11 = <(|I * psi1| * psi2)(|I * psi1| * psi2)*>
        Parameters
        ----------
        data : numpy array or torch tensor
            image set, with size [N_image, x-sidelength, y-sidelength]
        if_large_batch : Bool (=False)
            It is recommended to use "False" unless one meets a memory issue
        C11_criteria : str or None (=None)
            Only C11 coefficients that satisfy this criteria will be computed.
            Any expressions of j1, j2, and j3 that can be evaluated as a Bool
            is accepted.The default "None" corresponds to "j1 <= j2 <= j3".
        use_ref : Bool (=False)
            When normalizing, whether or not to use the normalization factor
            computed from a reference field. For just computing the statistics,
            the default is False. However, for synthesis, set it to "True" will
            stablize the optimization process.
        normalization : str 'P00' or 'P11' (='P00')
            Whether 'P00' or 'P11' is used as the normalization factor for C01
            and C11.
        remove_edge : Bool (=False)
            If true, the edge region with a width of rougly the size of the largest
            wavelet involved is excluded when taking the global average to obtain
            the scattering coefficients.

        Returns
        -------
        'P00'       : torch tensor with size [N_image, J, L] (# image, j1, l1)
            the power in each wavelet bands (the orig. x orig. term)
        'S1'        : torch tensor with size [N_image, J, L] (# image, j1, l1)
            the 1st-order scattering coefficients, i.e., the mean of wavelet modulus fields
        'C01'       : torch tensor with size [N_image, J, J, L, L] (# image, j1, j2, l1, l2)
            the orig. x modulus terms. Elements with j1 < j2 are all set to np.nan and not computed.
        'C11'       : torch tensor with size [N_image, J, J, J, L, L, L] (# image, j1, j2, j3, l1, l2, l3)
            the modulus x modulus terms. Elements not satisfying j1 <= j2 <= j3 and the conditions
            defined in 'C11_criteria' are all set to np.nan and not computed.
        'C11_pre_norm' and 'C11_pre_norm_iso': pre-normalized modulus x modulus terms.
        'P11'       : torch tensor with size [N_image, J, J, L, L] (# image, j1, j2, l1, l2)
            the modulus x modulus terms with the two wavelets within modulus the same. Elements not following
            j1 <= j3 are set to np.nan and not computed.
        'P11_iso'   : torch tensor with size [N_image, J, J, L] (# image, j1, j2, l2-l1)
            'P11' averaged over l1 while keeping l2-l1 constant.
        """
        if S4_criteria is None:
            S4_criteria = "j2>=j1"

        if not edge and in_mask is not None:
            edge = True

        if self.all_bk_type == "float32":
            C_ONE = np.complex64(1.0)
        else:
            C_ONE = np.complex128(1.0)

        # determine jmax and nside corresponding to the input map
        im_shape = data.shape
        if self.use_2D:
            if len(data.shape) == 2:
                nside = np.min([im_shape[0], im_shape[1]])
                M, N = im_shape[0], im_shape[1]
                N_image = 1
                N_image2 = 1
            else:
                nside = np.min([im_shape[1], im_shape[2]])
                M, N = im_shape[1], im_shape[2]
                N_image = data.shape[0]
                if data2 is not None:
                    N_image2 = data2.shape[0]
            J = int(np.log(nside) / np.log(2)) - 1  # Number of j scales
            dim=(-2,-1)
        elif self.use_1D:
            if len(data.shape) == 2:
                npix = int(im_shape[1])  # Number of pixels
                M = im_shape[1]
                N=0
                N_image = 1
                N_image2 = 1
            else:
                npix = int(im_shape[0])  # Number of pixels
                N_image = data.shape[0]
                M = im_shape[0]
                N=0
                if data2 is not None:
                    N_image2 = data2.shape[0]

            nside = int(npix)
            dim=(-1)

            J = int(np.log(nside) / np.log(2)) - 1  # Number of j scales
        else:
            if len(data.shape) == 2:
                npix = int(im_shape[1])  # Number of pixels
                N_image = 1
                N_image2 = 1
            else:
                npix = int(im_shape[0])  # Number of pixels
                N_image = data.shape[0]
                if data2 is not None:
                    N_image2 = data2.shape[0]

            if spin==0:
                nside = int(np.sqrt(npix // 12))
            else:
                nside = int(np.sqrt(npix // 24))

            J = int(np.log(nside) / np.log(2))  # Number of j scales

        if Jmax is not None:

            if Jmax > J:
                print("==========\n\n")
                print(
                    "The Jmax you requested is larger than the data size, which may cause problems while computing the scattering transform."
                )
                print("\n\n==========")
            J = Jmax  # Number of steps for the loop on scales
            
        L = self.NORIENT
        norm_factor_S3 = 1.0

        if self.backend.BACKEND == "torch":
            if (M, N, J, L) not in self.filters_set:
                self.filters_set[(M, N, J, L)] = self.computer_filter(
                    M, N, J, L
                )  # self.computer_filter(M,N,J,L)

            filters_set = self.filters_set[(M, N, J, L)]

            # weight = self.weight
            if use_ref:
                if normalization == "S2":
                    ref_S2 = self.ref_scattering_cov_S2
                else:
                    ref_P11 = self.ref_scattering_cov["P11"]

            # convert numpy array input into self.backend.bk_ tensors
            data = self.backend.bk_cast(data)
            data_f = self.backend.bk_fftn(data, dim=dim)
            if data2 is not None:
                data2 = self.backend.bk_cast(data2)
                data2_f = self.backend.bk_fftn(data2, dim=dim)

            # initialize tensors for scattering coefficients
            S2 = self.backend.bk_zeros((N_image, J, L), dtype=data.dtype)
            S1 = self.backend.bk_zeros((N_image, J, L), dtype=data.dtype)

            Ndata_S3 = J * (J + 1) // 2
            Ndata_S4 = J * (J + 1) * (J + 2) // 6
            J_S4 = {}

            S3 = self.backend.bk_zeros((N_image, Ndata_S3, L, L), dtype=data_f.dtype)
            if data2 is not None:
                S3p = self.backend.bk_zeros(
                    (N_image, Ndata_S3, L, L), dtype=data_f.dtype
                )
            S4_pre_norm = self.backend.bk_zeros(
                (N_image, Ndata_S4, L, L, L), dtype=data_f.dtype
            )
            S4 = self.backend.bk_zeros((N_image, Ndata_S4, L, L, L), dtype=data_f.dtype)

            # variance
            if get_variance:
                S2_sigma = self.backend.bk_zeros((N_image, J, L), dtype=data.dtype)
                S1_sigma = self.backend.bk_zeros((N_image, J, L), dtype=data.dtype)
                S3_sigma = self.backend.bk_zeros(
                    (N_image, Ndata_S3, L, L), dtype=data_f.dtype
                )
                if data2 is not None:
                    S3p_sigma = self.backend.bk_zeros(
                        (N_image, Ndata_S3, L, L), dtype=data_f.dtype
                    )
                S4_sigma = self.backend.bk_zeros(
                    (N_image, Ndata_S4, L, L, L), dtype=data_f.dtype
                )

            if iso_ang:
                S3_iso = self.backend.bk_zeros(
                    (N_image, Ndata_S3, L), dtype=data_f.dtype
                )
                S4_iso = self.backend.bk_zeros(
                    (N_image, Ndata_S4, L, L), dtype=data_f.dtype
                )
                if get_variance:
                    S3_sigma_iso = self.backend.bk_zeros(
                        (N_image, Ndata_S3, L), dtype=data_f.dtype
                    )
                    S4_sigma_iso = self.backend.bk_zeros(
                        (N_image, Ndata_S4, L, L), dtype=data_f.dtype
                    )
                if data2 is not None:
                    S3p_iso = self.backend.bk_zeros(
                        (N_image, Ndata_S3, L), dtype=data_f.dtype
                    )
                    if get_variance:
                        S3p_sigma_iso = self.backend.bk_zeros(
                            (N_image, Ndata_S3, L), dtype=data_f.dtype
                        )

            #
            if edge:
                if (M, N, J) not in self.edge_masks:
                    self.edge_masks[(M, N, J)] = self.get_edge_masks(
                        M, N, J, in_mask=in_mask
                    )

                edge_mask = self.edge_masks[(M, N, J)]
            else:
                edge_mask = 1

            # calculate scattering fields
            if data2 is None:
                if self.use_2D:
                    if len(data.shape) == 2:
                        I1 = self.backend.bk_ifftn(
                            data_f[None, None, None, :, :]
                            * filters_set[None, :J, :, :, :],
                            dim=dim,
                        ).abs()
                    else:
                        I1 = self.backend.bk_ifftn(
                            data_f[:, None, None, :, :]
                            * filters_set[None, :J, :, :, :],
                            dim=dim,
                        ).abs()
                elif self.use_1D:
                    if len(data.shape) == 1:
                        I1 = self.backend.bk_ifftn(
                            data_f[None, None, None, :] * filters_set[None, :J, :, :],
                            dim=(-1),
                        ).abs()
                    else:
                        I1 = self.backend.bk_ifftn(
                            data_f[:, None, None, :] * filters_set[None, :J, :, :],
                            dim=(-1),
                        ).abs()
                else:
                    print("todo")

                S2 = (I1**2 * edge_mask).mean(dim)
                S1 = (I1 * edge_mask).mean(dim)

                if get_variance:
                    S2_sigma = (I1**2 * edge_mask).std(dim)
                    S1_sigma = (I1 * edge_mask).std(dim)

            else:
                if self.use_2D:
                    if len(data.shape) == 2:
                        I1 = self.backend.bk_ifftn(
                            data_f[None, None, None, :, :]
                            * filters_set[None, :J, :, :, :],
                            dim=dim,
                        )
                        I2 = self.backend.bk_ifftn(
                            data2_f[None, None, None, :, :]
                            * filters_set[None, :J, :, :, :],
                            dim=dim,
                        )
                    else:
                        I1 = self.backend.bk_ifftn(
                            data_f[:, None, None, :, :]
                            * filters_set[None, :J, :, :, :],
                            dim=dim,
                        )
                        I2 = self.backend.bk_ifftn(
                            data2_f[:, None, None, :, :]
                            * filters_set[None, :J, :, :, :],
                            dim=dim,
                        )
                elif self.use_1D:
                    if len(data.shape) == 1:
                        I1 = self.backend.bk_ifftn(
                            data_f[None, None, None, :] * filters_set[None, :J, :, :],
                            dim=dim,
                        )
                        I2 = self.backend.bk_ifftn(
                            data2_f[None, None, None, :] * filters_set[None, :J, :, :],
                            dim=dim,
                        )
                    else:
                        I1 = self.backend.bk_ifftn(
                            data_f[:, None, None, :] * filters_set[None, :J, :, :],
                            dim=dim,
                        )
                        I2 = self.backend.bk_ifftn(
                            data2_f[:, None, None, :] * filters_set[None, :J, :, :],
                            dim=dim,
                        )
                else:
                    print("todo")

                I1 = self.backend.bk_real(I1 * self.backend.bk_conjugate(I2))

                S2 = self.backend.bk_reduce_mean((I1 * edge_mask), axis=dim)
                if get_variance:
                    S2_sigma = self.backend.bk_reduce_std(
                        (I1 * edge_mask), axis=dim
                    )

                I1 = self.backend.bk_L1(I1)

                S1 = self.backend.bk_reduce_mean((I1 * edge_mask), axis=dim)

                if get_variance:
                    S1_sigma = self.backend.bk_reduce_std(
                        (I1 * edge_mask), axis=dim
                    )

            I1_f = self.backend.bk_fftn(I1, dim=dim)

            if pseudo_coef != 1:
                I1 = I1**pseudo_coef

            Ndata_S3 = 0
            Ndata_S4 = 0

            # calculate the covariance and correlations of the scattering fields
            # only use the low-k Fourier coefs when calculating large-j scattering coefs.
            for j3 in range(0, J):
                J_S4[j3] = Ndata_S4

                dx3, dy3 = self.get_dxdy(j3, M, N)
                I1_f_small = self.cut_high_k_off(
                    I1_f[:, : j3 + 1], dx3, dy3
                )  # Nimage, J, L, x, y
                data_f_small = self.cut_high_k_off(data_f, dx3, dy3)
                if data2 is not None:
                    data2_f_small = self.cut_high_k_off(data2_f, dx3, dy3)
                if edge:
                    I1_small = self.backend.bk_ifftn(
                        I1_f_small, dim=dim, norm="ortho"
                    )
                    data_small = self.backend.bk_ifftn(
                        data_f_small, dim=dim, norm="ortho"
                    )
                    if data2 is not None:
                        data2_small = self.backend.bk_ifftn(
                            data2_f_small, dim=dim, norm="ortho"
                        )

                wavelet_f3 = self.cut_high_k_off(filters_set[j3], dx3, dy3)  # L,x,y
                _, M3, N3 = wavelet_f3.shape
                wavelet_f3_squared = wavelet_f3**2
                if edge is True:
                    if (M3, N3, J, j3) not in self.edge_masks:

                        edge_dx = min(4, int(2**j3 * dx3 * 2 / M))
                        edge_dy = min(4, int(2**j3 * dy3 * 2 / N))

                        self.edge_masks[(M3, N3, J, j3)] = self.get_edge_masks(
                            M3, N3, J, in_mask=in_mask, edge_dx=edge_dx, edge_dy=edge_dy
                        )

                    edge_mask = self.edge_masks[(M3, N3, J, j3)]
                else:
                    edge_mask = 1

                # a normalization change due to the cutoff of frequency space
                fft_factor = 1 / (M3 * N3) * (M3 * N3 / M / N) ** 2
                for j2 in range(0, j3 + 1):
                    I1_f2_wf3_small = I1_f_small[:, j2].view(
                        N_image, L, 1, M3, N3
                    ) * wavelet_f3.view(1, 1, L, M3, N3)
                    I1_f2_wf3_2_small = I1_f_small[:, j2].view(
                        N_image, L, 1, M3, N3
                    ) * wavelet_f3_squared.view(1, 1, L, M3, N3)
                    if edge:
                        I12_w3_small = self.backend.bk_ifftn(
                            I1_f2_wf3_small, dim=dim, norm="ortho"
                        )
                        I12_w3_2_small = self.backend.bk_ifftn(
                            I1_f2_wf3_2_small, dim=dim, norm="ortho"
                        )
                    if use_ref:
                        if normalization == "P11":
                            norm_factor_S3 = (
                                ref_S2[:, None, j3, :]
                                * ref_P11[:, j2, j3, :, :] ** pseudo_coef
                            ) ** 0.5
                        if normalization == "S2":
                            norm_factor_S3 = (
                                ref_S2[:, None, j3, :]
                                * ref_S2[:, j2, :, None] ** pseudo_coef
                            ) ** 0.5
                    else:
                        if normalization == "P11":
                            # [N_image,l2,l3,x,y]
                            P11_temp = (I1_f2_wf3_small.abs() ** 2).mean(
                                dim
                            ) * fft_factor
                            norm_factor_S3 = (
                                S2[:, None, j3, :] * P11_temp**pseudo_coef
                            ) ** 0.5
                        if normalization == "S2":
                            norm_factor_S3 = (
                                S2[:, None, j3, :] * S2[:, j2, :, None] ** pseudo_coef
                            ) ** 0.5

                    if not edge:
                        S3[:, Ndata_S3, :, :] = (
                            (
                                data_f_small.view(N_image, 1, 1, M3, N3)
                                * self.backend.bk_conjugate(I1_f2_wf3_small)
                            ).mean(dim)
                            * fft_factor
                            / norm_factor_S3
                        )

                        if get_variance:
                            S3_sigma[:, Ndata_S3, :, :] = (
                                (
                                    data_f_small.view(N_image, 1, 1, M3, N3)
                                    * self.backend.bk_conjugate(I1_f2_wf3_small)
                                ).std(dim)
                                * fft_factor
                                / norm_factor_S3
                            )
                    else:
                        S3[:, Ndata_S3, :, :] = (
                            (
                                data_small.view(N_image, 1, 1, M3, N3)
                                * self.backend.bk_conjugate(I12_w3_small)
                                * edge_mask[None, None, None, :, :]
                            ).mean(  # [..., edge_dx : M3 - edge_dx, edge_dy : N3 - edge_dy]
                                dim
                            )
                            * fft_factor
                            / norm_factor_S3
                        )
                        if get_variance:
                            S3_sigma[:, Ndata_S3, :, :] = (
                                (
                                    data_small.view(N_image, 1, 1, M3, N3)
                                    * self.backend.bk_conjugate(I12_w3_small)
                                    * edge_mask[None, None, None, :, :]
                                ).std(dim)
                                * fft_factor
                                / norm_factor_S3
                            )
                    if data2 is not None:
                        if not edge:
                            S3p[:, Ndata_S3, :, :] = (
                                (
                                    data2_f_small.view(N_image2, 1, 1, M3, N3)
                                    * self.backend.bk_conjugate(I1_f2_wf3_small)
                                ).mean(dim)
                                * fft_factor
                                / norm_factor_S3
                            )

                            if get_variance:
                                S3p_sigma[:, Ndata_S3, :, :] = (
                                    (
                                        data2_f_small.view(N_image2, 1, 1, M3, N3)
                                        * self.backend.bk_conjugate(I1_f2_wf3_small)
                                    ).std(dim)
                                    * fft_factor
                                    / norm_factor_S3
                                )
                        else:
                            S3p[:, Ndata_S3, :, :] = (
                                (
                                    data2_small.view(N_image2, 1, 1, M3, N3)
                                    * self.backend.bk_conjugate(I12_w3_small)
                                    * edge_mask[None, None, None, :, :]
                                ).mean(dim)
                                * fft_factor
                                / norm_factor_S3
                            )
                            if get_variance:
                                S3p_sigma[:, Ndata_S3, :, :] = (
                                    (
                                        data2_small.view(N_image2, 1, 1, M3, N3)
                                        * self.backend.bk_conjugate(I12_w3_small)
                                        * edge_mask[None, None, None, :, :]
                                    ).std(dim)
                                    * fft_factor
                                    / norm_factor_S3
                                )
                    Ndata_S3 += 1
                    if j2 <= j3:
                        beg_n = Ndata_S4
                        for j1 in range(0, j2 + 1):
                            if eval(S4_criteria):
                                if not edge:
                                    if not if_large_batch:
                                        # [N_image,l1,l2,l3,x,y]
                                        S4_pre_norm[:, Ndata_S4, :, :, :] = (
                                            I1_f_small[:, j1].view(
                                                N_image, L, 1, 1, M3, N3
                                            )
                                            * self.backend.bk_conjugate(
                                                I1_f2_wf3_2_small.view(
                                                    N_image, 1, L, L, M3, N3
                                                )
                                            )
                                        ).mean(dim) * fft_factor
                                        if get_variance:
                                            S4_sigma[:, Ndata_S4, :, :, :] = (
                                                I1_f_small[:, j1].view(
                                                    N_image, L, 1, 1, M3, N3
                                                )
                                                * self.backend.bk_conjugate(
                                                    I1_f2_wf3_2_small.view(
                                                        N_image, 1, L, L, M3, N3
                                                    )
                                                )
                                            ).std(dim) * fft_factor
                                    else:
                                        for l1 in range(L):
                                            # [N_image,l2,l3,x,y]
                                            S4_pre_norm[:, Ndata_S4, l1, :, :] = (
                                                I1_f_small[:, j1, l1].view(
                                                    N_image, 1, 1, M3, N3
                                                )
                                                * self.backend.bk_conjugate(
                                                    I1_f2_wf3_2_small.view(
                                                        N_image, L, L, M3, N3
                                                    )
                                                )
                                            ).mean(dim) * fft_factor
                                            if get_variance:
                                                S4_sigma[:, Ndata_S4, l1, :, :] = (
                                                    I1_f_small[:, j1, l1].view(
                                                        N_image, 1, 1, M3, N3
                                                    )
                                                    * self.backend.bk_conjugate(
                                                        I1_f2_wf3_2_small.view(
                                                            N_image, L, L, M3, N3
                                                        )
                                                    )
                                                ).std(dim) * fft_factor
                                else:
                                    if not if_large_batch:
                                        # [N_image,l1,l2,l3,x,y]
                                        S4_pre_norm[:, Ndata_S4, :, :, :] = (
                                            I1_small[:, j1].view(
                                                N_image, L, 1, 1, M3, N3
                                            )
                                            * self.backend.bk_conjugate(
                                                I12_w3_2_small.view(
                                                    N_image, 1, L, L, M3, N3
                                                )
                                            )
                                            * edge_mask[None, None, None, None, :, :]
                                        ).mean(dim) * fft_factor
                                        if get_variance:
                                            S4_sigma[:, Ndata_S4, :, :, :] = (
                                                I1_small[:, j1].view(
                                                    N_image, L, 1, 1, M3, N3
                                                )
                                                * self.backend.bk_conjugate(
                                                    I12_w3_2_small.view(
                                                        N_image, 1, L, L, M3, N3
                                                    )
                                                )
                                                * edge_mask[
                                                    None, None, None, None, :, :
                                                ]
                                            ).std(dim) * fft_factor
                                    else:
                                        for l1 in range(L):
                                            # [N_image,l2,l3,x,y]
                                            S4_pre_norm[:, Ndata_S4, l1, :, :] = (
                                                I1_small[:, j1].view(
                                                    N_image, 1, 1, M3, N3
                                                )
                                                * self.backend.bk_conjugate(
                                                    I12_w3_2_small.view(
                                                        N_image, L, L, M3, N3
                                                    )
                                                )
                                                * edge_mask[
                                                    None, None, None, None, :, :
                                                ]
                                            ).mean(dim) * fft_factor
                                            if get_variance:
                                                S4_sigma[:, Ndata_S4, l1, :, :] = (
                                                    I1_small[:, j1].view(
                                                        N_image, 1, 1, M3, N3
                                                    )
                                                    * self.backend.bk_conjugate(
                                                        I12_w3_2_small.view(
                                                            N_image, L, L, M3, N3
                                                        )
                                                    )
                                                    * edge_mask[
                                                        None, None, None, None, :, :
                                                    ]
                                                ).std(dim) * fft_factor

                                Ndata_S4 += 1

                        if normalization == "S2":
                            if use_ref:
                                P = (
                                    ref_S2[:, j3 : j3 + 1, :, None, None]
                                    * ref_S2[:, j2 : j2 + 1, None, :, None]
                                ) ** (0.5 * pseudo_coef)
                            else:
                                P = (
                                    S2[:, j3 : j3 + 1, :, None, None]
                                    * S2[:, j2 : j2 + 1, None, :, None]
                                ) ** (0.5 * pseudo_coef)

                            S4[:, beg_n:Ndata_S4, :, :, :] = (
                                S4_pre_norm[:, beg_n:Ndata_S4, :, :, :].clone() / P
                            )

                            if get_variance:
                                S4_sigma[:, beg_n:Ndata_S4, :, :, :] = (
                                    S4_sigma[:, beg_n:Ndata_S4, :, :, :] / P
                                )
                        else:
                            S4 = S4_pre_norm

            # average over l1 to obtain simple isotropic statistics
            if iso_ang:
                S2_iso = S2.mean(-1)
                S1_iso = S1.mean(-1)
                for l1 in range(L):
                    for l2 in range(L):
                        S3_iso[..., (l2 - l1) % L] += S3[..., l1, l2]
                        if data2 is not None:
                            S3p_iso[..., (l2 - l1) % L] += S3p[..., l1, l2]
                        for l3 in range(L):
                            S4_iso[..., (l2 - l1) % L, (l3 - l1) % L] += S4[
                                ..., l1, l2, l3
                            ]
                S3_iso /= L
                S4_iso /= L
                if data2 is not None:
                    S3p_iso /= L

                if get_variance:
                    S2_sigma_iso = S2_sigma.mean(-1)
                    S1_sigma_iso = S1_sigma.mean(-1)
                    for l1 in range(L):
                        for l2 in range(L):
                            S3_sigma_iso[..., (l2 - l1) % L] += S3_sigma[..., l1, l2]
                            if data2 is not None:
                                S3p_sigma_iso[..., (l2 - l1) % L] += S3p_sigma[
                                    ..., l1, l2
                                ]
                            for l3 in range(L):
                                S4_sigma_iso[
                                    ..., (l2 - l1) % L, (l3 - l1) % L
                                ] += S4_sigma[..., l1, l2, l3]
                    S3_sigma_iso /= L
                    S4_sigma_iso /= L
                    if data2 is not None:
                        S3p_sigma_iso /= L

            mean_data = self.backend.bk_zeros((N_image, 1), dtype=data.dtype)
            std_data = self.backend.bk_zeros((N_image, 1), dtype=data.dtype)

            if data2 is None:
                mean_data[:, 0] = data.mean(dim)
                std_data[:, 0] = data.std(dim)
            else:
                mean_data[:, 0] = (data2 * data).mean(dim)
                std_data[:, 0] = (data2 * data).std(dim)

            if get_variance:
                ref_sigma = {}
                if iso_ang:
                    ref_sigma["std_data"] = std_data
                    ref_sigma["S1_sigma"] = S1_sigma_iso
                    ref_sigma["S2_sigma"] = S2_sigma_iso
                    ref_sigma["S3_sigma"] = S3_sigma_iso
                    if data2 is not None:
                        ref_sigma["S3p_sigma"] = S3p_sigma_iso
                    ref_sigma["S4_sigma"] = S4_sigma_iso
                else:
                    ref_sigma["std_data"] = std_data
                    ref_sigma["S1_sigma"] = S1_sigma
                    ref_sigma["S2_sigma"] = S2_sigma
                    ref_sigma["S3_sigma"] = S3_sigma
                    if data2 is not None:
                        ref_sigma["S3p_sigma"] = S3p_sigma
                    ref_sigma["S4_sigma"] = S4_sigma

            if data2 is None:
                if iso_ang:
                    if ref_sigma is not None:
                        if return_table:
                            return (S1_iso / ref_sigma["S1_sigma"]), \
                                (S2_iso / ref_sigma["S2_sigma"]) , \
                                (S3_iso / ref_sigma["S3_sigma"]) , \
                                (S4_iso / ref_sigma["S4_sigma"]) 
                        
                        for_synthesis = self.backend.backend.cat(
                            (
                                mean_data / ref_sigma["std_data"],
                                std_data / ref_sigma["std_data"],
                                (S2_iso / ref_sigma["S2_sigma"])
                                .reshape((N_image, -1))
                                .log(),
                                (S1_iso / ref_sigma["S1_sigma"])
                                .reshape((N_image, -1))
                                .log(),
                                (S3_iso / ref_sigma["S3_sigma"])
                                .reshape((N_image, -1))
                                .real,
                                (S3_iso / ref_sigma["S3_sigma"])
                                .reshape((N_image, -1))
                                .imag,
                                (S4_iso / ref_sigma["S4_sigma"])
                                .reshape((N_image, -1))
                                .real,
                                (S4_iso / ref_sigma["S4_sigma"])
                                .reshape((N_image, -1))
                                .imag,
                            ),
                            dim=-1,
                        )
                    else:
                        if return_table:
                            return S1_iso,S2_iso,S3_iso,S4_iso
                        
                        for_synthesis = self.backend.backend.cat(
                            (
                                mean_data / std_data,
                                std_data,
                                S2_iso.reshape((N_image, -1)).log(),
                                S1_iso.reshape((N_image, -1)).log(),
                                S3_iso.reshape((N_image, -1)).real,
                                S3_iso.reshape((N_image, -1)).imag,
                                S4_iso.reshape((N_image, -1)).real,
                                S4_iso.reshape((N_image, -1)).imag,
                            ),
                            dim=-1,
                        )
                else:
                    if ref_sigma is not None:
                        if return_table:
                            return (S1 / ref_sigma["S1_sigma"]), \
                                (S2 / ref_sigma["S2_sigma"]), \
                                (S3 / ref_sigma["S3_sigma"]), \
                                (S4 / ref_sigma["S4_sigma"])
                        
                        for_synthesis = self.backend.backend.cat(
                            (
                                mean_data / ref_sigma["std_data"],
                                std_data / ref_sigma["std_data"],
                                (S2 / ref_sigma["S2_sigma"])
                                .reshape((N_image, -1))
                                .log(),
                                (S1 / ref_sigma["S1_sigma"])
                                .reshape((N_image, -1))
                                .log(),
                                (S3 / ref_sigma["S3_sigma"])
                                .reshape((N_image, -1))
                                .real,
                                (S3 / ref_sigma["S3_sigma"])
                                .reshape((N_image, -1))
                                .imag,
                                (S4 / ref_sigma["S4_sigma"])
                                .reshape((N_image, -1))
                                .real,
                                (S4 / ref_sigma["S4_sigma"])
                                .reshape((N_image, -1))
                                .imag,
                            ),
                            dim=-1,
                        )
                    else:
                        if return_table:
                            return S1,S2,S3,S4
                        
                        for_synthesis = self.backend.backend.cat(
                            (
                                mean_data / std_data,
                                std_data,
                                S2.reshape((N_image, -1)).log(),
                                S1.reshape((N_image, -1)).log(),
                                S3.reshape((N_image, -1)).real,
                                S3.reshape((N_image, -1)).imag,
                                S4.reshape((N_image, -1)).real,
                                S4.reshape((N_image, -1)).imag,
                            ),
                            dim=-1,
                        )
            else:
                if iso_ang:
                    if ref_sigma is not None:
                        if return_table:
                            return (S1_iso / ref_sigma["S1_sigma"]), \
                                (S2_iso / ref_sigma["S2_sigma"]), \
                                (S3_iso / ref_sigma["S3_sigma"]), \
                                (S4_iso / ref_sigma["S4_sigma"])
                        
                        for_synthesis = self.backend.backend.cat(
                            (
                                mean_data / ref_sigma["std_data"],
                                std_data / ref_sigma["std_data"],
                                (S2_iso / ref_sigma["S2_sigma"]).reshape((N_image, -1)),
                                (S1_iso / ref_sigma["S1_sigma"]).reshape((N_image, -1)),
                                (S3_iso / ref_sigma["S3_sigma"])
                                .reshape((N_image, -1))
                                .real,
                                (S3_iso / ref_sigma["S3_sigma"])
                                .reshape((N_image, -1))
                                .imag,
                                (S3p_iso / ref_sigma["S3p_sigma"])
                                .reshape((N_image, -1))
                                .real,
                                (S3p_iso / ref_sigma["S3p_sigma"])
                                .reshape((N_image, -1))
                                .imag,
                                (S4_iso / ref_sigma["S4_sigma"])
                                .reshape((N_image, -1))
                                .real,
                                (S4_iso / ref_sigma["S4_sigma"])
                                .reshape((N_image, -1))
                                .imag,
                            ),
                            dim=-1,
                        )
                    else:
                        if return_table:
                            return S1_iso,S2_iso,S3_iso,S4_iso
                        
                        for_synthesis = self.backend.backend.cat(
                            (
                                mean_data / std_data,
                                std_data,
                                S2_iso.reshape((N_image, -1)),
                                S1_iso.reshape((N_image, -1)),
                                S3_iso.reshape((N_image, -1)).real,
                                S3_iso.reshape((N_image, -1)).imag,
                                S3p_iso.reshape((N_image, -1)).real,
                                S3p_iso.reshape((N_image, -1)).imag,
                                S4_iso.reshape((N_image, -1)).real,
                                S4_iso.reshape((N_image, -1)).imag,
                            ),
                            dim=-1,
                        )
                else:
                    if ref_sigma is not None:
                        if return_table:
                            return (S1 / ref_sigma["S1_sigma"]), \
                                (S2 / ref_sigma["S2_sigma"]), \
                                (S3 / ref_sigma["S3_sigma"]), \
                                (S4 / ref_sigma["S4_sigma"])
                        
                        for_synthesis = self.backend.backend.cat(
                            (
                                mean_data / ref_sigma["std_data"],
                                std_data / ref_sigma["std_data"],
                                (S2 / ref_sigma["S2_sigma"]).reshape((N_image, -1)),
                                (S1 / ref_sigma["S1_sigma"]).reshape((N_image, -1)),
                                (S3 / ref_sigma["S3_sigma"])
                                .reshape((N_image, -1))
                                .real,
                                (S3 / ref_sigma["S3_sigma"])
                                .reshape((N_image, -1))
                                .imag,
                                (S3p / ref_sigma["S3p_sigma"])
                                .reshape((N_image, -1))
                                .real,
                                (S3p / ref_sigma["S3p_sigma"])
                                .reshape((N_image, -1))
                                .imag,
                                (S4 / ref_sigma["S4_sigma"])
                                .reshape((N_image, -1))
                                .real,
                                (S4 / ref_sigma["S4_sigma"])
                                .reshape((N_image, -1))
                                .imag,
                            ),
                            dim=-1,
                        )
                    else:
                        if return_table:
                            return S1,S2,S3,S4
                        
                        for_synthesis = self.backend.backend.cat(
                            (
                                mean_data / std_data,
                                std_data,
                                S2.reshape((N_image, -1)),
                                S1.reshape((N_image, -1)),
                                S3.reshape((N_image, -1)).real,
                                S3.reshape((N_image, -1)).imag,
                                S3p.reshape((N_image, -1)).real,
                                S3p.reshape((N_image, -1)).imag,
                                S4.reshape((N_image, -1)).real,
                                S4.reshape((N_image, -1)).imag,
                            ),
                            dim=-1,
                        )

            if not use_ref:
                self.ref_scattering_cov_S2 = S2

            if get_variance:
                return for_synthesis, ref_sigma

            return for_synthesis

        if (M, N, J, L) not in self.filters_set:
            self.filters_set[(M, N, J, L)] = self.computer_filter(
                M, N, J, L
            )  # self.computer_filter(M,N,J,L)

        filters_set = self.filters_set[(M, N, J, L)]

        # weight = self.weight
        if use_ref:
            if normalization == "S2":
                ref_S2 = self.ref_scattering_cov_S2
            else:
                ref_P11 = self.ref_scattering_cov["P11"]

        # convert numpy array input into self.backend.bk_ tensors
        data = self.backend.bk_cast(data)
        data_f = self.backend.bk_fftn(data, dim=dim)
        if data2 is not None:
            data2 = self.backend.bk_cast(data2)
            data2_f = self.backend.bk_fftn(data2, dim=dim)

        # initialize tensors for scattering coefficients

        Ndata_S3 = J * (J + 1) // 2
        Ndata_S4 = J * (J + 1) * (J + 2) // 6
        J_S4 = {}

        S3 = []
        if data2 is not None:
            S3p = []
        S4_pre_norm = []
        S4 = []

        # variance
        if get_variance:
            S3_sigma = []
            if data2 is not None:
                S3p_sigma = []
            S4_sigma = []

        if iso_ang:
            S3_iso = []
            if data2 is not None:
                S3p_iso = []

            S4_iso = []
            if get_variance:
                S3_sigma_iso = []
                if data2 is not None:
                    S3p_sigma_iso = []
                S4_sigma_iso = []

        #
        if edge:
            if (M, N, J) not in self.edge_masks:
                self.edge_masks[(M, N, J)] = self.get_edge_masks(
                    M, N, J, in_mask=in_mask
                )
            edge_mask = self.edge_masks[(M, N, J)]
        else:
            edge_mask = 1

        # calculate scattering fields
        if data2 is None:
            if self.use_2D:
                if len(data.shape) == 2:
                    I1 = self.backend.bk_abs(
                        self.backend.bk_ifftn(
                            data_f[None, None, None, :, :]
                            * filters_set[None, :J, :, :, :],
                            dim=dim,
                        )
                    )
                else:
                    I1 = self.backend.bk_abs(
                        self.backend.bk_ifftn(
                            data_f[:, None, None, :, :]
                            * filters_set[None, :J, :, :, :],
                            dim=dim,
                        )
                    )
            elif self.use_1D:
                if len(data.shape) == 1:
                    I1 = self.backend.bk_abs(
                        self.backend.bk_ifftn(
                            data_f[None, None, None, :] * filters_set[None, :J, :, :],
                            dim=(-1),
                        )
                    )
                else:
                    I1 = self.backend.bk_abs(
                        self.backend.bk_ifftn(
                            data_f[:, None, None, :] * filters_set[None, :J, :, :],
                            dim=(-1),
                        )
                    )
            else:
                print("todo")

            S2 = self.backend.bk_reduce_mean((I1**2 * edge_mask), axis=dim)
            S1 = self.backend.bk_reduce_mean(I1 * edge_mask, axis=dim)

            if get_variance:
                S2_sigma = self.backend.bk_reduce_std(
                    (I1**2 * edge_mask), axis=dim
                )
                S1_sigma = self.backend.bk_reduce_std((I1 * edge_mask), axis=dim)

            I1_f = self.backend.bk_fftn(I1, dim=dim)

        else:
            if self.use_2D:
                if len(data.shape) == 2:
                    I1 = self.backend.bk_ifftn(
                        data_f[None, None, None, :, :] * filters_set[None, :J, :, :, :],
                        dim=dim,
                    )
                    I2 = self.backend.bk_ifftn(
                        data2_f[None, None, None, :, :]
                        * filters_set[None, :J, :, :, :],
                        dim=dim,
                    )
                else:
                    I1 = self.backend.bk_ifftn(
                        data_f[:, None, None, :, :] * filters_set[None, :J, :, :, :],
                        dim=dim,
                    )
                    I2 = self.backend.bk_ifftn(
                        data2_f[:, None, None, :, :] * filters_set[None, :J, :, :, :],
                        dim=dim,
                    )
            elif self.use_1D:
                if len(data.shape) == 1:
                    I1 = self.backend.bk_ifftn(
                        data_f[None, None, None, :] * filters_set[None, :J, :, :],
                        dim=(-1),
                    )
                    I2 = self.backend.bk_ifftn(
                        data2_f[None, None, None, :] * filters_set[None, :J, :, :],
                        dim=(-1),
                    )
                else:
                    I1 = self.backend.bk_ifftn(
                        data_f[:, None, None, :] * filters_set[None, :J, :, :], dim=(-1)
                    )
                    I2 = self.backend.bk_ifftn(
                        data2_f[:, None, None, :] * filters_set[None, :J, :, :],
                        dim=(-1),
                    )
            else:
                print("todo")

            I1 = self.backend.bk_real(I1 * self.backend.bk_conjugate(I2))

            S2 = self.backend.bk_reduce_mean((I1 * edge_mask), axis=dim)
            if get_variance:
                S2_sigma = self.backend.bk_reduce_std((I1 * edge_mask), axis=dim)

            I1 = self.backend.bk_L1(I1)

            S1 = self.backend.bk_reduce_mean((I1 * edge_mask), axis=dim)

            if get_variance:
                S1_sigma = self.backend.bk_reduce_std((I1 * edge_mask), axis=dim)

            I1_f = self.backend.bk_fftn(I1, dim=dim)

        if pseudo_coef != 1:
            I1 = I1**pseudo_coef

        Ndata_S3 = 0
        Ndata_S4 = 0

        # calculate the covariance and correlations of the scattering fields
        # only use the low-k Fourier coefs when calculating large-j scattering coefs.
        for j3 in range(0, J):
            J_S4[j3] = Ndata_S4

            dx3, dy3 = self.get_dxdy(j3, M, N)
            I1_f_small = self.cut_high_k_off(
                I1_f[:, : j3 + 1], dx3, dy3
            )  # Nimage, J, L, x, y
            data_f_small = self.cut_high_k_off(data_f, dx3, dy3)
            if data2 is not None:
                data2_f_small = self.cut_high_k_off(data2_f, dx3, dy3)
            if edge:
                I1_small = self.backend.bk_ifftn(I1_f_small, dim=dim, norm="ortho")
                data_small = self.backend.bk_ifftn(
                    data_f_small, dim=dim, norm="ortho"
                )
                if data2 is not None:
                    data2_small = self.backend.bk_ifftn(
                        data2_f_small, dim=dim, norm="ortho"
                    )
            wavelet_f3 = self.cut_high_k_off(filters_set[j3], dx3, dy3)  # L,x,y
            _, M3, N3 = wavelet_f3.shape
            wavelet_f3_squared = wavelet_f3**2
            edge_dx = min(4, int(2**j3 * dx3 * 2 / M))
            edge_dy = min(4, int(2**j3 * dy3 * 2 / N))
            # a normalization change due to the cutoff of frequency space
            if self.all_bk_type == "float32":
                fft_factor = np.complex64(1 / (M3 * N3) * (M3 * N3 / M / N) ** 2)
            else:
                fft_factor = np.complex128(1 / (M3 * N3) * (M3 * N3 / M / N) ** 2)
            for j2 in range(0, j3 + 1):
                # I1_f2_wf3_small = I1_f_small[:,j2].view(N_image,L,1,M3,N3) * wavelet_f3.view(1,1,L,M3,N3)
                # I1_f2_wf3_2_small = I1_f_small[:,j2].view(N_image,L,1,M3,N3) * wavelet_f3_squared.view(1,1,L,M3,N3)
                I1_f2_wf3_small = self.backend.bk_reshape(
                    I1_f_small[:, j2], [N_image, 1, L, 1, M3, N3]
                ) * self.backend.bk_reshape(wavelet_f3, [1, 1, 1, L, M3, N3])
                I1_f2_wf3_2_small = self.backend.bk_reshape(
                    I1_f_small[:, j2], [N_image, 1, L, 1, M3, N3]
                ) * self.backend.bk_reshape(wavelet_f3_squared, [1, 1, 1, L, M3, N3])
                if edge:
                    I12_w3_small = self.backend.bk_ifftn(
                        I1_f2_wf3_small, dim=dim, norm="ortho"
                    )
                    I12_w3_2_small = self.backend.bk_ifftn(
                        I1_f2_wf3_2_small, dim=dim, norm="ortho"
                    )
                if use_ref:
                    if normalization == "P11":
                        norm_factor_S3 = (
                            ref_S2[:, None, j3, :]
                            * ref_P11[:, j2, j3, :, :] ** pseudo_coef
                        ) ** 0.5
                        norm_factor_S3 = self.backend.bk_complex(
                            norm_factor_S3, 0 * norm_factor_S3
                        )
                    elif normalization == "S2":
                        norm_factor_S3 = (
                            ref_S2[:, None, j3, :]
                            * ref_S2[:, j2, :, None] ** pseudo_coef
                        ) ** 0.5
                        norm_factor_S3 = self.backend.bk_complex(
                            norm_factor_S3, 0 * norm_factor_S3
                        )
                    else:
                        norm_factor_S3 = C_ONE
                else:
                    if normalization == "P11":
                        # [N_image,l2,l3,x,y]
                        P11_temp = (
                            self.backend.bk_reduce_mean(
                                (I1_f2_wf3_small.abs() ** 2), axis=dim
                            )
                            * fft_factor
                        )
                        norm_factor_S3 = (
                            S2[:, None, j3, :] * P11_temp**pseudo_coef
                        ) ** 0.5
                        norm_factor_S3 = self.backend.bk_complex(
                            norm_factor_S3, 0 * norm_factor_S3
                        )
                    elif normalization == "S2":
                        norm_factor_S3 = (
                            S2[:, None, j3, None, :]
                            * S2[:, None, j2, :, None] ** pseudo_coef
                        ) ** 0.5
                        norm_factor_S3 = self.backend.bk_complex(
                            norm_factor_S3, 0 * norm_factor_S3
                        )
                    else:
                        norm_factor_S3 = C_ONE

                if not edge:
                    S3.append(
                        self.backend.bk_reduce_mean(
                            self.backend.bk_reshape(
                                data_f_small, [N_image, 1, 1, 1, M3, N3]
                            )
                            * self.backend.bk_conjugate(I1_f2_wf3_small),
                            axis=dim,
                        )
                        * fft_factor
                        / norm_factor_S3
                    )
                    if get_variance:
                        S3_sigma.append(
                            self.backend.bk_reduce_std(
                                self.backend.bk_reshape(
                                    data_f_small, [N_image, 1, 1, 1, M3, N3]
                                )
                                * self.backend.bk_conjugate(I1_f2_wf3_small),
                                axis=dim,
                            )
                            * fft_factor
                            / norm_factor_S3
                        )
                else:
                    S3.append(
                        self.backend.bk_reduce_mean(
                            (
                                self.backend.bk_reshape(
                                    data_small, [N_image, 1, 1, 1, M3, N3]
                                )
                                * self.backend.bk_conjugate(I12_w3_small)
                            )[..., edge_dx : M3 - edge_dx, edge_dy : N3 - edge_dy],
                            axis=dim,
                        )
                        * fft_factor
                        / norm_factor_S3
                    )
                    if get_variance:
                        S3_sigma.apend(
                            self.backend.bk_reduce_std(
                                (
                                    self.backend.bk_reshape(
                                        data_small, [N_image, 1, 1, 1, M3, N3]
                                    )
                                    * self.backend.bk_conjugate(I12_w3_small)
                                )[..., edge_dx : M3 - edge_dx, edge_dy : N3 - edge_dy],
                                axis=dim,
                            )
                            * fft_factor
                            / norm_factor_S3
                        )
                if data2 is not None:
                    if not edge:
                        S3p.append(
                            self.backend.bk_reduce_mean(
                                (
                                    self.backend.bk_reshape(
                                        data2_f_small, [N_image2, 1, 1, 1, M3, N3]
                                    )
                                    * self.backend.bk_conjugate(I1_f2_wf3_small)
                                ),
                                axis=dim,
                            )
                            * fft_factor
                            / norm_factor_S3
                        )

                        if get_variance:
                            S3p_sigma.append(
                                self.backend.bk_reduce_std(
                                    (
                                        self.backend.bk_reshape(
                                            data2_f_small, [N_image2, 1, 1, 1, M3, N3]
                                        )
                                        * self.backend.bk_conjugate(I1_f2_wf3_small)
                                    ),
                                    axis=dim,
                                )
                                * fft_factor
                                / norm_factor_S3
                            )
                    else:

                        S3p.append(
                            self.backend.bk_reduce_mean(
                                (
                                    self.backend.bk_reshape(
                                        data2_small, [N_image2, 1, 1, 1, M3, N3]
                                    )
                                    * self.backend.bk_conjugate(I12_w3_small)
                                )[..., edge_dx : M3 - edge_dx, edge_dy : N3 - edge_dy],
                                axis=dim,
                            )
                            * fft_factor
                            / norm_factor_S3
                        )
                        if get_variance:
                            S3p_sigma.append(
                                self.backend.bk_reduce_std(
                                    (
                                        self.backend.bk_reshape(
                                            data2_small, [N_image2, 1, 1, 1, M3, N3]
                                        )
                                        * self.backend.bk_conjugate(I12_w3_small)
                                    )[
                                        ...,
                                        edge_dx : M3 - edge_dx,
                                        edge_dy : N3 - edge_dy,
                                    ],
                                    axis=dim,
                                )
                                * fft_factor
                                / norm_factor_S3
                            )

                if j2 <= j3:
                    if normalization == "S2":
                        if use_ref:
                            P = 1 / (
                                (
                                    ref_S2[:, j3 : j3 + 1, :, None, None]
                                    * ref_S2[:, j2 : j2 + 1, None, :, None]
                                )
                                ** (0.5 * pseudo_coef)
                            )
                        else:
                            P = 1 / (
                                (
                                    S2[:, j3 : j3 + 1, :, None, None]
                                    * S2[:, j2 : j2 + 1, None, :, None]
                                )
                                ** (0.5 * pseudo_coef)
                            )
                        P = self.backend.bk_complex(P, 0.0 * P)
                    else:
                        P = C_ONE

                    for j1 in range(0, j2 + 1):
                        if not edge:
                            if not if_large_batch:
                                # [N_image,l1,l2,l3,x,y]
                                S4.append(
                                    self.backend.bk_reduce_mean(
                                        (
                                            self.backend.bk_reshape(
                                                I1_f_small[:, j1],
                                                [N_image, 1, L, 1, 1, M3, N3],
                                            )
                                            * self.backend.bk_conjugate(
                                                self.backend.bk_reshape(
                                                    I1_f2_wf3_2_small,
                                                    [N_image, 1, 1, L, L, M3, N3],
                                                )
                                            )
                                        ),
                                        axis=dim,
                                    )
                                    * fft_factor
                                    * P
                                )
                                if get_variance:
                                    S4_sigma.append(
                                        self.backend.bk_reduce_std(
                                            (
                                                self.backend.bk_reshape(
                                                    I1_f_small[:, j1],
                                                    [N_image, 1, L, 1, 1, M3, N3],
                                                )
                                                * self.backend.bk_conjugate(
                                                    self.backend.bk_reshape(
                                                        I1_f2_wf3_2_small,
                                                        [N_image, 1, 1, L, L, M3, N3],
                                                    )
                                                )
                                            ),
                                            axis=dim,
                                        )
                                        * fft_factor
                                        * P
                                    )
                            else:
                                for l1 in range(L):
                                    # [N_image,l2,l3,x,y]
                                    S4.append(
                                        self.backend.bk_reduce_mean(
                                            (
                                                self.backend.bk_reshape(
                                                    I1_f_small[:, j1, l1],
                                                    [N_image, 1, 1, 1, M3, N3],
                                                )
                                                * self.backend.bk_conjugate(
                                                    self.backend.bk_reshape(
                                                        I1_f2_wf3_2_small,
                                                        [N_image, 1, L, L, M3, N3],
                                                    )
                                                )
                                            ),
                                            axis=dim,
                                        )
                                        * fft_factor
                                        * P
                                    )
                                    if get_variance:
                                        S4_sigma.append(
                                            self.backend.bk_reduce_std(
                                                (
                                                    self.backend.bk_reshape(
                                                        I1_f_small[:, j1, l1],
                                                        [N_image, 1, 1, 1, M3, N3],
                                                    )
                                                    * self.backend.bk_conjugate(
                                                        self.backend.bk_reshape(
                                                            I1_f2_wf3_2_small,
                                                            [N_image, 1, L, L, M3, N3],
                                                        )
                                                    )
                                                ),
                                                axis=dim,
                                            )
                                            * fft_factor
                                            * P
                                        )
                        else:
                            if not if_large_batch:
                                # [N_image,l1,l2,l3,x,y]
                                S4.append(
                                    self.backend.bk_reduce_mean(
                                        (
                                            self.backend.bk_reshape(
                                                I1_small[:, j1],
                                                [N_image, 1, L, 1, 1, M3, N3],
                                            )
                                            * self.backend.bk_conjugate(
                                                self.backend.bk_reshape(
                                                    I12_w3_2_small,
                                                    [N_image, 1, 1, L, L, M3, N3],
                                                )
                                            )
                                        )[..., edge_dx:-edge_dx, edge_dy:-edge_dy],
                                        axis=dim,
                                    )
                                    * fft_factor
                                    * P
                                )
                                if get_variance:
                                    S4_sigma.append(
                                        self.backend.bk_reduce_std(
                                            (
                                                self.backend.bk_reshape(
                                                    I1_small[:, j1],
                                                    [N_image, 1, L, 1, 1, M3, N3],
                                                )
                                                * self.backend.bk_conjugate(
                                                    self.backend.bk_reshape(
                                                        I12_w3_2_small,
                                                        [N_image, 1, 1, L, L, M3, N3],
                                                    )
                                                )
                                            )[..., edge_dx:-edge_dx, edge_dy:-edge_dy],
                                            axis=dim,
                                        )
                                        * fft_factor
                                        * P
                                    )
                            else:
                                for l1 in range(L):
                                    # [N_image,l2,l3,x,y]
                                    S4.append(
                                        self.backend.bk_reduce_mean(
                                            (
                                                self.backend.bk_reshape(
                                                    I1_small[:, j1],
                                                    [N_image, 1, 1, 1, M3, N3],
                                                )
                                                * self.backend.bk_conjugate(
                                                    self.backend.bk_reshape(
                                                        I12_w3_2_small,
                                                        [N_image, 1, L, L, M3, N3],
                                                    )
                                                )
                                            )[..., edge_dx:-edge_dx, edge_dy:-edge_dy],
                                            axis=dim,
                                        )
                                        * fft_factor
                                        * P
                                    )
                                    if get_variance:
                                        S4_sigma.append(
                                            self.backend.bk_reduce_std(
                                                (
                                                    self.backend.bk_reshape(
                                                        I1_small[:, j1],
                                                        [N_image, 1, 1, 1, M3, N3],
                                                    )
                                                    * self.backend.bk_conjugate(
                                                        self.backend.bk_reshape(
                                                            I12_w3_2_small,
                                                            [N_image, 1, L, L, M3, N3],
                                                        )
                                                    )
                                                )[
                                                    ...,
                                                    edge_dx:-edge_dx,
                                                    edge_dy:-edge_dy,
                                                ],
                                                axis=dim,
                                            )
                                            * fft_factor
                                            * P
                                        )

        S3 = self.backend.bk_concat(S3, axis=1)
        S4 = self.backend.bk_concat(S4, axis=1)

        if get_variance:
            S3_sigma = self.backend.bk_concat(S3_sigma, axis=1)
            S4_sigma = self.backend.bk_concat(S4_sigma, axis=1)

        if data2 is not None:
            S3p = self.backend.bk_concat(S3p, axis=1)
            if get_variance:
                S3p_sigma = self.backend.bk_concat(S3p_sigma, axis=1)

        # average over l1 to obtain simple isotropic statistics
        if iso_ang:
            S2_iso = self.backend.bk_reduce_mean(S2, axis=(-1))
            S1_iso = self.backend.bk_reduce_mean(S1, axis=(-1))
            for l1 in range(L):
                for l2 in range(L):
                    S3_iso[..., (l2 - l1) % L] += S3[..., l1, l2]
                    if data2 is not None:
                        S3p_iso[..., (l2 - l1) % L] += S3p[..., l1, l2]
                    for l3 in range(L):
                        S4_iso[..., (l2 - l1) % L, (l3 - l1) % L] += S4[..., l1, l2, l3]
            S3_iso /= L
            S4_iso /= L
            if data2 is not None:
                S3p_iso /= L

            if get_variance:
                S2_sigma_iso = self.backend.bk_reduce_mean(S2_sigma, axis=(-1))
                S1_sigma_iso = self.backend.bk_reduce_mean(S1_sigma, axis=(-1))
                for l1 in range(L):
                    for l2 in range(L):
                        S3_sigma_iso[..., (l2 - l1) % L] += S3_sigma[..., l1, l2]
                        if data2 is not None:
                            S3p_sigma_iso[..., (l2 - l1) % L] += S3p_sigma[..., l1, l2]
                        for l3 in range(L):
                            S4_sigma_iso[..., (l2 - l1) % L, (l3 - l1) % L] += S4_sigma[
                                ..., l1, l2, l3
                            ]
                S3_sigma_iso /= L
                S4_sigma_iso /= L
                if data2 is not None:
                    S3p_sigma_iso /= L

        if data2 is None:
            mean_data = self.backend.bk_reshape(
                self.backend.bk_reduce_mean(data, axis=dim), [N_image, 1]
            )
            std_data = self.backend.bk_reshape(
                self.backend.bk_reduce_std(data, axis=dim), [N_image, 1]
            )
        else:
            mean_data = self.backend.bk_reshape(
                self.backend.bk_reduce_mean(data * data2, axis=dim), [N_image, 1]
            )
            std_data = self.backend.bk_reshape(
                self.backend.bk_reduce_std(data * data2, axis=dim), [N_image, 1]
            )

        if get_variance:
            ref_sigma = {}
            if iso_ang:
                ref_sigma["std_data"] = std_data
                ref_sigma["S1_sigma"] = S1_sigma_iso
                ref_sigma["S2_sigma"] = S2_sigma_iso
                ref_sigma["S3_sigma"] = S3_sigma_iso
                ref_sigma["S4_sigma"] = S4_sigma_iso
                if data2 is not None:
                    ref_sigma["S3p_sigma"] = S3p_sigma_iso
            else:
                ref_sigma["std_data"] = std_data
                ref_sigma["S1_sigma"] = S1_sigma
                ref_sigma["S2_sigma"] = S2_sigma
                ref_sigma["S3_sigma"] = S3_sigma
                ref_sigma["S4_sigma"] = S4_sigma
                if data2 is not None:
                    ref_sigma["S3p_sigma"] = S3_sigma

        if data2 is None:
            if iso_ang:
                if ref_sigma is not None:
                    for_synthesis = self.backend.bk_concat(
                        (
                            mean_data / ref_sigma["std_data"],
                            std_data / ref_sigma["std_data"],
                            self.backend.bk_reshape(
                                self.backend.bk_log(S2_iso / ref_sigma["S2_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_log(S1_iso / ref_sigma["S1_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S3_iso / ref_sigma["S3_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S3_iso / ref_sigma["S3_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S4_iso / ref_sigma["S4_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S4_iso / ref_sigma["S4_sigma"]),
                                [N_image, -1],
                            ),
                        ),
                        axis=-1,
                    )
                else:
                    for_synthesis = self.backend.bk_concat(
                        (
                            mean_data / std_data,
                            std_data,
                            self.backend.bk_reshape(
                                self.backend.bk_log(S2_iso), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_log(S1_iso), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S3_iso), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S3_iso), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S4_iso), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S4_iso), [N_image, -1]
                            ),
                        ),
                        axis=-1,
                    )
            else:
                if ref_sigma is not None:
                    for_synthesis = self.backend.bk_concat(
                        (
                            mean_data / ref_sigma["std_data"],
                            std_data / ref_sigma["std_data"],
                            self.backend.bk_reshape(
                                self.backend.bk_log(S2 / ref_sigma["S2_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_log(S1 / ref_sigma["S1_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S3 / ref_sigma["S3_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S3 / ref_sigma["S3_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S4 / ref_sigma["S4_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S4 / ref_sigma["S4_sigma"]),
                                [N_image, -1],
                            ),
                        ),
                        axis=-1,
                    )
                else:
                    for_synthesis = self.backend.bk_concat(
                        (
                            mean_data / std_data,
                            std_data,
                            self.backend.bk_reshape(
                                self.backend.bk_log(S2), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_log(S1), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S3), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S3), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S4), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S4), [N_image, -1]
                            ),
                        ),
                        axis=-1,
                    )
        else:
            if iso_ang:
                if ref_sigma is not None:
                    for_synthesis = self.backend.backend.cat(
                        (
                            mean_data / ref_sigma["std_data"],
                            std_data / ref_sigma["std_data"],
                            self.backend.bk_reshape(
                                self.backend.bk_real(S2_iso / ref_sigma["S2_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S1_iso / ref_sigma["S1_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S3_iso / ref_sigma["S3_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S3_iso / ref_sigma["S3_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S3p_iso / ref_sigma["S3p_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S3p_iso / ref_sigma["S3p_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S4_iso / ref_sigma["S4_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S4_iso / ref_sigma["S4_sigma"]),
                                [N_image, -1],
                            ),
                        ),
                        axis=-1,
                    )
                else:
                    for_synthesis = self.backend.backend.cat(
                        (
                            mean_data / std_data,
                            std_data,
                            self.backend.bk_reshape(
                                self.backend.bk_real(S2_iso), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S1_iso), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S3_iso), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S3_iso), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S3p_iso), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S3p_iso), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S4_iso), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S4_iso), [N_image, -1]
                            ),
                        ),
                        axis=-1,
                    )
            else:
                if ref_sigma is not None:
                    for_synthesis = self.backend.backend.cat(
                        (
                            mean_data / ref_sigma["std_data"],
                            std_data / ref_sigma["std_data"],
                            self.backend.bk_reshape(
                                self.backend.bk_real(S2 / ref_sigma["S2_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S1 / ref_sigma["S1_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S3 / ref_sigma["S3_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S3 / ref_sigma["S3_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S3p / ref_sigma["S3p_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S3p / ref_sigma["S3p_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S4 / ref_sigma["S4_sigma"]),
                                [N_image, -1],
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S4 / ref_sigma["S4_sigma"]),
                                [N_image, -1],
                            ),
                        ),
                        axis=-1,
                    )
                else:
                    for_synthesis = self.backend.bk_concat(
                        (
                            mean_data / std_data,
                            std_data,
                            self.backend.bk_reshape(
                                self.backend.bk_real(S2), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S1), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S3), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S3), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S3p), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S3p), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_real(S4), [N_image, -1]
                            ),
                            self.backend.bk_reshape(
                                self.backend.bk_imag(S4), [N_image, -1]
                            ),
                        ),
                        axis=-1,
                    )

        if not use_ref:
            self.ref_scattering_cov_S2 = S2

        if get_variance:
            return for_synthesis, ref_sigma

        return for_synthesis

    def purge_edge_mask(self):

        list_edge = []
        for k in self.edge_masks:
            list_edge.append(k)
        for k in list_edge:
            del self.edge_masks[k]

        self.edge_masks = {}

    def to_gaussian(self, x, in_mask=None):
        from scipy.interpolate import interp1d
        from scipy.stats import norm

        if in_mask is not None:
            m_idx = np.where(in_mask.flatten() > 0)[0]
            idx = np.argsort(x.flatten()[m_idx])
            p = norm.ppf((np.arange(1, idx.shape[0] + 1) - 0.5) / idx.shape[0])
            im_target = x.flatten()
            im_target[m_idx[idx]] = p

            self.f_gaussian = interp1d(
                im_target[m_idx[idx]], x.flatten()[m_idx[idx]], kind="cubic"
            )
            self.val_min = im_target[m_idx][idx[0]]
            self.val_max = im_target[m_idx][idx[-1]]
        else:
            idx = np.argsort(x.flatten())
            p = (np.arange(1, idx.shape[0] + 1) - 0.5) / idx.shape[0]
            im_target = x.flatten()
            im_target[idx] = norm.ppf(p)

            # Interpolation cubique
            self.f_gaussian = interp1d(im_target[idx], x.flatten()[idx], kind="cubic")
            self.val_min = im_target[idx[0]]
            self.val_max = im_target[idx[-1]]
        return im_target.reshape(x.shape)

    def from_gaussian(self, x):

        x = self.backend.bk_clip_by_value(x,
                                          self.val_min+1E-7*(self.val_max-self.val_min),
                                          self.val_max-1E-7*(self.val_max-self.val_min))
        return self.f_gaussian(self.backend.to_numpy(x))

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
            result = (
                self.backend.bk_reduce_sum(self.backend.bk_abs(x.S0))
                + self.backend.bk_reduce_sum(self.backend.bk_abs(x.S2))
                + self.backend.bk_reduce_sum(self.backend.bk_abs(x.S3))
                + self.backend.bk_reduce_sum(self.backend.bk_abs(x.S4))
            )

            N = (
                self.backend.bk_size(x.S0)
                + self.backend.bk_size(x.S2)
                + self.backend.bk_size(x.S3)
                + self.backend.bk_size(x.S4)
            )

            if x.S1 is not None:
                result = result + self.backend.bk_reduce_sum(self.backend.bk_abs(x.S1))
                N = N + self.backend.bk_size(x.S1)
            if x.S3P is not None:
                result = result + self.backend.bk_reduce_sum(self.backend.bk_abs(x.S3P))
                N = N + self.backend.bk_size(x.S3P)
            return result / self.backend.bk_cast(N)
        else:
            return self.backend.bk_reduce_mean(x, axis=0)

    def reduce_mean_batch(self, x):

        if isinstance(x, scat_cov):

            sS0 = self.backend.bk_reduce_mean(x.S0, axis=0)
            sS2 = self.backend.bk_reduce_mean(x.S2, axis=0)
            sS3 = self.backend.bk_reduce_mean(x.S3, axis=0)
            sS4 = self.backend.bk_reduce_mean(x.S4, axis=0)
            sS1 = None
            sS3P = None
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

            sS0 = self.backend.bk_reduce_sum(x.S0, axis=0)
            sS2 = self.backend.bk_reduce_sum(x.S2, axis=0)
            sS3 = self.backend.bk_reduce_sum(x.S3, axis=0)
            sS4 = self.backend.bk_reduce_sum(x.S4, axis=0)
            sS1 = None
            sS3P = None
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
                tmp = self.diff_data(x,y)
            else:
                tmp = self.diff_data(x,y,sigma=sigma)
                
            # do abs in case of complex values
            return tmp/x.shape[0]

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

    @tf_function
    def eval_comp_fast(
        self,
        image1,
        image2=None,
        mask=None,
        norm=None,
        cmat=None,
        cmat2=None,
    ):

        res = self.eval(image1, image2=image2, mask=mask, cmat=cmat, cmat2=cmat2)
        return res.S0, res.S2, res.S1, res.S3, res.S4, res.S3P

    def eval_fast(
        self,
        image1,
        image2=None,
        mask=None,
        norm=None,
        cmat=None,
        cmat2=None,
    ):
        s0, s2, s1, s3, s4, s3p = self.eval_comp_fast(
            image1, image2=image2, mask=mask, cmat=cmat, cmat2=cmat2
        )
        return scat_cov(
            s0, s2, s3, s4, s1=s1, s3p=s3p, backend=self.backend, use_1D=self.use_1D
        )
    def calc_matrix_orientation(self,noise_map,image2=None):
        # Décalage circulaire par matrice de permutation
        def circ_shift_matrix(N,k):
            return np.roll(np.eye(N), shift=-k, axis=1)
        Norient = self.NORIENT
        im=self.convol(noise_map)
        if image2 is None:
            mm=np.mean(abs(self.backend.to_numpy(im)),0)
        else:
            im2=self.convol(self.backend.bk_cast(image2))
            mm=np.mean(self.backend.to_numpy(
                self.backend.bk_L1(im*self.backend.bk_conjugate(im2))).real,0)
        
        Norient=mm.shape[0]
        xx=np.cos(np.arange(Norient)/Norient*2*np.pi)
        yy=np.sin(np.arange(Norient)/Norient*2*np.pi)

        a=np.sum(mm*xx[:,None],0)
        b=np.sum(mm*yy[:,None],0)
        
        o=np.fmod(Norient*np.arctan2(-b,a)/(2*np.pi)+Norient,Norient)
        xx=np.arange(Norient)
        alpha = o[None,:]-xx[:,None]
        beta = np.fmod(1+o[None,:]-xx[:,None],Norient)
        alpha=(1-alpha)*(alpha<1)*(alpha>0)+beta*(beta<1)*(beta>0)
        
        m=np.zeros([Norient,Norient,mm.shape[1]])
        for k in range(Norient):
            m[k,:,:]=np.roll(alpha,k,0)
        #m=np.mean(m,0)
        return self.backend.bk_cast(m)
        
    def synthesis(
        self,
        image_target,
        reference=None,
        nstep=4,
        seed=1234,
        Jmax=None,
        edge=False,
        to_gaussian=True,
        use_variance=True,
        synthesised_N=1,
        input_image=None,
        grd_mask=None,
        in_mask=None,
        iso_ang=False,
        EVAL_FREQUENCY=100,
        NUM_EPOCHS=300,
    ):

        import time

        import foscat.Synthesis as synthe

        l_edge = edge
        if in_mask is not None:
            l_edge = True

        if edge:
            self.purge_edge_mask()

        def The_loss_ref_image(u, scat_operator, args):
            input_image = args[0]
            mask = args[1]

            loss = 1e-3 * scat_operator.backend.bk_reduce_mean(
                scat_operator.backend.bk_square(mask * (input_image - u))
            )
            return loss

        def The_loss(u, scat_operator, args):
            ref = args[0]
            sref = args[1]
            use_v = args[2]
            ljmax = args[3]

            # compute scattering covariance of the current synthetised map called u
            if use_v:
                learn = scat_operator.reduce_mean_batch(
                    scat_operator.scattering_cov(
                        u,
                        edge=l_edge,
                        Jmax=ljmax,
                        ref_sigma=sref,
                        use_ref=True,
                        iso_ang=iso_ang,
                    )
                )
            else:
                learn = scat_operator.reduce_mean_batch(
                    scat_operator.scattering_cov(
                        u, edge=l_edge, Jmax=ljmax, use_ref=True, iso_ang=iso_ang
                    )
                )

            # make the difference withe the reference coordinates
            loss = scat_operator.backend.bk_reduce_mean(
                scat_operator.backend.bk_square(learn - ref)
            )
            return loss

        def The_lossH(u, scat_operator, args):
            ref = args[0]
            sref = args[1]
            use_v = args[2]
            ljmax = args[3]

            learn = scat_operator.eval(
                u,
                Jmax=ljmax,
                norm='auto'
                )
            
            if synthesised_N>1:
                learn = scat_operator.reduce_mean_batch(learn)
            
            # compute scattering covariance of the current synthetised map called u
            if use_v:
                loss = scat_operator.reduce_distance(learn,ref,sigma=sref)
            else:
                loss = scat_operator.reduce_distance(learn,ref)

            return loss
        
        def The_lossX(u, scat_operator, args):
            ref = args[0]
            sref = args[1]
            use_v = args[2]
            im2 = args[3]
            ljmax = args[4]

            # compute scattering covariance of the current synthetised map called u
            if use_v:
                learn = scat_operator.reduce_mean_batch(
                    scat_operator.scattering_cov(
                        u,
                        data2=im2,
                        edge=l_edge,
                        Jmax=ljmax,
                        ref_sigma=sref,
                        use_ref=True,
                        iso_ang=iso_ang,
                    )
                )
            else:
                learn = scat_operator.reduce_mean_batch(
                    scat_operator.scattering_cov(
                        u,
                        data2=im2,
                        edge=l_edge,
                        Jmax=ljmax,
                        use_ref=True,
                        iso_ang=iso_ang,
                    )
                )

            # make the difference withe the reference coordinates
            loss = scat_operator.backend.bk_reduce_mean(
                scat_operator.backend.bk_square(learn - ref)
            )
            return loss

        if to_gaussian:
            # Change the data histogram to gaussian distribution
            im_target = self.to_gaussian(image_target, in_mask=in_mask)
        else:
            im_target = image_target

        axis = len(im_target.shape) - 1
        if self.use_2D:
            axis -= 1
        if axis == 0:
            im_target = self.backend.bk_expand_dims(im_target, 0)

        # compute the number of possible steps
        if self.use_2D:
            jmax = int(
                np.min([np.log(im_target.shape[1]), np.log(im_target.shape[2])])
                / np.log(2)
            )
        elif self.use_1D:
            jmax = int(np.log(im_target.shape[1]) / np.log(2))
        else:
            jmax = int((np.log(im_target.shape[1] // 12) / np.log(2)) / 2)
            nside = 2**jmax

        if nstep > jmax - 1:
            nstep = jmax - 1

        t1 = time.time()
        tmp = {}

        l_grd_mask = {}
        l_in_mask = {}
        l_input_image = {}
        l_ref = {}
        l_jmax = {}

        tmp[nstep - 1] = self.backend.bk_cast(im_target)
        l_jmax[nstep - 1] = Jmax

        if reference is not None:
            l_ref[nstep - 1] = self.backend.bk_cast(reference)
        else:
            l_ref[nstep - 1] = None

        if grd_mask is not None:
            l_grd_mask[nstep - 1] = self.backend.bk_cast(grd_mask)
        else:
            l_grd_mask[nstep - 1] = None
        if in_mask is not None:
            l_in_mask[nstep - 1] = in_mask
        else:
            l_in_mask[nstep - 1] = None

        if input_image is not None:
            l_input_image[nstep - 1] = input_image

        for ell in range(nstep - 2, -1, -1):
            tmp[ell], _ = self.ud_grade_2(tmp[ell + 1], axis=1)

            if grd_mask is not None:
                l_grd_mask[ell], _ = self.ud_grade_2(l_grd_mask[ell + 1], axis=1)
            else:
                l_grd_mask[ell] = None

            if in_mask is not None:
                l_in_mask[ell], _ = self.ud_grade_2(l_in_mask[ell + 1])
                l_in_mask[ell] = self.backend.to_numpy(l_in_mask[ell])
            else:
                l_in_mask[ell] = None

            if input_image is not None:
                l_input_image[ell], _ = self.ud_grade_2(l_input_image[ell + 1], axis=1)

            if reference is not None:
                l_ref[ell], _ = self.ud_grade_2(l_ref[ell + 1], axis=1)
            else:
                l_ref[ell] = None

            if l_jmax[ell + 1] is None:
                l_jmax[ell] = None
            else:
                l_jmax[ell] = l_jmax[ell + 1] - 1

        if not self.use_2D and not self.use_1D:
            l_nside = nside // (2 ** (nstep - 1))

        for k in range(nstep):
            if k == 0:
                if input_image is None:
                    np.random.seed(seed)
                    if self.use_2D:
                        imap = self.backend.bk_cast(
                            np.random.randn(
                                synthesised_N, tmp[k].shape[1], tmp[k].shape[2]
                            )
                        )
                    else:
                        imap = self.backend.bk_cast(
                            np.random.randn(synthesised_N, tmp[k].shape[1])
                        )
                else:
                    if self.use_2D:
                        imap = self.backend.bk_reshape(
                            self.backend.bk_tile(
                                self.backend.bk_cast(l_input_image[k].flatten()),
                                synthesised_N,
                            ),
                            [synthesised_N, tmp[k].shape[1], tmp[k].shape[2]],
                        )
                    else:
                        imap = self.backend.bk_reshape(
                            self.backend.bk_tile(
                                self.backend.bk_cast(l_input_image[k].flatten()),
                                synthesised_N,
                            ),
                            [synthesised_N, tmp[k].shape[1]],
                        )
            else:
                # Increase the resolution between each step
                if self.use_2D:
                    imap = self.up_grade(
                        omap,
                        imap.shape[1] * 2,
                        axis=-2,
                        nouty=imap.shape[2] * 2
                    )
                elif self.use_1D:
                    imap = self.up_grade(omap, imap.shape[1] * 2)
                else:
                    imap = self.up_grade(omap, l_nside)
                    
            if grd_mask is not None:
                imap = imap * l_grd_mask[k] + tmp[k] * (1 - l_grd_mask[k])

            
            if self.use_2D:
                # compute the coefficients for the target image
                if use_variance:
                    ref, sref = self.scattering_cov(
                        tmp[k],
                        data2=l_ref[k],
                        get_variance=True,
                        edge=l_edge,
                        Jmax=l_jmax[k],
                        in_mask=l_in_mask[k],
                        iso_ang=iso_ang,
                    )
                else:
                    ref = self.scattering_cov(
                        tmp[k],
                        data2=l_ref[k],
                        in_mask=l_in_mask[k],
                        edge=l_edge,
                        Jmax=l_jmax[k],
                        iso_ang=iso_ang,
                    )
                    sref = ref
            else:
                self.clean_norm()
                
                ref = self.eval(
                        tmp[k],
                        image2=l_ref[k],
                        mask=l_in_mask[k],
                        Jmax=l_jmax[k],
                        norm='auto'
                    )
                
                # compute the coefficients for the target image
                if use_variance:
                    ref, sref = self.eval(
                        tmp[k],
                        image2=l_ref[k],
                        mask=l_in_mask[k],
                        Jmax=l_jmax[k],
                        calc_var=True,
                        norm='auto'
                    )
                else:
                    ref = self.eval(
                        tmp[k],
                        image2=l_ref[k],
                        mask=l_in_mask[k],
                        Jmax=l_jmax[k],
                        norm='auto'
                    )
                    sref = ref
                    
                if iso_ang:
                    ref=ref.iso_mean()
                    sref=sref.iso_mean()

            # compute the mean of the population does nothing if only one map is given
            ref = self.reduce_mean_batch(ref)

            if l_in_mask[k] is not None:
                self.purge_edge_mask()

            if l_ref[k] is None:
                if self.use_2D:
                    # define a loss to minimize
                    loss = synthe.Loss(The_loss, self, ref, sref, use_variance, l_jmax[k])
                else:
                    loss = synthe.Loss(The_lossH, self, ref, sref, use_variance, l_jmax[k])
            else:
                # define a loss to minimize
                if self.use_2D:
                    loss = synthe.Loss(
                        The_lossX, self, ref, sref, use_variance, l_ref[k], l_jmax[k]
                    )
                else:
                    loss = synthe.Loss(
                        The_lossXH, self, ref, sref, use_variance, l_ref[k], l_jmax[k]
                    )

            if input_image is not None:
                # define a loss to minimize
                loss_input = synthe.Loss(
                    The_loss_ref_image,
                    self,
                    self.backend.bk_cast(l_input_image[k]),
                    self.backend.bk_cast(l_in_mask[k]),
                )

                sy = synthe.Synthesis([loss])  # ,loss_input])
            else:
                sy = synthe.Synthesis([loss])

            # initialize the synthesised map
            if self.use_2D:
                print("Synthesis scale [ %d x %d ]" % (imap.shape[1], imap.shape[2]))
            elif self.use_1D:
                print("Synthesis scale [ %d ]" % (imap.shape[1]))
            else:
                print("Synthesis scale nside=%d" % (l_nside))
                l_nside *= 2

            # do the minimization
            omap = sy.run(
                imap,
                EVAL_FREQUENCY=EVAL_FREQUENCY,
                NUM_EPOCHS=NUM_EPOCHS,
                grd_mask=l_grd_mask[k],
            )

        if not self.use_2D:
            self.clean_norm()
            
        t2 = time.time()
        print("Total computation %.2fs" % (t2 - t1))

        if to_gaussian:
            omap = self.from_gaussian(omap)

        if axis == 0 and synthesised_N == 1:
            return omap[0]
        else:
            return omap

    def to_numpy(self, x):
        return self.backend.to_numpy(x)
