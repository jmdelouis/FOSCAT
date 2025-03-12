class BackendBase:
    
    def __init__(self, name, mpi_rank=0, all_type="float64", gpupos=0, silent=False):
        
        self.BACKEND=name
        self.mpi_rank=mpi_rank
        self.all_type=all_type
        self.gpupos=gpupos
        self.silent=silent
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
        
    # ---------------------------------------------−---------
    # --             BACKEND DEFINITION                    --
    # ---------------------------------------------−---------
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
            
    def bk_fftn(self, data,dim=None):
        raise NotImplementedError("This is an abstract class.")

    def bk_ifftn(self, data,dim=None,norm=None):
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

    def bk_clip_by_value(self, x,xmin,xmax):
        raise NotImplementedError("This is an abstract class.")

    def bk_cast(self, x):
        raise NotImplementedError("This is an abstract class.")
            
    def bk_variable(self,x):
        raise NotImplementedError("This is an abstract class.")
        
    def bk_assign(self,x,y):
        raise NotImplementedError("This is an abstract class.")
            
    def bk_constant(self,x):
        raise NotImplementedError("This is an abstract class.")
        
    def to_numpy(self,x):
        raise NotImplementedError("This is an abstract class.")
