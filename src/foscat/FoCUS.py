import os
import sys

import healpy as hp
import numpy as np
import foscat.HealSpline as HS
from scipy.interpolate import griddata

TMPFILE_VERSION = "V7_0"


class FoCUS:
    def __init__(
            self,
            NORIENT=4,
            LAMBDA=1.2,
            KERNELSZ=3,
            slope=1.0,
            all_type="float32",
            nstep_max=20,
            padding="SAME",
            gpupos=0,
            mask_thres=None,
            mask_norm=False,
            isMPI=False,
            TEMPLATE_PATH=None,
            BACKEND="torch",
            use_2D=False,
            use_1D=False,
            return_data=False,
            JmaxDelta=0,
            DODIV=False,
            InitWave=None,
            silent=True,
            mpi_size=1,
            mpi_rank=0
    ):

        self.__version__ = "2025.08.4"
        # P00 coeff for normalization for scat_cov
        self.TMPFILE_VERSION = TMPFILE_VERSION
        self.P1_dic = None
        self.P2_dic = None
        self.isMPI = isMPI
        self.mask_thres = mask_thres
        self.mask_norm = mask_norm
        self.InitWave = InitWave
        self.mask_mask = None
        self.mpi_size = mpi_size
        self.mpi_rank = mpi_rank
        self.return_data = return_data
        self.silent = silent

        self.kernel_smooth = {}
        self.padding_smooth = {}
        self.kernelR_conv = {}
        self.kernelI_conv = {}
        self.padding_conv = {}

        if not self.silent:
            print("================================================")
            print("          START FOSCAT CONFIGURATION")
            print("================================================")
            sys.stdout.flush()

        home_dir = os.environ["HOME"]
        
        if TEMPLATE_PATH is None:
            TEMPLATE_PATH=home_dir+"/.FOSCAT/data"
            
        self.TEMPLATE_PATH = TEMPLATE_PATH
        if not os.path.exists(self.TEMPLATE_PATH):
            if not self.silent:
                print(
                    "The directory %s to store temporary information for FoCUS does not exist: Try to create it"
                    % (self.TEMPLATE_PATH)
                )
            try:
                os.system("mkdir -p %s" % (self.TEMPLATE_PATH))
                if not self.silent:
                    print("The directory %s is created")
            except:
                print("Impossible to create the directory %s" % (self.TEMPLATE_PATH))
                return None

        self.number_of_loss = 0

        self.history = np.zeros([10])
        self.nlog = 0
        self.padding = padding

        if JmaxDelta != 0:
            print(
                "OPTION JmaxDelta is not avialable anymore after version 3.6.2. Please use Jmax option in eval function"
            )
            return None

        self.OSTEP = JmaxDelta
        self.use_2D = use_2D
        self.use_1D = use_1D

        if isMPI:
            from mpi4py import MPI

            self.comm = MPI.COMM_WORLD
            if all_type == "float32":
                self.MPI_ALL_TYPE = MPI.FLOAT
            else:
                self.MPI_ALL_TYPE = MPI.DOUBLE
        else:
            self.MPI_ALL_TYPE = None

        self.all_type = all_type
        self.BACKEND = BACKEND

        if BACKEND == "torch":
            from foscat.BkTorch import BkTorch

            self.backend = BkTorch(
                all_type=all_type,
                mpi_rank=mpi_rank,
                gpupos=gpupos,
                silent=self.silent,
            )
        elif BACKEND == "tensorflow":
            from foscat.BkTensorflow import BkTensorflow

            self.backend = BkTensorflow(
                all_type=all_type,
                mpi_rank=mpi_rank,
                gpupos=gpupos,
                silent=self.silent,
            )
        else:
            from foscat.BkNumpy import BkNumpy

            self.backend = BkNumpy(
                all_type=all_type,
                mpi_rank=mpi_rank,
                gpupos=gpupos,
                silent=self.silent,
            )

        self.all_bk_type = self.backend.all_bk_type
        self.all_cbk_type = self.backend.all_cbk_type
        self.gpulist = self.backend.gpulist
        self.ngpu = self.backend.ngpu
        self.rank = mpi_rank

        self.gpupos = (gpupos + mpi_rank) % self.backend.ngpu

        if not self.silent:
            print("============================================================")
            print("==                                                        ==")
            print("==                                                        ==")
            print(
                "==     RUN ON GPU Rank %d : %s                          =="
                % (mpi_rank, self.gpulist[self.gpupos % self.ngpu])
            )
            print("==                                                        ==")
            print("==                                                        ==")
            print("============================================================")
            sys.stdout.flush()

        l_NORIENT = NORIENT
        if DODIV:
            l_NORIENT = NORIENT + 2

        self.NORIENT = l_NORIENT
        self.LAMBDA = LAMBDA
        self.slope = slope

        self.R_off = (KERNELSZ - 1) // 2
        if (self.R_off // 2) * 2 < self.R_off:
            self.R_off += 1

        self.ww_Real = {}
        self.ww_Imag = {}
        self.ww_CNN_Transpose = {}
        self.ww_CNN = {}
        self.X_CNN = {}
        self.Y_CNN = {}
        self.Z_CNN = {}

        self.Idx_CNN = {}
        self.Idx_WCNN = {}
        
        self.filters_set = {}
        self.edge_masks = {}

        wwc = np.zeros([l_NORIENT, KERNELSZ**2]).astype(all_type)
        wws = np.zeros([l_NORIENT, KERNELSZ**2]).astype(all_type)

        x = np.repeat(np.arange(KERNELSZ) - KERNELSZ // 2, KERNELSZ).reshape(
            KERNELSZ, KERNELSZ
        )
        y = x.T

        if NORIENT == 1:
            xx = (3 / float(KERNELSZ)) * LAMBDA * x
            yy = (3 / float(KERNELSZ)) * LAMBDA * y

            if KERNELSZ == 5:
                # w_smooth=np.exp(-2*((3.0/float(KERNELSZ)*xx)**2+(3.0/float(KERNELSZ)*yy)**2))
                w_smooth = np.exp(-(xx**2 + yy**2))
                tmp = np.exp(-2 * (xx**2 + yy**2)) - 0.25 * np.exp(
                    -0.5 * (xx**2 + yy**2)
                )
            else:
                w_smooth = np.exp(-0.5 * (xx**2 + yy**2))
                tmp = np.exp(-2 * (xx**2 + yy**2)) - 0.25 * np.exp(
                    -0.5 * (xx**2 + yy**2)
                )

            wwc[0] = tmp.flatten() - tmp.mean()
            tmp = 0 * w_smooth
            wws[0] = tmp.flatten()
            sigma = np.sqrt((wwc[:, 0] ** 2).mean())
            wwc[0] /= sigma
            wws[0] /= sigma

            w_smooth = w_smooth.flatten()
        else:
            for i in range(NORIENT):
                a = (
                    (NORIENT - 1 - i) / float(NORIENT) * np.pi
                )  # get the same angle number than scattering lib
                if KERNELSZ < 5:
                    xx = (
                        (3 / float(KERNELSZ)) * LAMBDA * (x * np.cos(a) + y * np.sin(a))
                    )
                    yy = (
                        (3 / float(KERNELSZ)) * LAMBDA * (x * np.sin(a) - y * np.cos(a))
                    )
                else:
                    xx = (3 / 5) * LAMBDA * (x * np.cos(a) + y * np.sin(a))
                    yy = (3 / 5) * LAMBDA * (x * np.sin(a) - y * np.cos(a))
                if KERNELSZ == 5:
                    w_smooth = np.exp(
                        -2
                        * (
                            (3.0 / float(KERNELSZ) * xx) ** 2
                            + (3.0 / float(KERNELSZ) * yy) ** 2
                        )
                    )
                else:
                    w_smooth = np.exp(-0.5 * (xx**2 + yy**2))
                tmp1 = np.cos(yy * np.pi) * w_smooth
                tmp2 = np.sin(yy * np.pi) * w_smooth

                wwc[i] = tmp1.flatten() - tmp1.mean()
                wws[i] = tmp2.flatten() - tmp2.mean()
                # sigma = np.sqrt((wwc[:, i] ** 2).mean())
                sigma = np.mean(w_smooth)
                wwc[i] /= sigma
                wws[i] /= sigma

                if DODIV and i == 0:
                    r = xx**2 + yy**2
                    theta = np.arctan2(yy, xx)
                    theta[KERNELSZ // 2, KERNELSZ // 2] = 0.0
                    tmp1 = r * np.cos(2 * theta) * w_smooth
                    tmp2 = r * np.sin(2 * theta) * w_smooth

                    wwc[NORIENT] = tmp1.flatten() - tmp1.mean()
                    wws[NORIENT] = tmp2.flatten() - tmp2.mean()
                    # sigma = np.sqrt((wwc[:, NORIENT] ** 2).mean())
                    sigma = np.mean(w_smooth)

                    wwc[NORIENT] /= sigma
                    wws[NORIENT] /= sigma
                    tmp1 = r * np.cos(2 * theta + np.pi)
                    tmp2 = r * np.sin(2 * theta + np.pi)

                    wwc[NORIENT + 1] = tmp1.flatten() - tmp1.mean()
                    wws[NORIENT + 1] = tmp2.flatten() - tmp2.mean()
                    # sigma = np.sqrt((wwc[:, NORIENT + 1] ** 2).mean())
                    sigma = np.mean(w_smooth)
                    wwc[NORIENT + 1] /= sigma
                    wws[NORIENT + 1] /= sigma

                w_smooth = w_smooth.flatten()

        if self.use_1D:
            KERNELSZ = 5

        self.KERNELSZ = KERNELSZ

        self.Idx_Neighbours = {}
        self.w_smooth = {}

        
        if self.use_1D:
            self.w_smooth = slope * (w_smooth / w_smooth.sum()).astype(self.all_type)
            self.ww_RealT = {}
            self.ww_ImagT = {}
            self.ww_SmoothT = {}
            if KERNELSZ == 5:
                xx = np.arange(5) - 2
                w = np.exp(-0.25 * (xx) ** 2)
                c = w * np.cos((xx) * np.pi / 2)
                s = w * np.sin((xx) * np.pi / 2)

                w = w / np.sum(w)
                c = c - np.mean(c)
                s = s - np.mean(s)
                r = np.sum(np.sqrt(c * c + s * s))
                c = c / r
                s = s / r
                self.ww_RealT[1] = self.backend.bk_cast(
                    self.backend.bk_constant(np.array(c).reshape(xx.shape[0]))
                )
                self.ww_ImagT[1] = self.backend.bk_cast(
                    self.backend.bk_constant(np.array(s).reshape(xx.shape[0]))
                )
                self.ww_SmoothT[1] = self.backend.bk_cast(
                    self.backend.bk_constant(np.array(w).reshape(xx.shape[0]))
                )

        if self.use_2D:
            self.w_smooth = slope * (w_smooth / w_smooth.sum()).astype(self.all_type)
            self.ww_RealT = {}
            self.ww_ImagT = {}
            self.ww_SmoothT = {}

            self.ww_SmoothT[1] = self.backend.bk_constant(
                self.w_smooth.reshape(1, KERNELSZ, KERNELSZ)
            )
            self.ww_RealT[1] = self.backend.bk_constant(
                self.backend.bk_reshape(
                    wwc.astype(self.all_type), [NORIENT, KERNELSZ, KERNELSZ]
                )
            )
            self.ww_ImagT[1] = self.backend.bk_constant(
                self.backend.bk_reshape(
                    wws.astype(self.all_type), [NORIENT, KERNELSZ, KERNELSZ]
                )
            )

            def doorientw(x):
                y = np.zeros(
                    [KERNELSZ, KERNELSZ, NORIENT, NORIENT * NORIENT],
                    dtype=self.all_type,
                )
                for k in range(NORIENT):
                    y[:, :, k, k * NORIENT : k * NORIENT + NORIENT] = x.reshape(
                        KERNELSZ, KERNELSZ, NORIENT
                    )
                return y

            self.ww_RealT[NORIENT] = self.backend.bk_constant(
                doorientw(wwc.astype(self.all_type))
            )
            self.ww_ImagT[NORIENT] = self.backend.bk_constant(
                doorientw(wws.astype(self.all_type))
            )
        self.pix_interp_val = {}
        self.weight_interp_val = {}
        self.ring2nest = {}
        self.ampnorm = {}

        self.loss = {}

        self.dtype_dcode_map = {
            0: np.int64,
            1: np.int32,
            2: np.float32,
            3: np.float64,
            4: np.complex64,
            5: np.complex128
        }
        self.dtype_code_map = {
            np.int64: 0,
            np.int32: 1,
            np.float32: 2,
            np.float64: 3,
            np.complex64: 4,
            np.complex128: 5
        }

    # this is for the storage only
    def get_dtype_code(self, dtype):
        for key, code in self.dtype_code_map.items():
            if np.dtype(dtype) == np.dtype(key):
                return code
        raise ValueError(f"Unsupported data type: {dtype}")
    
    def get_type(self):
        return self.all_type

    def get_mpi_type(self):
        return self.MPI_ALL_TYPE

    # ---------------------------------------------−---------
    # --       COMPUTE 3X3 INDEX FOR HEALPIX WORK          --
    # ---------------------------------------------−---------
    def conv_to_FoCUS(self, x, axis=0):
        if self.use_2D and isinstance(x, np.ndarray):
            return self.to_R(x, axis, chans=self.chans)
        return x

    def diffang(self, a, b):
        return np.arctan2(np.sin(a) - np.sin(b), np.cos(a) - np.cos(b))

    def corr_idx_wXX(self, x, y):
        idx = np.where(x == -1)[0]
        res = x
        res[idx] = y[idx]
        return res

    # ---------------------------------------------−---------
    # make the CNN working : index reporjection of the kernel on healpix

    def calc_indices_convol(self, nside, kernel, rotation=None):
        to, po = hp.pix2ang(nside, np.arange(12 * nside * nside), nest=True)
        x, y, z = hp.pix2vec(nside, np.arange(12 * nside * nside), nest=True)

        idx = np.argsort((x - 1.0) ** 2 + y**2 + z**2)[0:kernel]
        x0, y0, z0 = hp.pix2vec(nside, idx[0], nest=True)
        t0, p0 = hp.pix2ang(nside, idx[0], nest=True)

        idx = np.argsort((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)[0:kernel]
        im = np.ones([12 * nside**2]) * -1
        im[idx] = np.arange(len(idx))

        xc, yc, zc = hp.pix2vec(nside, idx, nest=True)

        xc -= x0
        yc -= y0
        zc -= z0

        vec = np.concatenate(
            [np.expand_dims(x, -1), np.expand_dims(y, -1), np.expand_dims(z, -1)], 1
        )

        indices = np.zeros([12 * nside**2 * 250, 2], dtype="int")
        weights = np.zeros([12 * nside**2 * 250])
        nn = 0
        for k in range(12 * nside * nside):
            if k % (nside * nside) == nside * nside - 1:
                print(
                    "Nside=%d KenelSZ=%d %.2f%%"
                    % (nside, kernel, k / (12 * nside**2) * 100)
                )
            if nside < 4:
                idx2 = np.arange(12 * nside**2)
            else:
                idx2 = hp.query_disc(
                    nside, vec[k], np.pi / nside, inclusive=True, nest=True
                )
            t2, p2 = hp.pix2ang(nside, idx2, nest=True)
            if rotation is None:
                rot = [po[k] / np.pi * 180.0, (t0 - to[k]) / np.pi * 180.0]
            else:
                rot = [po[k] / np.pi * 180.0, (t0 - to[k]) / np.pi * 180.0, rotation[k]]

            r = hp.Rotator(rot=rot)
            t2, p2 = r(t2, p2)

            ii, ww = hp.get_interp_weights(nside, t2, p2, nest=True)

            ii = im[ii]

            for l_rotation in range(4):
                iii = np.where(ii[l_rotation] != -1)[0]
                if len(iii) > 0:
                    indices[nn : nn + len(iii), 1] = idx2[iii]
                    indices[nn : nn + len(iii), 0] = k * kernel + ii[l_rotation, iii]
                    weights[nn : nn + len(iii)] = ww[l_rotation, iii]
                    nn += len(iii)

        indices = indices[0:nn]
        weights = weights[0:nn]
        if k % (nside * nside) == nside * nside - 1:
            print(
                "Nside=%d KenelSZ=%d Total Number of value=%d Ratio of the matrix %.2g%%"
                % (
                    nside,
                    kernel,
                    nn,
                    100 * nn / (kernel * 12 * nside**2 * 12 * nside**2),
                )
            )
        return indices, weights, xc, yc, zc

    #======================================================================================
    # The next two functions prepare the ability of FOSCAT to work with large indexed file
    #======================================================================================
    
    def save_index(self, filepath, data, offset=0, count=None):
        """
        Save an N-dimensional NumPy array with shape (N, ...) to binary file.
        A 12x int64 header is written, describing dtype and shape beyond axis 0.

        Header layout (12 x int64):
        [0] = dtype code (0=int64, 1=int32, 2=float32, 3=float64, 4=complex64, 5=complex128)
        [1] = number of extra dimensions (i.e., data.ndim - 1)
        [2:12] = shape[1:] padded with zeros

        Parameters:
        - filepath: target binary file path
        - data: NumPy array with shape (N, ...)
        - offset: number of items to skip on axis 0
        - count: number of items to write on axis 0 (default: rest of the array)
        """
        if filepath is None:
            raise ValueError("No filepath specified for writing.")

        data = np.asarray(data)
        if data.ndim < 1:
            raise ValueError("Data must have at least one dimension.")

        extra_dims = data.shape[1:]
        if len(extra_dims) > 10:
            raise ValueError(f"Too many dimensions: {data.ndim}. Max supported is 11 (1 + 10 extra).")

        dtype_code = self.get_dtype_code(data.dtype)
        itemsize = data.dtype.itemsize
        item_shape = data.shape[1:]
        item_count = np.prod(item_shape, dtype=np.int64) if item_shape else 1

        if count is None:
            count = data.shape[0]

        header = np.zeros(12, dtype=np.int64)
        header[0] = dtype_code
        header[1] = len(extra_dims)
        header[2:2 + len(extra_dims)] = extra_dims

        mode = 'r+b' if os.path.exists(filepath) else 'w+b'
        with open(filepath, mode) as f:
            if os.path.getsize(filepath) == 0:
                f.write(header.tobytes())

            byte_offset = 12 * 8 + offset * itemsize * item_count  # header is 96 bytes
            f.seek(byte_offset)
            f.write(data[offset:offset + count].tobytes())

    def read_index(self, filepath, offset=0, count=None):
        """
        Load a NumPy array from a binary file with a 12x int64 header.

        Header layout:
        [0] = dtype code
        [1] = number of extra dimensions (D)
        [2:2+D] = shape[1:] of each sample (shape after axis 0)

        Parameters:
        - filepath: path to the binary file
        - offset: number of samples to skip on axis 0
        - count: number of samples to read (default: all remaining)

        Returns:
        - data: NumPy array with shape (count, ...) and correct dtype
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'rb') as f:
            header_bytes = f.read(12 * 8)
            if len(header_bytes) != 96:
                raise ValueError("Invalid or missing header (expected 96 bytes).")

            header = np.frombuffer(header_bytes, dtype=np.int64)
            dtype_code = header[0]
            ndim_extra = header[1]
            if dtype_code not in self.dtype_dcode_map:
                raise ValueError(f"Unknown dtype code in header: {dtype_code}")

            dtype = self.dtype_dcode_map[dtype_code]
            shape1 = tuple(header[2:2 + ndim_extra])
            itemsize = np.dtype(dtype).itemsize
            item_count = np.prod(shape1, dtype=np.int64) if shape1 else 1
            bytes_per_sample = itemsize * item_count

            # Seek to data block
            f.seek(12 * 8 + offset * bytes_per_sample)

            # Determine number of items
            if count is None:
                remaining_bytes = os.path.getsize(filepath) - (12 * 8 + offset * bytes_per_sample)
                count = remaining_bytes // bytes_per_sample

            raw = f.read(count * bytes_per_sample)
            data = np.frombuffer(raw, dtype=dtype)

            if shape1:
                data = data.reshape((count,) + shape1)
            else:
                data = data.reshape((count,))

            return data
    
    # ---------------------------------------------−---------
    # ---------------------------------------------−---------
    def healpix_layer(self, im, ww, indices=None, weights=None):
        #ww [N_i,NORIENT,KERNELSZ*KERNELSZ//2,N_o,NORIENT]
        #im [N_batch,N_i,  NORIENT,N]
        nside=int(np.sqrt(im.shape[-1]//12))
        if indices is None:
            if (nside,self.NORIENT,self.KERNELSZ) not in self.ww_CNN:
                self.init_index_cnn(nside,self.NORIENT)
            indices = self.Idx_CNN[(nside,self.NORIENT,self.KERNELSZ)]
            mat = self.Idx_WCNN[(nside,self.NORIENT,self.KERNELSZ)]
            
        wim = self.backend.bk_gather(im,indices.flatten(),axis=3) #[N_batch,N_i,NORIENT,K*(K+1),N_o,NORIENT,N,N_w]
        
        wim = self.backend.bk_reshape(wim,[im.shape[0],im.shape[1],im.shape[2]]+list(indices.shape))*mat[None,...]
        #win is [N_batch,N_i,  NORIENT,K*(K+1),1,  NORIENT,N,N_w]
        #ww is  [1,      N_i,  NORIENT,K*(K+1),N_o,NORIENT]
        wim = self.backend.bk_reduce_sum(wim[:,:,:,:,None]*ww[None,:,:,:,:,:,None,None],[1,2,3])

        wim = self.backend.bk_reduce_sum(wim,-1)
        return self.backend.bk_reshape(wim,[im.shape[0],ww.shape[3],ww.shape[4],im.shape[-1]])

    # ---------------------------------------------−---------

    # ---------------------------------------------−---------
    def get_rank(self):
        return self.rank

    # ---------------------------------------------−---------
    def get_size(self):
        return self.size

    # ---------------------------------------------−---------
    def barrier(self):
        if self.isMPI:
            self.comm.Barrier()

    # ---------------------------------------------−---------
    def toring(self, image, axis=0):
        lout = int(np.sqrt(image.shape[axis] // 12))

        if lout not in self.ring2nest:
            self.ring2nest[lout] = hp.ring2nest(lout, np.arange(12 * lout**2))

        return image.numpy()[self.ring2nest[lout]]

    # --------------------------------------------------------
    def ud_grade(self, im, j, axis=0, cell_ids=None, nside=None):
        rim = im
        for k in range(j):
            # rim = self.smooth(rim, axis=axis)
            rim = self.ud_grade_2(rim, axis=axis,
                                  cell_ids=cell_ids,
                                  nside=nside)
        return rim

    # --------------------------------------------------------
    def ud_grade_2(self, im, axis=0, cell_ids=None, nside=None):

        if self.use_2D:
            ishape = list(im.shape)
            if len(ishape) < axis + 2:
                if not self.silent:
                    print("Use of 2D scat with data that has less than 2D")
                return None, None

            npix = im.shape[-2]
            npiy = im.shape[-1]

            ndata = 1
            for k in range(len(im.shape)-2):
                ndata = ndata * ishape[k]

            tim = self.backend.bk_reshape(
                self.backend.bk_cast(im), [ndata, npix, npiy, 1]
            )
            tim = self.backend.bk_reshape(
                tim[:, 0 : 2 * (npix // 2), 0 : 2 * (npiy // 2), :],
                [ndata, npix // 2, 2, npiy // 2, 2, 1],
            )

            res = self.backend.bk_reduce_sum(self.backend.bk_reduce_sum(tim, 4), 2) / 4

            if len(ishape) == 2:
                return (
                    self.backend.bk_reshape(
                        res, [npix // 2, npiy // 2]
                    ),
                    None,
                )
            else:
                return (
                    self.backend.bk_reshape(
                        res,
                        ishape[0:-2]
                        + [npix // 2, npiy // 2],
                    ),
                    None,
                )

            return self.backend.bk_reshape(res, [npix // 2, npiy // 2]), None
        elif self.use_1D:
            ishape = list(im.shape)

            npix = ishape[-1]

            ndata = 1
            for k in range(len(ishape) - 1):
                ndata = ndata * ishape[k]

            tim = self.backend.bk_reshape(
                self.backend.bk_cast(im), [ndata, npix // 2, 2]
            )

            res = self.backend.bk_reduce_mean(tim, -1)

            return self.backend.bk_reshape(res, ishape[0:-1] + [npix // 2]), None

        else:
            shape = list(im.shape)
            if cell_ids is not None:
                sim, new_cell_ids = self.backend.binned_mean(im, cell_ids)
                return sim, new_cell_ids
            
            return self.backend.bk_reduce_mean(
                self.backend.bk_reshape(im, shape[0:-1]+[shape[-1]//4,4]), axis=-1
                ),None

    # --------------------------------------------------------
    def up_grade(self, im, nout,
                 axis=-1,
                 nouty=None,
                 cell_ids=None,
                 o_cell_ids=None,
                 nside=None):

        ishape = list(im.shape)
        if self.use_2D:
            if len(ishape) < 2:
                if not self.silent:
                    print("Use of 2D scat with data that has less than 2D")
                return None

            if nouty is None:
                nouty = nout

            if ishape[-2] == nout and ishape[-1] == nouty:
                return im

            npix = im.shape[-2]
            npiy = im.shape[-1]

            ndata = 1
            for k in range(len(im.shape)-2):
                ndata = ndata * ishape[k]

            tim = self.backend.bk_reshape(
                self.backend.bk_cast(im), [ndata, npix, npiy,1]
            )

            res = self.backend.bk_resize_image(tim, [nout, nouty])

            if len(ishape) == 2:
                return self.backend.bk_reshape(res, [nout, nouty])
            else:
                return self.backend.bk_reshape(
                        res, ishape[0:-2] + [nout, nouty] 
                )

        elif self.use_1D:
            if len(ishape) < axis + 1:
                if not self.silent:
                    print("Use of 1D scat with data that has less than 1D")
                return None

            if ishape[axis] == nout:
                return im

            npix = im.shape[axis]
            odata = 1
            
            ndata = 1
            if len(ishape)>1:
                for k in range(len(ishape)-1):
                    ndata = ndata * ishape[k]

            tim = self.backend.bk_reshape(
                self.backend.bk_cast(im), [ndata, npix, odata]
            )

            while tim.shape[1] != nout:
                res2 = self.backend.bk_expand_dims(
                    self.backend.bk_concat(
                        [(tim[:, 1:, :] + 3 * tim[:, :-1, :]) / 4, tim[:, -1:, :]], 1
                    ),
                    -2,
                )
                res1 = self.backend.bk_expand_dims(
                    self.backend.bk_concat(
                        [tim[:, 0:1, :], (tim[:, 1:, :] * 3 + tim[:, :-1, :]) / 4], 1
                    ),
                    -2,
                )
                tim = self.backend.bk_reshape(
                    self.backend.bk_concat([res1, res2], -2),
                    [ndata, tim.shape[1] * 2, odata],
                )
            return self.backend.bk_reshape(tim, ishape[0:-1] + [nout])

        else:
            if nside is None:
                lout = int(np.sqrt(im.shape[-1] // 12))
            else:
                lout = nside
                
            if (lout,nout) not in self.pix_interp_val:
                if not self.silent:
                    print("compute lout nout", lout, nout)
                if cell_ids is None:
                    o_cell_ids=np.arange(12 * nout**2, dtype="int")
                    i_npix=12*lout**2
                
                    #level=int(np.log2(lout)) # nside=128

                    #sp = HS.heal_spline(level,gamma=2.0)

                    th, ph = hp.pix2ang(
                        nout, o_cell_ids, nest=True
                    )
                    
                    all_idx,www=hp.get_interp_weights(lout,th,ph,nest=True)
                                          
                    #www,all_idx,hidx=sp.ang2weigths(th,ph,nest=True)

                    w=www.T
                    p=all_idx.T
                    
                    w=w.flatten()
                    p=p.flatten()
                    
                    indice = np.zeros([o_cell_ids.shape[0] * 4, 2], dtype="int")
                    indice[:, 1] = np.repeat(np.arange(o_cell_ids.shape[0]), 4)
                    indice[:, 0] = p
                    
                    self.pix_interp_val[(lout,nout)] = 1
                    self.weight_interp_val[(lout,nout)] = self.backend.bk_SparseTensor(
                        self.backend.bk_constant(indice),
                        self.backend.bk_constant(self.backend.bk_cast(w)),
                        dense_shape=[i_npix,o_cell_ids.shape[0]],
                    )

                else:
                    ratio=(nout//lout)**2
                    if o_cell_ids is None:
                        o_cell_ids=np.tile(cell_ids,ratio)*ratio+np.repeat(np.arange(ratio),cell_ids.shape[0])
                    i_npix=cell_ids.shape[0]

                    #level=int(np.log2(lout)) # nside=128

                    #sp = HS.heal_spline(level,gamma=2.0)

                    th, ph = hp.pix2ang(
                        nout, o_cell_ids, nest=True
                    )

                    all_idx,www=hp.get_interp_weights(lout,th,ph,nest=True)
                    #www,all_idx,hidx=sp.ang2weigths(th,ph,nest=True)

                    hidx,inv_idx = np.unique(all_idx,
                                    return_inverse=True)
                    all_idx = inv_idx
                    
                    sorter = np.argsort(hidx)
                    
                    index=sorter[np.searchsorted(hidx,
                                                 cell_ids,
                                                 sorter=sorter)]
                    
                    mask        = -np.ones([hidx.shape[0]])
                    
                    mask[index] = np.arange(index.shape[0],dtype='int')

                    all_idx=mask[all_idx]

                    www[all_idx==-1]=0.0
                    www/=np.sum(www,0)[None,:]
                    
                    all_idx[all_idx==-1]=0
               
                    w=www.T
                    p=all_idx.T
                    
                    w=w.flatten()
                    p=p.flatten()
                    
                    indice = np.zeros([o_cell_ids.shape[0] * 4, 2], dtype="int")
                    indice[:, 1] = np.repeat(np.arange(o_cell_ids.shape[0]), 4)
                    indice[:, 0] = p
                    
                    self.pix_interp_val[(lout,nout)] = 1
                    self.weight_interp_val[(lout,nout)] = self.backend.bk_SparseTensor(
                        self.backend.bk_constant(indice),
                        self.backend.bk_constant(self.backend.bk_cast(w)),
                        dense_shape=[i_npix,o_cell_ids.shape[0]],
                    )
                    
                    del w
                    del p

            if lout == nout:
                imout = im
            else:
                # work only on the last column

                ndata = 1
                for k in range(len(ishape)-1):
                    ndata = ndata * ishape[k]
                tim = self.backend.bk_reshape(
                    self.backend.bk_cast(im), [ndata, ishape[-1]]
                )
                if tim.dtype == self.all_cbk_type:
                    rr = self.backend.bk_sparse_dense_matmul(
                            self.backend.bk_real(tim),
                            self.weight_interp_val[(lout,nout)],
                        )
                    ii = self.backend.bk_sparse_dense_matmul(
                            self.backend.bk_real(tim),
                            self.weight_interp_val[(lout,nout)],
                        )
                    imout = self.backend.bk_complex(rr, ii)
                else:
                    imout = self.backend.bk_sparse_dense_matmul(
                        tim,
                        self.weight_interp_val[(lout,nout)],
                    )

                if len(ishape) == 1:
                    return self.backend.bk_reshape(imout, [imout.shape[-1]])
                else:
                    return self.backend.bk_reshape(
                        imout, ishape[0:-1]+[imout.shape[-1]]
                    )
        return imout

    # --------------------------------------------------------
    def fill_1d(self, i_arr, nullval=0):
        arr = i_arr.copy()
        # Indices des éléments non nuls
        non_zero_indices = np.where(arr != nullval)[0]

        # Indices de tous les éléments
        all_indices = np.arange(len(arr))

        # Interpoler linéairement en utilisant np.interp
        # np.interp(x, xp, fp) : x sont les indices pour lesquels on veut obtenir des valeurs
        # xp sont les indices des données existantes, fp sont les valeurs des données existantes
        interpolated_values = np.interp(
            all_indices, non_zero_indices, arr[non_zero_indices]
        )

        # Mise à jour du tableau original
        arr[arr == nullval] = interpolated_values[arr == nullval]

        return arr

    def fill_2d(self, i_arr, nullval=0):
        arr = i_arr.copy()
        # Créer une grille de coordonnées correspondant aux indices du tableau
        x, y = np.indices(arr.shape)

        # Extraire les coordonnées des points non nuls ainsi que leurs valeurs
        non_zero_points = np.array((x[arr != nullval], y[arr != nullval])).T
        non_zero_values = arr[arr != nullval]

        # Extraire les coordonnées des points nuls
        zero_points = np.array((x[arr == nullval], y[arr == nullval])).T

        # Interpolation linéaire
        interpolated_values = griddata(
            non_zero_points, non_zero_values, zero_points, method="linear"
        )

        # Remplacer les valeurs nulles par les valeurs interpolées
        arr[arr == nullval] = interpolated_values

        return arr

    def fill_healpy(self, i_map, nmax=10, nullval=hp.UNSEEN):
        map = 1 * i_map
        # Trouver les pixels nuls
        nside = hp.npix2nside(len(map))
        null_indices = np.where(map == nullval)[0]

        itt = 0
        while null_indices.shape[0] > 0 and itt < nmax:
            # Trouver les coordonnées theta, phi pour les pixels nuls
            theta, phi = hp.pix2ang(nside, null_indices)

            # Interpoler les valeurs en utilisant les pixels voisins
            # La fonction get_interp_val peut être utilisée pour obtenir les valeurs interpolées
            # pour des positions données en theta et phi.
            i_idx = hp.get_all_neighbours(nside, theta, phi)

            i_w = (map[i_idx] != nullval) * (i_idx != -1)
            vv = np.sum(i_w, 0)
            interpolated_values = np.sum(i_w * map[i_idx], 0)

            # Remplacer les valeurs nulles par les valeurs interpolées
            map[null_indices[vv > 0]] = interpolated_values[vv > 0] / vv[vv > 0]

            null_indices = np.where(map == nullval)[0]
            itt += 1

        return map

    # --------------------------------------------------------
    def ud_grade_1d(self, im, nout, axis=0):
        npix = im.shape[axis]

        ishape = list(im.shape)
        odata = 1
        for k in range(axis + 1, len(ishape)):
            odata = odata * ishape[k]

        ndata = 1
        for k in range(axis):
            ndata = ndata * ishape[k]

        nscale = npix // nout
        if npix % nscale == 0:
            tim = self.backend.bk_reshape(
                self.backend.bk_cast(im), [ndata, npix // nscale, nscale, odata]
            )
        else:
            im = self.backend.bk_reshape(self.backend.bk_cast(im), [ndata, npix, odata])
            tim = self.backend.bk_reshape(
                self.backend.bk_cast(im[:, 0 : nscale * (npix // nscale), :]),
                [ndata, npix // nscale, nscale, odata],
            )
        res = self.backend.bk_reduce_mean(tim, 2)

        if axis == 0:
            if len(ishape) == 1:
                return self.backend.bk_reshape(res, [nout])
            else:
                return self.backend.bk_reshape(res, [nout] + ishape[axis + 1 :])
        else:
            if len(ishape) == axis + 1:
                return self.backend.bk_reshape(res, ishape[0:axis] + [nout])
            else:
                return self.backend.bk_reshape(
                    res, ishape[0:axis] + [nout] + ishape[axis + 1 :]
                )
        return self.backend.bk_reshape(res, [nout])

    # --------------------------------------------------------
    def up_grade_2_1d(self, im, axis=0):

        npix = im.shape[axis]

        ishape = list(im.shape)
        odata = 1
        for k in range(axis + 1, len(ishape)):
            odata = odata * ishape[k]

        ndata = 1
        for k in range(axis):
            ndata = ndata * ishape[k]

        tim = self.backend.bk_reshape(self.backend.bk_cast(im), [ndata, npix, odata])

        res2 = self.backend.bk_expand_dims(
            self.backend.bk_concat(
                [(tim[:, 1:, :] + 3 * tim[:, :-1, :]) / 4, tim[:, -1:, :]], 1
            ),
            -2,
        )
        res1 = self.backend.bk_expand_dims(
            self.backend.bk_concat(
                [tim[:, 0:1, :], (tim[:, 1:, :] * 3 + tim[:, :-1, :]) / 4], 1
            ),
            -2,
        )
        res = self.backend.bk_concat([res1, res2], -2)

        if axis == 0:
            if len(ishape) == 1:
                return self.backend.bk_reshape(res, [npix * 2])
            else:
                return self.backend.bk_reshape(res, [npix * 2] + ishape[axis + 1 :])
        else:
            if len(ishape) == axis + 1:
                return self.backend.bk_reshape(res, ishape[0:axis] + [npix * 2])
            else:
                return self.backend.bk_reshape(
                    res, ishape[0:axis] + [npix * 2] + ishape[axis + 1 :]
                )
        return self.backend.bk_reshape(res, [npix * 2])

    # --------------------------------------------------------
    def convol_1d(self, im, axis=0):

        xx = np.arange(5) - 2
        w = np.exp(-0.17328679514 * (xx) ** 2)
        c = np.cos((xx) * np.pi / 2)
        s = np.sin((xx) * np.pi / 2)

        wr = np.array(w * c).reshape(xx.shape[0], 1, 1)
        wi = np.array(w * s).reshape(xx.shape[0], 1, 1)

        npix = im.shape[axis]

        ishape = list(im.shape)
        odata = 1
        for k in range(axis + 1, len(ishape)):
            odata = odata * ishape[k]

        ndata = 1
        for k in range(axis):
            ndata = ndata * ishape[k]

        if odata > 1:
            wr = np.repeat(wr, odata, 2)
            wi = np.repeat(wi, odata, 2)

        wr = self.backend.bk_cast(self.backend.bk_constant(wr))
        wi = self.backend.bk_cast(self.backend.bk_constant(wi))

        tim = self.backend.bk_reshape(self.backend.bk_cast(im), [ndata, npix, odata])

        if tim.dtype == self.all_cbk_type:
            rr1 = self.backend.bk_conv1d(self.backend.bk_real(tim), wr)
            ii1 = self.backend.bk_conv1d(self.backend.bk_real(tim), wi)
            rr2 = self.backend.bk_conv1d(self.backend.bk_imag(tim), wr)
            ii2 = self.backend.bk_conv1d(self.backend.bk_imag(tim), wi)
            res = self.backend.bk_complex(rr1 - ii2, ii1 + rr2)
        else:
            rr = self.backend.bk_conv1d(tim, wr)
            ii = self.backend.bk_conv1d(tim, wi)

            res = self.backend.bk_complex(rr, ii)

        if axis == 0:
            if len(ishape) == 1:
                return self.backend.bk_reshape(res, [npix])
            else:
                return self.backend.bk_reshape(res, [npix] + ishape[axis + 1 :])
        else:
            if len(ishape) == axis + 1:
                return self.backend.bk_reshape(res, ishape[0:axis] + [npix])
            else:
                return self.backend.bk_reshape(
                    res, ishape[0:axis] + [npix] + ishape[axis + 1 :]
                )
        return self.backend.bk_reshape(res, [npix])

    # --------------------------------------------------------
    def smooth_1d(self, im, axis=0):

        xx = np.arange(5) - 2
        w = np.exp(-0.17328679514 * (xx) ** 2)
        w = w / w.sum()
        w = np.array(w).reshape(xx.shape[0], 1, 1)

        npix = im.shape[axis]

        ishape = list(im.shape)
        odata = 1
        for k in range(axis + 1, len(ishape)):
            odata = odata * ishape[k]

        ndata = 1
        for k in range(axis):
            ndata = ndata * ishape[k]

        if odata > 1:
            w = np.repeat(w, odata, 2)

        w = self.backend.bk_cast(self.backend.bk_constant(w))

        tim = self.backend.bk_reshape(self.backend.bk_cast(im), [ndata, npix, odata])

        if tim.dtype == self.all_cbk_type:
            rr = self.backend.bk_conv1d(self.backend.bk_real(tim), w)
            ii = self.backend.bk_conv1d(self.backend.bk_real(tim), w)
            res = self.backend.bk_complex(rr, ii)
        else:
            res = self.backend.bk_conv1d(tim, w)

        if axis == 0:
            if len(ishape) == 1:
                return self.backend.bk_reshape(res, [npix])
            else:
                return self.backend.bk_reshape(res, [npix] + ishape[axis + 1 :])
        else:
            if len(ishape) == axis + 1:
                return self.backend.bk_reshape(res, ishape[0:axis] + [npix])
            else:
                return self.backend.bk_reshape(
                    res, ishape[0:axis] + [npix] + ishape[axis + 1 :]
                )
        return self.backend.bk_reshape(res, [npix])

    # --------------------------------------------------------
    def up_grade_1d(self, im, nout, axis=0):

        lout = int(im.shape[axis])
        nscale = int(np.log(nout // lout) / np.log(2))
        res = self.backend.bk_cast(im)
        for k in range(nscale):
            res = self.up_grade_2_1d(res, axis=axis)
        return res

    # ---------------------------------------------−---------
    def init_index(self, nside, kernel=-1, cell_ids=None, spin=0):

        if kernel == -1:
            l_kernel = self.KERNELSZ
        else:
            l_kernel = kernel

        if cell_ids is not None:
            ncell = cell_ids.shape[0]
        else:
            ncell = 12 * nside * nside

        try:
            if self.use_2D:
                tmp = self.read_index("%s/W%d_%s_%d_IDX.fst"
                                      % (self.TEMPLATE_PATH, l_kernel**2,TMPFILE_VERSION, nside)
                )
            else:
                if cell_ids is not None and spin==0:
                    tmp = self.read_index(
                        "%s/XXXX_%s_W%d_%d_%d_PIDX.fst"  # can not work
                        % (
                            self.TEMPLATE_PATH,
                            TMPFILE_VERSION,
                            l_kernel**2,
                            self.NORIENT,
                            nside,  # if cell_ids computes the index
                        )
                    )

                else:
                    '''
                    print('LOAD ',"%s/FOSCAT_%s_W%d_%d_%d_PIDX-SPIN%d.fst"
                        % (
                            self.TEMPLATE_PATH,
                            TMPFILE_VERSION,
                            l_kernel**2,
                            self.NORIENT,
                            nside,spin  # if cell_ids computes the index
                        ))
                    '''
                    tmp = self.read_index(
                        "%s/FOSCAT_%s_W%d_%d_%d_PIDX-SPIN%d.fst"
                        % (
                            self.TEMPLATE_PATH,
                            TMPFILE_VERSION,
                            l_kernel**2,
                            self.NORIENT,
                            nside,spin  # if cell_ids computes the index
                        )
                    )
                        
        except:
            if cell_ids is not None and spin!=0:
                self.init_index(nside, kernel=kernel, spin=spin)
                
            if not self.use_2D:
                if spin!=0:
                    # keep the print here as spin!=0 can be long
                    print('NOT FOUND THEN COMPUTE %s/FOSCAT_%s_W%d_%d_%d_PIDX-SPIN%d.fst'
                        % (
                            self.TEMPLATE_PATH,
                            TMPFILE_VERSION,
                            l_kernel**2,
                            self.NORIENT,
                            nside,spin  # if cell_ids computes the index
                        )
                      )
                    try:
                        tmp = self.read_index(
                            "%s/FOSCAT_%s_W%d_%d_%d_PIDX-SPIN0.fst"
                            % (
                                self.TEMPLATE_PATH,
                                TMPFILE_VERSION,
                                l_kernel**2,
                                self.NORIENT,
                                nside
                            )
                        )
                    except:
                        '''
                        print('NOT FOUND THEN COMPUTE %s/FOSCAT_%s_W%d_%d_%d_PIDX-SPIN0.fst'
                              % (
                                  self.TEMPLATE_PATH,
                                  TMPFILE_VERSION,
                                  l_kernel**2,
                                  self.NORIENT,
                                  nside
                              )
                              )
                        '''
                        self.init_index(nside, kernel=kernel, spin=0)
                        
                        tmp = self.read_index(
                            "%s/FOSCAT_%s_W%d_%d_%d_PIDX-SPIN0.fst"
                            % (
                                self.TEMPLATE_PATH,
                                TMPFILE_VERSION,
                                l_kernel**2,
                                self.NORIENT,
                                nside
                            )
                        )
                        
                    tmpw = self.read_index("%s/FOSCAT_%s_W%d_%d_%d_WAVE-SPIN0.fst"% (
                                            self.TEMPLATE_PATH,
                                            self.TMPFILE_VERSION,
                                            self.KERNELSZ**2,
                                            self.NORIENT,
                                            nside,
                                        )
                                    )

                    '''
                    nn=self.NORIENT*12*nside**2
                    idxEB=np.concatenate([tmp,tmp,tmp,tmp],0)
                    idxEB[tmp.shape[0]:2*tmp.shape[0],0]+=12*nside**2
                    idxEB[3*tmp.shape[0]:,0]+=12*nside**2
                    idxEB[2*tmp.shape[0]:,1]+=nn

                    tmpEB=np.zeros([tmpw.shape[0]*4],dtype='complex')
                    '''
                    import foscat.HOrientedConvol as hs

                    hconvol=hs.HOrientedConvol(nside,3*self.KERNELSZ,cell_ids=cell_ids)
                    
                    if cell_ids is None:
                        l_cell_ids=np.arange(12*nside**2)
                    else:
                        l_cell_ids=cell_ids
                        
                    nvalid=self.KERNELSZ**2
                    idxEB=hconvol.idx_nn[:,0:nvalid]
                    tmpEB=np.zeros([self.NORIENT,4,l_cell_ids.shape[0],nvalid],dtype='complex')       
                    tmpS=np.zeros([4,l_cell_ids.shape[0],nvalid],dtype='float')           
                    
                    idx={}
                    nn=0
                    nn2=1
                    if nside<64:
                        pp=10
                    else:
                        pp=1
                    while nn2>0:
                        idx2={}
                        nn2=0
                        im=np.zeros([12*nside**2])
                        for n in range(l_cell_ids.shape[0]):
                            if im[hconvol.idx_nn[n,0]]==0 and n not in idx:
                                im[hconvol.idx_nn[n,:]]=1.0
                                idx[hconvol.idx_nn[n,0]]=1.0
                                idx2[hconvol.idx_nn[n,0]]=1.0
                                nn+=1
                                nn2+=1
                        im=np.zeros([12*nside**2])
                        for k in idx2:
                            im[k]=1.0
                        r=self.convol(im)
                        for k in range(self.NORIENT):
                            ralm=hp.map2alm(hp.reorder(r[k].cpu().numpy().real,n2r=True))[None,:]
                            ialm=hp.map2alm(hp.reorder(r[k].cpu().numpy().imag,n2r=True))[None,:]
                            
                            alm=np.concatenate([ralm,0*ralm,0*ralm],0)
                            rqe,rue,rie=hp.alm2map_spin(alm,nside,spin,3*nside-1)
                            alm=np.concatenate([ialm,0*ialm,0*ialm],0)
                            iqe,iue,iie=hp.alm2map_spin(alm,nside,spin,3*nside-1)
                            
                            alm=np.concatenate([0*ralm,ralm,0*ralm],0)
                            rqb,rub,rib=hp.alm2map_spin(alm,nside,spin,3*nside-1)
                            alm=np.concatenate([0*ialm,ialm,0*ialm],0)
                            iqb,iub,iib=hp.alm2map_spin(alm,nside,spin,3*nside-1)
                            
                            rqe=hp.reorder(rqe,r2n=True)
                            rue=hp.reorder(rue,r2n=True)
                            rqb=hp.reorder(rqb,r2n=True)
                            rub=hp.reorder(rub,r2n=True)
                            
                            iqe=hp.reorder(iqe,r2n=True)
                            iue=hp.reorder(iue,r2n=True)
                            iqb=hp.reorder(iqb,r2n=True)
                            iub=hp.reorder(iub,r2n=True)
                            
                            for l in idx2:
                                tmpEB[k,0,l]=rqe[idxEB[l,:]]+1J*iqe[idxEB[l,:]]
                                tmpEB[k,1,l]=rue[idxEB[l,:]]+1J*iue[idxEB[l,:]]
                                tmpEB[k,2,l]=rqb[idxEB[l,:]]+1J*iqb[idxEB[l,:]]
                                tmpEB[k,3,l]=rub[idxEB[l,:]]+1J*iub[idxEB[l,:]]
                                
                        r=self.smooth(im)
                        
                        ralm=hp.map2alm(hp.reorder(r.cpu().numpy(),n2r=True))[None,:]
                            
                        alm=np.concatenate([ralm,0*ralm,0*ralm],0)
                        rqe,rue,rie=hp.alm2map_spin(alm,nside,spin,3*nside-1)
                            
                        alm=np.concatenate([0*ralm,ralm,0*ralm],0)
                        rqb,rub,rib=hp.alm2map_spin(alm,nside,spin,3*nside-1)
                        
                        rqe=hp.reorder(rqe,r2n=True)
                        rue=hp.reorder(rue,r2n=True)
                        rqb=hp.reorder(rqb,r2n=True)
                        rub=hp.reorder(rub,r2n=True)
                            
                        for l in idx2:
                            tmpS[0,l,:]=rqe[idxEB[l,:]]
                            tmpS[1,l,:]=rue[idxEB[l,:]]
                            tmpS[2,l,:]=rqb[idxEB[l,:]]
                            tmpS[3,l,:]=rub[idxEB[l,:]]
                        if 100*nn/(l_cell_ids.shape[0])>pp:
                            if nside<64:
                                pp+=10
                            else:
                                pp+=1
                            print('%.2f%% Done'%(100*nn/(l_cell_ids.shape[0])))
                        
                    wav=tmpEB.flatten()
                    wwav=tmpS.flatten()
                    ndata=l_cell_ids.shape[0]*nvalid
                    indice_1_1=np.tile(idxEB.flatten(),4*self.NORIENT)
                    for k in range(self.NORIENT):
                        indice_1_1[(4*k+1)*ndata:(4*k+2)*ndata]+=l_cell_ids.shape[0]
                        indice_1_1[(4*k+3)*ndata:(4*k+4)*ndata]+=l_cell_ids.shape[0]
                    
                    indice_1_0=np.tile(np.tile(np.repeat(np.arange(l_cell_ids.shape[0]),nvalid),4),self.NORIENT)
                    for k in range(self.NORIENT):
                        indice_1_0[(4*k+2)*ndata:(4*k+4)*ndata]+=self.NORIENT*l_cell_ids.shape[0]
                        indice_1_0[(4*k)*ndata:(4*k+4)*ndata]+=k*l_cell_ids.shape[0]
                    '''
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.subplot(2,2,1)
                    plt.plot(indice_1_0)
                    plt.subplot(2,2,2)
                    plt.plot(indice_1_1)
                    plt.subplot(2,2,3)
                    plt.plot(wav.real)
                    plt.subplot(2,2,4)
                    plt.plot(abs(wav))
                    
                    iarg=np.argsort(indice_1_0)
                    indice_1_1=indice_1_1[iarg]
                    indice_1_0=indice_1_0[iarg]
                    wav=wav[iarg]
                    '''
                    
                    indice=np.concatenate([indice_1_1[:,None],indice_1_0[:,None]],1)
                    
                    indice_2_1=np.tile(idxEB.flatten(),4)
                    indice_2_1[ndata:2*ndata]+=l_cell_ids.shape[0]
                    indice_2_1[3*ndata:4*ndata]+=l_cell_ids.shape[0]
                    indice_2_0=np.tile(np.repeat(np.arange(l_cell_ids.shape[0]),nvalid),4)
                    indice_2_0[2*ndata:]+=l_cell_ids.shape[0]
                    '''
                    plt.figure()
                    plt.subplot(2,2,1)
                    plt.plot(indice_2_0)
                    plt.subplot(2,2,2)
                    plt.plot(indice_2_1)
                    plt.subplot(2,2,3)
                    plt.plot(wav.real)
                    plt.subplot(2,2,4)
                    plt.plot(wwav)
                    
                    iarg=np.argsort(indice_2_0)
                    indice_2_1=indice_2_1[iarg]
                    indice_2_0=indice_2_0[iarg]
                    wwav=wwav[iarg]
                    '''
                    indice2=np.concatenate([indice_2_1[:,None],indice_2_0[:,None]],1)
                    
                    '''
                    for k in range(self.NORIENT*12*nside**2):
                        if k%(nside**2)==0:
                            print('Init index 1/2 spin=%d Please wait %d done against %d nside=%d kernel=%d'%(spin,k//(nside**2),
                                                                                                              self.NORIENT*12,
                                                                                                              nside,
                                                                                                              self.KERNELSZ))
                        idx=np.where(tmp[:,1]==k)[0]

                        im=np.zeros([12*nside**2])
                        im[tmp[idx,0]]=tmpw[idx].real
                        almR=hp.map2alm(hp.reorder(im,n2r=True))
                        im[tmp[idx,0]]=tmpw[idx].imag
                        almI=hp.map2alm(hp.reorder(im,n2r=True))

                        i,q,u=hp.alm2map_spin([almR,almR*0,0*almR],nside,spin,3*nside-1)
                        i2,q2,u2=hp.alm2map_spin([almI,0*almI,0*almI],nside,spin,3*nside-1)

                        tmpEB[idx]=hp.reorder(i,r2n=True)[tmp[idx,0]]+1J*hp.reorder(i2,r2n=True)[tmp[idx,0]]
                        tmpEB[idx+tmp.shape[0]]=hp.reorder(q,r2n=True)[tmp[idx,0]]+1J*hp.reorder(q2,r2n=True)[tmp[idx,0]]

                        i,q,u=hp.alm2map_spin([0*almR,almR,0*almR],nside,spin,3*nside-1)
                        i2,q2,u2=hp.alm2map_spin([0*almI,almI,0*almI],nside,spin,3*nside-1)

                        tmpEB[idx+2*tmp.shape[0]]=hp.reorder(i,r2n=True)[tmp[idx,0]]+1J*hp.reorder(i2,r2n=True)[tmp[idx,0]]
                        tmpEB[idx+3*tmp.shape[0]]=hp.reorder(q,r2n=True)[tmp[idx,0]]+1J*hp.reorder(q2,r2n=True)[tmp[idx,0]]

                    '''
                    self.save_index("%s/FOSCAT_%s_W%d_%d_%d_PIDX-SPIN%d.fst"% (self.TEMPLATE_PATH,
                                                                       self.TMPFILE_VERSION,
                                                                       self.KERNELSZ**2,
                                                                       self.NORIENT,
                                                                       nside,
                                                                       spin
                                                                       ),
                                    indice
                                    )
                    self.save_index("%s/FOSCAT_%s_W%d_%d_%d_WAVE-SPIN%d.fst"% (self.TEMPLATE_PATH,
                                                                       self.TMPFILE_VERSION,
                                                                       self.KERNELSZ**2,
                                                                       self.NORIENT,
                                                                       nside,
                                                                       spin,
                                                                       ),
                                    wav
                                    )
                    '''
                    tmp = self.read_index(
                            "%s/FOSCAT_%s_W%d_%d_%d_PIDX2-SPIN0.fst"
                            % (
                                self.TEMPLATE_PATH,
                                TMPFILE_VERSION,
                                l_kernel**2,
                                self.NORIENT,
                                nside
                            )
                        )
                        
                    tmpw = self.read_index("%s/FOSCAT_%s_W%d_%d_%d_SMOO-SPIN0.fst"% (
                                            self.TEMPLATE_PATH,
                                            self.TMPFILE_VERSION,
                                            self.KERNELSZ**2,
                                            self.NORIENT,
                                            nside,
                                        )
                                    )
                    for k in range(12*nside**2):
                        if k%(nside**2)==0:
                            print('Init index 2/2 spin=%d Please wait %d done against %d nside=%d kernel=%d'%(spin,k//(nside**2),
                                                                                                              12,
                                                                                                              nside,
                                                                                                              self.KERNELSZ))
                        idx=np.where(tmp[:,1]==k)[0]

                        im=np.zeros([12*nside**2])
                        im[tmp[idx,0]]=tmpw[idx]
                        almR=hp.map2alm(hp.reorder(im,n2r=True))

                        i,q,u=hp.alm2map_spin([almR,almR*0,0*almR],nside,spin,3*nside-1)

                        tmpEB[idx]=hp.reorder(i,r2n=True)[tmp[idx,0]]
                        tmpEB[idx+tmp.shape[0]]=hp.reorder(q,r2n=True)[tmp[idx,0]]

                        i,q,u=hp.alm2map_spin([0*almR,almR,0*almR],nside,spin,3*nside-1)

                        tmpEB[idx+2*tmp.shape[0]]=hp.reorder(i,r2n=True)[tmp[idx,0]]
                        tmpEB[idx+3*tmp.shape[0]]=hp.reorder(q,r2n=True)[tmp[idx,0]]

                    '''
                    self.save_index("%s/FOSCAT_%s_W%d_%d_%d_PIDX2-SPIN%d.fst"% (self.TEMPLATE_PATH,
                                                                       self.TMPFILE_VERSION,
                                                                       self.KERNELSZ**2,
                                                                       self.NORIENT,
                                                                       nside,
                                                                       spin
                                                                       ),
                                    indice2
                                    )
                    self.save_index("%s/FOSCAT_%s_W%d_%d_%d_SMOO-SPIN%d.fst"% (self.TEMPLATE_PATH,
                                                                       self.TMPFILE_VERSION,
                                                                       self.KERNELSZ**2,
                                                                       self.NORIENT,
                                                                       nside,
                                                                       spin,
                                                                       ),
                                    wwav
                                    )
                    
                else:
                    '''
                    if l_kernel == 5:
                        pw = 0.5
                        pw2 = 0.5
                        threshold = 2e-4

                    elif l_kernel == 3:
                        pw = 1.0 / np.sqrt(2)
                        pw2 = 1.0
                        threshold = 1e-3

                    elif l_kernel == 7:
                        pw = 0.5
                        pw2 = 0.25
                        threshold = 4e-5
                    '''
                    import foscat.HOrientedConvol as hs

                    hconvol=hs.HOrientedConvol(nside,l_kernel,cell_ids=cell_ids)
                    
                    orientations=np.pi*np.arange(self.NORIENT)/self.NORIENT
                    
                    wav,indice,wwav,indice2=hconvol.make_wavelet_matrix(orientations,
                                                                        polar=True,
                                                                        return_index=True,
                                                                        return_smooth=True)
                    
                    '''
                    if cell_ids is not None and nside>256:
                        if not isinstance(cell_ids, np.ndarray):
                            cell_ids = self.backend.to_numpy(cell_ids)
                        th, ph = hp.pix2ang(nside, cell_ids, nest=True)
                        x, y, z = hp.pix2vec(nside, cell_ids, nest=True)

                        t, p = hp.pix2ang(nside, cell_ids, nest=True)
                        phi = [p[k] / np.pi * 180 for k in range(ncell)]
                        thi = [t[k] / np.pi * 180 for k in range(ncell)]

                        indice2 = np.zeros([ncell * 64, 2], dtype="int")
                        indice = np.zeros([ncell * 64 * self.NORIENT, 2], dtype="int")
                        wav = np.zeros([ncell * 64 * self.NORIENT], dtype="complex")
                        wwav = np.zeros([ncell * 64 * self.NORIENT], dtype="float")

                    else:

                        th, ph = hp.pix2ang(nside, np.arange(12 * nside**2), nest=True)
                        x, y, z = hp.pix2vec(nside, np.arange(12 * nside**2), nest=True)

                        t, p = hp.pix2ang(nside, np.arange(12 * nside * nside), nest=True)
                        phi = [p[k] / np.pi * 180 for k in range(12 * nside * nside)]
                        thi = [t[k] / np.pi * 180 for k in range(12 * nside * nside)]

                        indice2 = np.zeros([12 * nside * nside * 64, 2],
                                           dtype="int")
                        
                        indice = np.zeros(
                            [12 * nside * nside * 64 * self.NORIENT, 2],
                            dtype="int"
                        )
                        wav = np.zeros(
                            [12 * nside * nside * 64 * self.NORIENT],
                            dtype="complex"
                        )
                        wwav = np.zeros(
                            [12 * nside * nside * 64 * self.NORIENT],
                            dtype="float"
                        )
                    iv = 0
                    iv2 = 0

                    for iii in range(ncell):
                        if cell_ids is None:
                            if iii % (nside * nside) == nside * nside - 1:
                                if not self.silent:
                                    print(
                                        "Pre-compute nside=%6d %.2f%%"
                                        % (nside, 100 * iii / (12 * nside * nside))
                                    )

                        if cell_ids is not None:
                            hidx = np.where(
                                (x - x[iii]) ** 2 + (y - y[iii]) ** 2 + (z - z[iii]) ** 2
                                < (2 * np.pi / nside) ** 2
                            )[0]
                        else:
                            hidx = hp.query_disc(
                                nside,
                                [x[iii], y[iii], z[iii]],
                                2 * np.pi / nside,
                                nest=True,
                            )

                        R = hp.Rotator(rot=[phi[iii], -thi[iii]], eulertype="ZYZ")

                        t2, p2 = R(th[hidx], ph[hidx])

                        vec2 = hp.ang2vec(t2, p2)

                        x2 = vec2[:, 0]
                        y2 = vec2[:, 1]
                        z2 = vec2[:, 2]

                        ww = np.exp(
                            -pw2
                            * ((nside) ** 2)
                            * ((x2) ** 2 + (y2) ** 2 + (z2 - 1.0) ** 2)
                        )
                        idx = np.where((ww**2) > threshold)[0]
                        nval2 = len(idx)
                        indice2[iv2 : iv2 + nval2, 1] = iii
                        indice2[iv2 : iv2 + nval2, 0] = hidx[idx]
                        wwav[iv2 : iv2 + nval2] = ww[idx] / np.sum(ww[idx])
                        iv2 += nval2

                        for l_rotation in range(self.NORIENT):

                            angle = (
                                l_rotation / 4.0 * np.pi
                                - phi[iii] / 180.0 * np.pi * (z[hidx] > 0)
                                - (180.0 - phi[iii]) / 180.0 * np.pi * (z[hidx] < 0)
                            )

                            # posi=2*(0.5-(z[hidx]<0))

                            axes = y2 * np.cos(angle) - x2 * np.sin(angle)
                            wresr = ww * np.cos(pw * axes * (nside) * np.pi)
                            wresi = ww * np.sin(pw * axes * (nside) * np.pi)

                            vnorm = wresr * wresr + wresi * wresi
                            idx = np.where(vnorm > threshold)[0]

                            nval = len(idx)
                            indice[iv : iv + nval, 1] = iii + l_rotation * ncell
                            indice[iv : iv + nval, 0] = hidx[idx]
                            # print([hidx[k] for k in idx])
                            # print(hp.query_disc(nside, [x[iii],y[iii],z[iii]], np.pi/nside,nest=True))
                            normr = np.mean(wresr[idx])
                            normi = np.mean(wresi[idx])

                            val = wresr[idx] - normr + 1j * (wresi[idx] - normi)
                            r = abs(val).sum()

                            if r > 0:
                                val = val / r

                            wav[iv : iv + nval] = val
                            iv += nval

                    indice = indice[:iv, :]
                    wav = wav[:iv]
                    indice2 = indice2[:iv2, :]
                    wwav = wwav[:iv2]
                    '''

                    if cell_ids is None:
                        if not self.silent:
                            print(
                                "Write %s/FOSCAT_%s_W%d_%d_%d_PIDX-SPIN%d.fst"
                                % ( self.TEMPLATE_PATH,
                                    TMPFILE_VERSION, self.KERNELSZ**2,
                                   self.NORIENT,
                                   nside,
                                spin)
                            )
                        self.save_index("%s/FOSCAT_%s_W%d_%d_%d_PIDX-SPIN%d.fst"
                                        % (
                                            self.TEMPLATE_PATH,
                                            TMPFILE_VERSION,
                                            self.KERNELSZ**2,
                                            self.NORIENT,
                                            nside,
                                            spin,
                                        ),
                                        indice
                                        )
                        self.save_index(
                            "%s/FOSCAT_%s_W%d_%d_%d_WAVE-SPIN%d.fst"
                            % (
                                self.TEMPLATE_PATH,
                                TMPFILE_VERSION,
                                self.KERNELSZ**2,
                                self.NORIENT,
                                nside,
                                spin,
                            ),
                            wav,
                        )
                        self.save_index(
                            "%s/FOSCAT_%s_W%d_%d_%d_PIDX2-SPIN%d.fst"
                            % (
                                self.TEMPLATE_PATH,
                                TMPFILE_VERSION,
                                self.KERNELSZ**2,
                                self.NORIENT,
                                nside,
                                spin,
                            ),
                            indice2,
                        )
                        self.save_index(
                            "%s/FOSCAT_%s_W%d_%d_%d_SMOO-SPIN%d.fst"
                            % (
                                self.TEMPLATE_PATH,
                                TMPFILE_VERSION,
                                self.KERNELSZ**2,
                                self.NORIENT,
                                nside,
                                spin,
                            ),
                            wwav,
                        )
            if self.use_2D:
                if l_kernel**2 == 9:
                    if self.rank == 0:
                        self.comp_idx_w9(nside)
                elif l_kernel**2 == 25:
                    if self.rank == 0:
                        self.comp_idx_w25(nside)
                else:
                    if self.rank == 0:
                        if not self.silent:
                            print(
                                "Only 3x3 and 5x5 kernel have been developped for Healpix and you ask for %dx%d"
                                % (self.KERNELSZ, self.KERNELSZ)
                            )
                        return None

        if cell_ids is None or spin!=0:
            self.barrier()
            if self.use_2D:
                tmp = self.read_index(
                    "%s/W%d_%s_%d_IDX-SPIN%d.fst"
                    % (
                        self.TEMPLATE_PATH,
                        l_kernel**2,
                        TMPFILE_VERSION,
                        nside,
                        spin)
                )
            else:
                tmp = self.read_index(
                    "%s/FOSCAT_%s_W%d_%d_%d_PIDX-SPIN%d.fst"
                    % (
                        self.TEMPLATE_PATH,
                        TMPFILE_VERSION,
                        self.KERNELSZ**2,
                        self.NORIENT,
                        nside,
                        spin,
                    )
                )
            tmp2 = self.read_index(
                "%s/FOSCAT_%s_W%d_%d_%d_PIDX2-SPIN%d.fst"
                % (
                    self.TEMPLATE_PATH,
                    TMPFILE_VERSION,
                    self.KERNELSZ**2,
                    self.NORIENT,
                    nside,
                    spin,
                )
            )
            wr = self.read_index(
                "%s/FOSCAT_%s_W%d_%d_%d_WAVE-SPIN%d.fst"
                % (
                    self.TEMPLATE_PATH,
                    TMPFILE_VERSION,
                    self.KERNELSZ**2,
                    self.NORIENT,
                    nside,
                    spin,
                )
            ).real
            wi = self.read_index(
                "%s/FOSCAT_%s_W%d_%d_%d_WAVE-SPIN%d.fst"
                % (
                    self.TEMPLATE_PATH,
                    TMPFILE_VERSION,
                    self.KERNELSZ**2,
                    self.NORIENT,
                    nside,
                    spin,
                )
            ).imag
            ws = self.slope * self.read_index(
                "%s/FOSCAT_%s_W%d_%d_%d_SMOO-SPIN%d.fst"
                % (
                    self.TEMPLATE_PATH,
                    TMPFILE_VERSION,
                    self.KERNELSZ**2,
                    self.NORIENT,
                    nside,
                    spin,
                )
            )
                                        
            if cell_ids is not None:
                idx_map=-np.ones([12*nside**2],dtype='int32')
                lcell_ids=cell_ids
                
                try:
                    idx_map[lcell_ids]=np.arange(lcell_ids.shape[0],dtype='int32')
                except:
                    lcell_ids=self.to_numpy(cell_ids)
                    idx_map[lcell_ids]=np.arange(lcell_ids.shape[0],dtype='int32')

                lidx=np.where(idx_map[tmp[:,1]%(12*nside**2)]!=-1)[0]
                orientation=tmp[lidx,1]//(12*nside**2)
                orientation2=tmp[lidx,0]//(12*nside**2)
                tmp=tmp[lidx]
                wr=wr[lidx]
                wi=wi[lidx]
                tmp=idx_map[tmp%(12*nside**2)]
                lidx=np.where(tmp[:,0]==-1)[0]
                wr[lidx]=0.0
                wi[lidx]=0.0
                tmp[lidx,0]=0
                tmp[:,1]+=orientation*lcell_ids.shape[0]
                tmp[:,0]+=orientation2*lcell_ids.shape[0]
                
                idx_map=-np.ones([12*nside**2],dtype='int32')
                idx_map[lcell_ids]=np.arange(cell_ids.shape[0],dtype='int32')
                lidx=np.where(idx_map[tmp2[:,1]%(12*nside**2)]!=-1)[0]
                i_id=tmp2[lidx,1]//(12*nside**2)
                i_id2=tmp2[lidx,0]//(12*nside**2)
                tmp2=tmp2[lidx]
                ws=ws[lidx]
                tmp2=idx_map[tmp2%(12*nside**2)]
                lidx=np.where(tmp2[:,0]==-1)[0]
                ws[lidx]=0.0
                tmp2[lidx,0]=0
                tmp2[:,1]+=i_id*lcell_ids.shape[0]
                tmp2[:,0]+=i_id2*lcell_ids.shape[0]

                #add normalisation
                ww=np.bincount(tmp2[:,1],weights=ws)
                ws/=ww[tmp2[:,1]]

                wh=np.bincount(tmp[:,1])
                ww=np.bincount(tmp[:,1],weights=wr)
                wr-=(ww/wh)[tmp[:,1]]
                ww=np.bincount(tmp[:,1],weights=wi)
                wi-=(ww/wh)[tmp[:,1]]
                
                ww=np.bincount(tmp[:,1],weights=np.sqrt(wr*wr+wi*wi))
                wr/=ww[tmp[:,1]]
                wi/=ww[tmp[:,1]]
                
        else:
            tmp = indice
            tmp2 = indice2
            wr = wav.real
            wi = wav.imag
            ws = self.slope * wwav

            
        if spin==0:
            wr = self.backend.bk_SparseTensor(
                self.backend.bk_constant(tmp),
                self.backend.bk_constant(self.backend.bk_cast(wr)),
                dense_shape=[ncell, self.NORIENT * ncell],
            )
            wi = self.backend.bk_SparseTensor(
                self.backend.bk_constant(tmp),
                self.backend.bk_constant(self.backend.bk_cast(wi)),
                dense_shape=[ncell, self.NORIENT * ncell],
            )
            ws = self.backend.bk_SparseTensor(
                self.backend.bk_constant(tmp2),
                self.backend.bk_constant(self.backend.bk_cast(ws)),
                dense_shape=[ncell, ncell],
            )
        else:
            wr = self.backend.bk_SparseTensor(
                self.backend.bk_constant(tmp),
                self.backend.bk_constant(self.backend.bk_cast(wr)),
                dense_shape=[2*ncell, 2*self.NORIENT * ncell],
            )
            wi = self.backend.bk_SparseTensor(
                self.backend.bk_constant(tmp),
                self.backend.bk_constant(self.backend.bk_cast(wi)),
                dense_shape=[2*ncell, 2*self.NORIENT * ncell],
            )
            ws = self.backend.bk_SparseTensor(
                self.backend.bk_constant(tmp2),
                self.backend.bk_constant(self.backend.bk_cast(ws)),
                dense_shape=[2*ncell, 2*ncell],
            )

        if kernel == -1:
            self.Idx_Neighbours[nside] = tmp

        if self.use_2D:
            if kernel != -1:
                return tmp

        return wr, wi, ws,tmp


    # ---------------------------------------------−---------
    def init_index_cnn(self, nside, NORIENT=4,kernel=-1, cell_ids=None):

        if kernel == -1:
            l_kernel = self.KERNELSZ
        else:
            l_kernel = kernel

        if cell_ids is not None:
            ncell = cell_ids.shape[0]
        else:
            ncell = 12 * nside * nside

        try:
            
            if cell_ids is not None:
                tmp = self.read_index(
                      "%s/XXXX_%s_W%d_%d_%d_PIDX.fst"  # can not work
                        % (
                            self.TEMPLATE_PATH,
                            TMPFILE_VERSION,
                            l_kernel**2,
                            NORIENT,
                            nside,  # if cell_ids computes the index
                        )
                    )

            else:
                    tmp = self.read_index(
                        "%s/CNN_FOSCAT_%s_W%d_%d_%d_PIDX.fst"
                        % (
                            self.TEMPLATE_PATH,
                            TMPFILE_VERSION,
                            l_kernel**2,
                            NORIENT,
                            nside,  # if cell_ids computes the index
                        )
                    )
        except:

            pw = 8.0 
            pw2 = 1.0
            threshold = 1e-3
            
            if l_kernel == 5:
                    pw = 8.0
                    pw2 = 0.5
                    threshold = 2e-4

            elif l_kernel == 3:
                    pw = 8.0 
                    pw2 = 1.0
                    threshold = 1e-3

            elif l_kernel == 7:
                    pw = 8.0
                    pw2 = 0.25
                    threshold = 4e-5
            
            n_weights = self.KERNELSZ*(self.KERNELSZ//2+1)
            
            if cell_ids is not None:
                    if not isinstance(cell_ids, np.ndarray):
                        cell_ids = self.backend.to_numpy(cell_ids)
                    th, ph = hp.pix2ang(nside, cell_ids, nest=True)
                    x, y, z = hp.pix2vec(nside, cell_ids, nest=True)

                    t, p = hp.pix2ang(nside, cell_ids, nest=True)
                    phi = [p[k] / np.pi * 180 for k in range(ncell)]
                    thi = [t[k] / np.pi * 180 for k in range(ncell)]

                    indice = np.zeros([n_weights, NORIENT, ncell,4], dtype="int")
                    
                    wav = np.zeros([n_weights, NORIENT, ncell,4], dtype="float")

            else:

                    th, ph = hp.pix2ang(nside, np.arange(12 * nside**2), nest=True)
                    x, y, z = hp.pix2vec(nside, np.arange(12 * nside**2), nest=True)

                    t, p = hp.pix2ang(nside, np.arange(12 * nside * nside), nest=True)
                    phi = [p[k] / np.pi * 180 for k in range(12 * nside * nside)]
                    thi = [t[k] / np.pi * 180 for k in range(12 * nside * nside)]

                    indice = np.zeros(
                        [n_weights, NORIENT, 12 * nside * nside,4], dtype="int"
                    )
                    wav = np.zeros(
                        [n_weights, NORIENT, 12 * nside * nside,4], dtype="float"
                    )
            iv = 0
            iv2 = 0

            for iii in range(ncell):
                if cell_ids is None:
                    if iii % (nside * nside) == nside * nside - 1:
                        if not self.silent:
                                print(
                                    "Pre-compute nside=%6d %.2f%%"
                                    % (nside, 100 * iii / (12 * nside * nside))
                                )

                if cell_ids is not None:
                        hidx = np.where(
                            (x - x[iii]) ** 2 + (y - y[iii]) ** 2 + (z - z[iii]) ** 2
                            < (2 * np.pi / nside) ** 2
                        )[0]
                else:
                        hidx = hp.query_disc(
                            nside,
                            [x[iii], y[iii], z[iii]],
                            2 * np.pi / nside,
                            nest=True,
                        )

                R = hp.Rotator(rot=[phi[iii], -thi[iii]], eulertype="ZYZ")

                t2, p2 = R(th[hidx], ph[hidx])

                vec2 = hp.ang2vec(t2, p2)

                x2 = vec2[:, 0]
                y2 = vec2[:, 1]
                z2 = vec2[:, 2]

                for l_rotation in range(NORIENT):

                        angle = (
                            l_rotation / 4.0 * np.pi
                            - phi[iii] / 180.0 * np.pi * (z[hidx] > 0)
                            - (180.0 - phi[iii]) / 180.0 * np.pi * (z[hidx] < 0)
                        )


                        axes  = y2 * np.cos(angle)  - x2 * np.sin(angle)
                        axes2 = -y2 * np.sin(angle) - x2 * np.cos(angle)
                        
                        for k_weights in range(self.KERNELSZ//2+1):
                            for l_weights in range(self.KERNELSZ):
                                
                                val=np.exp(-(pw*(axes2*(nside)-(k_weights-self.KERNELSZ//2))**2+pw*(axes*(nside)-(l_weights-self.KERNELSZ//2))**2))+ \
                                    np.exp(-(pw*(axes2*(nside)+(k_weights-self.KERNELSZ//2))**2+pw*(axes*(nside)-(l_weights-self.KERNELSZ//2))**2))

                                idx = np.argsort(-val)
                                idx = idx[0:4]
                                
                                nval = len(idx)
                                val=val[idx]
                                
                                r = abs(val).sum()

                                if r > 0:
                                    val = val / r
                                    
                                indice[k_weights*self.KERNELSZ+l_weights,l_rotation,iii,:] = hidx[idx]
                                wav[k_weights*self.KERNELSZ+l_weights,l_rotation,iii,:] = val
                
            if not self.silent:
                    print("Kernel Size ", iv / (NORIENT * 12 * nside * nside))
                
            if cell_ids is None:
                    if not self.silent:
                        print(
                            "Write FOSCAT_%s_W%d_%d_%d_PIDX.fst"
                            % (TMPFILE_VERSION, self.KERNELSZ**2, NORIENT, nside)
                        )
                    self.save_index(
                        "%s/CNN_FOSCAT_%s_W%d_%d_%d_PIDX.fst"
                        % (
                            self.TEMPLATE_PATH,
                            TMPFILE_VERSION,
                            self.KERNELSZ**2,
                            NORIENT,
                            nside,
                        ),
                        indice,
                    )
                    self.save_index(
                        "%s/CNN_FOSCAT_%s_W%d_%d_%d_WAVE.fst"
                        % (
                            self.TEMPLATE_PATH,
                            TMPFILE_VERSION,
                            self.KERNELSZ**2,
                            NORIENT,
                            nside,
                        ),
                        wav,
                    )

        if cell_ids is None:
            self.barrier()
            if self.use_2D:
                tmp = self.read_index(
                    "%s/W%d_%s_%d_IDX.fst"
                    % (self.TEMPLATE_PATH, l_kernel**2, TMPFILE_VERSION, nside)
                )
            else:
                tmp = self.read_index(
                    "%s/CNN_FOSCAT_%s_W%d_%d_%d_PIDX.fst"
                    % (
                        self.TEMPLATE_PATH,
                        TMPFILE_VERSION,
                        self.KERNELSZ**2,
                        NORIENT,
                        nside,
                    )
                )
            wav = self.read_index(
                "%s/CNN_FOSCAT_%s_W%d_%d_%d_WAVE.fst"
                % (
                    self.TEMPLATE_PATH,
                    TMPFILE_VERSION,
                    self.KERNELSZ**2,
                    NORIENT,
                    nside,
                )
            )
        else:
            tmp = indice
        
        self.Idx_CNN[(nside,NORIENT,self.KERNELSZ)] = tmp
        self.Idx_WCNN[(nside,NORIENT,self.KERNELSZ)] = self.backend.bk_cast(wav)

        return wav, tmp
        
    # ---------------------------------------------−---------
    # convert swap axes tensor x [....,a,....,b,....] to [....,b,....,a,....]
    def swapaxes(self, x, axis1, axis2):
        shape = list(x.shape)
        if axis1 < 0:
            laxis1 = len(shape) + axis1
        else:
            laxis1 = axis1
        if axis2 < 0:
            laxis2 = len(shape) + axis2
        else:
            laxis2 = axis2

        naxes = len(shape)
        thelist = [i for i in range(naxes)]
        thelist[laxis1] = laxis2
        thelist[laxis2] = laxis1
        return self.backend.bk_transpose(x, thelist)

    # ---------------------------------------------−---------
    # Mean using mask x [n_b,....,Npix], mask[Nmask,Npix]  to [n_b,Nmask,....]
    # if use_2D
    # Mean using mask x [n_b,....,N_1,N_2], mask[Nmask,N_1,N_2]  to [n_b,Nmask,....]
    def masked_mean(self, x, mask, rank=0, calc_var=False):

        # ==========================================================================
        # in input data=[Nbatch,...,NORIENT[,NORIENT],X[,Y]]
        # in input mask=[Nmask,X[,Y]]
        # if self.use_2D :  X[,Y]] = [X,Y]
        # if second level:  NORIENT[,NORIENT]= NORIENT,NORIENT
        # ==========================================================================

        shape = list(x.shape)

        if not self.use_2D and not self.use_1D:
            nside = int(np.sqrt(x.shape[-1] // 12))

        l_mask = mask
        if self.mask_norm:
            sum_mask = self.backend.bk_reduce_sum(
                self.backend.bk_reshape(
                    l_mask, [l_mask.shape[0], np.prod(np.array(l_mask.shape[1:]))]
                ),
                1,
            )

            if not self.use_2D:
                l_mask = (
                    12
                    * nside
                    * nside
                    * l_mask
                    / self.backend.bk_reshape(
                        sum_mask, [l_mask.shape[0]] + [1 for i in l_mask.shape[1:]]
                    )
                )
            elif self.use_2D:
                l_mask = (
                    mask.shape[1]
                    * mask.shape[2]
                    * l_mask
                    / self.backend.bk_reshape(
                        sum_mask, [l_mask.shape[0]] + [1 for i in l_mask.shape[1:]]
                    )
                )
            else:
                l_mask = (
                    mask.shape[1]
                    * l_mask
                    / self.backend.bk_reshape(
                        sum_mask, [l_mask.shape[0]] + [1 for i in l_mask.shape[1:]]
                    )
                )

        if self.use_2D:
            if self.padding == "VALID":
                l_mask = l_mask[
                    :,
                    self.KERNELSZ // 2 : -self.KERNELSZ // 2 + 1,
                    self.KERNELSZ // 2 : -self.KERNELSZ // 2 + 1,
                ]
                if shape[axis] != l_mask.shape[1]:
                    l_mask = l_mask[
                        :,
                        self.KERNELSZ // 2 : -self.KERNELSZ // 2 + 1,
                        self.KERNELSZ // 2 : -self.KERNELSZ // 2 + 1,
                    ]

            ichannel = 1
            for i in range(1, len(shape) - 2):
                ichannel *= shape[i]

            l_x = self.backend.bk_reshape(
                x, [shape[0], 1, ichannel, shape[-2], shape[-1]]
            )

            if self.padding == "VALID":
                oshape = [k for k in shape]
                oshape[axis] = oshape[axis] - self.KERNELSZ + 1
                oshape[axis + 1] = oshape[axis + 1] - self.KERNELSZ + 1
                l_x = self.backend.bk_reshape(
                    l_x[
                        :,
                        :,
                        self.KERNELSZ // 2 : -self.KERNELSZ // 2 + 1,
                        self.KERNELSZ // 2 : -self.KERNELSZ // 2 + 1,
                        :,
                    ],
                    oshape,
                )

        elif self.use_1D:
            if self.padding == "VALID":
                l_mask = l_mask[:, self.KERNELSZ // 2 : -self.KERNELSZ // 2 + 1]
                if shape[axis] != l_mask.shape[1]:
                    l_mask = l_mask[:, self.KERNELSZ // 2 : -self.KERNELSZ // 2 + 1]

            ichannel = 1
            for i in range(1, len(shape) - 1):
                ichannel *= shape[i]
            
            l_x = self.backend.bk_reshape(x, [shape[0], 1, ichannel,shape[-1]])

            if self.padding == "VALID":
                oshape = [k for k in shape]
                oshape[axis] = oshape[axis] - self.KERNELSZ + 1
                l_x = self.backend.bk_reshape(
                    l_x[:, :, self.KERNELSZ // 2 : -self.KERNELSZ // 2 + 1, :], oshape
                )
        else:   
            ichannel = 1
            if len(shape)>1:
                ichannel = shape[0]
                
            ochannel = 1
            for i in range(1,len(shape)-1):
                ochannel *= shape[i]

            l_x = self.backend.bk_reshape(x, [ichannel,1,ochannel,shape[-1]])

        # data=[Nbatch,...,NORIENT[,NORIENT],X[,Y]] => data=[Nbatch,...,1,NORIENT[,NORIENT],X[,Y]]
        # mask=[Nmask,X[,Y]] => mask=[1,Nmask,....,X[,Y]]
        
        if self.use_2D:
            l_mask = self.backend.bk_expand_dims(self.backend.bk_expand_dims(l_mask,0),-3)
        else:
            l_mask = self.backend.bk_expand_dims(self.backend.bk_expand_dims(l_mask,0),-2)

        if l_x.dtype == self.all_cbk_type:
            l_mask = self.backend.bk_complex(l_mask, self.backend.bk_cast(0.0 * l_mask))

        if self.use_2D:
            # if self.padding == "VALID":
            mtmp = l_mask
            vtmp = l_x
            # else:
            #    mtmp = l_mask[:,self.KERNELSZ // 2 : -self.KERNELSZ // 2,self.KERNELSZ // 2 : -self.KERNELSZ // 2,:]
            #    vtmp = l_x[:,self.KERNELSZ // 2 : -self.KERNELSZ // 2,self.KERNELSZ // 2 : -self.KERNELSZ // 2,:]

            v1 = self.backend.bk_reduce_sum(
                self.backend.bk_reduce_sum(mtmp * vtmp, axis=-1), -1
            )
            v2 = self.backend.bk_reduce_sum(
                self.backend.bk_reduce_sum(mtmp * vtmp * vtmp, axis=-1), -1
            )
            vh = self.backend.bk_reduce_sum(
                self.backend.bk_reduce_sum(mtmp, axis=-1), -1
            )

            res = v1 / vh

            oshape = [x.shape[0]] + [mask.shape[0]]
            if len(x.shape) > 3:
                oshape = oshape + list(x.shape[1:-2])
            else:
                oshape = oshape + [1]
                
            if calc_var:
                if self.backend.bk_is_complex(vtmp):
                    res2 = self.backend.bk_sqrt(
                        (
                            (
                                self.backend.bk_real(v2) / self.backend.bk_real(vh)
                                - self.backend.bk_real(res) * self.backend.bk_real(res)
                            )
                            + (
                                self.backend.bk_imag(v2) / self.backend.bk_real(vh)
                                - self.backend.bk_imag(res) * self.backend.bk_imag(res)
                            )
                        )
                        / self.backend.bk_real(vh)
                    )
                else:
                    res2 = self.backend.bk_sqrt((v2 / vh - res * res) / (vh))

                res = self.backend.bk_reshape(res, oshape)
                res2 = self.backend.bk_reshape(res2, oshape)
                return res, res2
            else:
                res = self.backend.bk_reshape(res, oshape)
                return res

        elif self.use_1D:
            mtmp = l_mask
            vtmp = l_x
            v1 = self.backend.bk_reduce_sum(l_mask * vtmp, axis=-1)
            v2 = self.backend.bk_reduce_sum(l_mask * vtmp * vtmp, axis=-1)
            vh = self.backend.bk_reduce_sum(l_mask , axis=-1)

            res = v1 / vh

            oshape = [x.shape[0]] + [mask.shape[0]]
            if len(x.shape) > 1:
                oshape = oshape + list(x.shape[1:-1])
            else:
                oshape = oshape + [1]

            if calc_var:
                if self.backend.bk_is_complex(vtmp):
                    res2 = self.backend.bk_sqrt(
                        (
                            (
                                self.backend.bk_real(v2) / self.backend.bk_real(vh)
                                - self.backend.bk_real(res) * self.backend.bk_real(res)
                            )
                            + (
                                self.backend.bk_imag(v2) / self.backend.bk_real(vh)
                                - self.backend.bk_imag(res) * self.backend.bk_imag(res)
                            )
                        )
                        / self.backend.bk_real(vh)
                    )
                else:
                    res2 = self.backend.bk_sqrt((v2 / vh - res * res) / (vh))
                res = self.backend.bk_reshape(res, oshape)
                res2 = self.backend.bk_reshape(res2, oshape)
                return res, res2
            else:
                res = self.backend.bk_reshape(res, oshape)
                return res

        else:
            v1 = self.backend.bk_reduce_sum(l_mask * l_x, axis=-1)
            v2 = self.backend.bk_reduce_sum(l_mask * l_x * l_x, axis=-1)
            vh = self.backend.bk_reduce_sum(l_mask, axis=-1)

            res = v1 / vh

            oshape = []
            if len(shape) > 1:
                oshape = [x.shape[0]]
            else:
                oshape = [1]
                
            oshape = oshape + [mask.shape[0]]
            if len(shape) > 2:
                oshape = oshape + shape[1:-1]
            else:
                oshape = oshape + [1]

            if calc_var:
                if self.backend.bk_is_complex(l_x):
                    res2 = self.backend.bk_sqrt(
                        (
                            self.backend.bk_real(v2) / self.backend.bk_real(vh)
                            - self.backend.bk_real(res) * self.backend.bk_real(res)
                            + self.backend.bk_imag(v2) / self.backend.bk_real(vh)
                            - self.backend.bk_imag(res) * self.backend.bk_imag(res)
                        )
                        / self.backend.bk_real(vh)
                    )
                else:
                    res2 = self.backend.bk_sqrt((v2 / vh - res * res) / (vh))

                res = self.backend.bk_reshape(res, oshape)
                res2 = self.backend.bk_reshape(res2, oshape)
                return res, res2
            else:
                res = self.backend.bk_reshape(res, oshape)
                return res

    # ---------------------------------------------−---------
    # convert tensor x [....,a,b,....] to [....,a*b,....]
    def reduce_dim(self, x, axis=0):
        shape = list(x.shape)

        if axis < 0:
            laxis = len(shape) + axis
        else:
            laxis = axis

        if laxis > 0:
            oshape = shape[0:laxis]
            oshape.append(shape[laxis] * shape[laxis + 1])
        else:
            oshape = [shape[laxis] * shape[laxis + 1]]

        if laxis < len(shape) - 1:
            oshape.extend(shape[laxis + 2 :])

        return self.backend.bk_reshape(x, oshape)

    # ---------------------------------------------−---------
    def conv2d(self, image, ww, axis=0):

        if len(ww.shape) == 2:
            norient = ww.shape[1]
        else:
            norient = ww.shape[2]

        shape = image.shape

        if axis > 0:
            o_shape = shape[0]
            for k in range(1, axis + 1):
                o_shape = o_shape * shape[k]
        else:
            o_shape = image.shape[0]

        if len(shape) > axis + 3:
            ishape = shape[axis + 3]
            for k in range(axis + 4, len(shape)):
                ishape = ishape * shape[k]

            oshape = [o_shape, shape[axis + 1], shape[axis + 2], ishape]

            # l_image=self.swapaxes(self.bk_reshape(image,oshape),-1,-3)
            l_image = self.backend.bk_reshape(image, oshape)

            l_ww = np.zeros([self.KERNELSZ, self.KERNELSZ, ishape, ishape * norient])
            for k in range(ishape):
                l_ww[:, :, k, k * norient : (k + 1) * norient] = ww.reshape(
                    self.KERNELSZ, self.KERNELSZ, norient
                )

            if self.backend.bk_is_complex(l_image):
                r = self.backend.conv2d(
                    self.backend.bk_real(l_image),
                    l_ww,
                    strides=[1, 1, 1, 1],
                    padding=self.padding,
                )
                i = self.backend.conv2d(
                    self.backend.bk_imag(l_image),
                    l_ww,
                    strides=[1, 1, 1, 1],
                    padding=self.padding,
                )
                res = self.backend.bk_complex(r, i)
            else:
                res = self.backend.conv2d(
                    l_image, l_ww, strides=[1, 1, 1, 1], padding=self.padding
                )

            res = self.backend.bk_reshape(
                res, [o_shape, shape[axis + 1], shape[axis + 2], ishape, norient]
            )
        else:
            oshape = [o_shape, shape[axis + 1], shape[axis + 2], 1]
            l_ww = self.backend.bk_reshape(
                ww, [self.KERNELSZ, self.KERNELSZ, 1, norient]
            )

            tmp = self.backend.bk_reshape(image, oshape)
            if self.backend.bk_is_complex(tmp):
                r = self.backend.conv2d(
                    self.backend.bk_real(tmp),
                    l_ww,
                    strides=[1, 1, 1, 1],
                    padding=self.padding,
                )
                i = self.backend.conv2d(
                    self.backend.bk_imag(tmp),
                    l_ww,
                    strides=[1, 1, 1, 1],
                    padding=self.padding,
                )
                res = self.backend.bk_complex(r, i)
            else:
                res = self.backend.conv2d(
                    tmp, l_ww, strides=[1, 1, 1, 1], padding=self.padding
                )

        return self.backend.bk_reshape(res, shape + [norient])

    def diff_data(self, x, y, is_complex=True, sigma=None):
        if sigma is None:
            if self.backend.bk_is_complex(x):
                r = self.backend.bk_square(
                    self.backend.bk_real(x) - self.backend.bk_real(y)
                )
                i = self.backend.bk_square(
                    self.backend.bk_imag(x) - self.backend.bk_imag(y)
                )
                return self.backend.bk_reduce_sum(r + i)
            else:
                r = self.backend.bk_square(x - y)
                return self.backend.bk_reduce_sum(r)
        else:
            if self.backend.bk_is_complex(x):
                r = self.backend.bk_square(
                    (self.backend.bk_real(x) - self.backend.bk_real(y)) / sigma
                )
                i = self.backend.bk_square(
                    (self.backend.bk_imag(x) - self.backend.bk_imag(y)) / sigma
                )
                return self.backend.bk_reduce_sum(r + i)
            else:
                r = self.backend.bk_square((x - y) / sigma)
                return self.backend.bk_reduce_sum(r)

    # ---------------------------------------------−---------
    def convol(self, in_image, axis=0, cell_ids=None, nside=None, spin=0):

        image = self.backend.bk_cast(in_image)

        if self.use_2D:
            ishape = list(in_image.shape)
            if len(ishape) < axis + 2:
                if not self.silent:
                    print("Use of 2D scat with data that has less than 2D")
                return None

            npix = ishape[-2]
            npiy = ishape[-1]

            ndata = 1
            for k in range(len(ishape) - 2):
                ndata = ndata * ishape[k]

            tim = self.backend.bk_reshape(
                self.backend.bk_cast(in_image), [ndata, npix, npiy]
            )

            if self.backend.bk_is_complex(tim):
                rr1 = self.backend.conv2d(self.backend.bk_real(tim), self.ww_RealT[1])
                ii1 = self.backend.conv2d(self.backend.bk_real(tim), self.ww_ImagT[1])
                rr2 = self.backend.conv2d(self.backend.bk_imag(tim), self.ww_RealT[1])
                ii2 = self.backend.conv2d(self.backend.bk_imag(tim), self.ww_ImagT[1])
                res = self.backend.bk_complex(rr1 - ii2, ii1 + rr2)
            else:
                rr = self.backend.conv2d(tim, self.ww_RealT[1])
                ii = self.backend.conv2d(tim, self.ww_ImagT[1])
                res = self.backend.bk_complex(rr, ii)

            return self.backend.bk_reshape(
                res, ishape[0:-2] + [self.NORIENT, npix, npiy]
            )

        elif self.use_1D:
            ishape = list(in_image.shape)

            npix = ishape[-1]

            ndata = 1
            for k in range(len(ishape) - 1):
                ndata = ndata * ishape[k]

            tim = self.backend.bk_reshape(self.backend.bk_cast(in_image), [ndata, npix])

            if self.backend.bk_is_complex(tim):
                rr1 = self.backend.conv1d(self.backend.bk_real(tim), self.ww_RealT[1])
                ii1 = self.backend.conv1d(self.backend.bk_real(tim), self.ww_ImagT[1])
                rr2 = self.backend.conv1d(self.backend.bk_imag(tim), self.ww_RealT[1])
                ii2 = self.backend.conv1d(self.backend.bk_imag(tim), self.ww_ImagT[1])
                res = self.backend.bk_complex(rr1 - ii2, ii1 + rr2)
            else:
                rr = self.backend.conv1d(tim, self.ww_RealT[1])
                ii = self.backend.conv1d(tim, self.ww_ImagT[1])
                res = self.backend.bk_complex(rr, ii)

            return self.backend.bk_reshape(res, ishape)

        else:
            ishape = list(image.shape)
            if nside is None:
                nside = int(np.sqrt(image.shape[-1] // 12))

            if (spin,nside) not in self.Idx_Neighbours:
                if self.InitWave is None:
                    wr, wi, ws, widx = self.init_index(nside, cell_ids=cell_ids,spin=spin)
                else:
                    wr, wi, ws, widx = self.InitWave(nside, cell_ids=cell_ids,spin=spin)

                self.Idx_Neighbours[(spin,nside)] = 1  # self.backend.bk_constant(tmp)
                self.ww_Real[(spin,nside)] = wr
                self.ww_Imag[(spin,nside)] = wi
                self.w_smooth[(spin,nside)] = ws

            l_ww_real = self.ww_Real[(spin,nside)]
            l_ww_imag = self.ww_Imag[(spin,nside)]

            # always convolve the last dimension

            ndata = 1
            if len(ishape) > 1:
                for k in range(len(ishape) - 1):
                    ndata = ndata * ishape[k]
            if spin>0:
                tim = self.backend.bk_reshape(
                    self.backend.bk_cast(image), [ndata//2,2*ishape[-1]]
                )
            else:
                tim = self.backend.bk_reshape(
                    self.backend.bk_cast(image), [ndata, ishape[-1]]
                )
            if tim.dtype == self.all_cbk_type:
                rr1 = self.backend.bk_reshape(
                    self.backend.bk_sparse_dense_matmul(
                        self.backend.bk_real(tim),
                        l_ww_real,
                    ),
                    [ndata, self.NORIENT, ishape[-1]],
                )
                ii1 = self.backend.bk_reshape(
                    self.backend.bk_sparse_dense_matmul(
                        self.backend.bk_real(tim),
                        l_ww_imag,
                    ),
                    [ndata, self.NORIENT, ishape[-1]],
                )
                rr2 = self.backend.bk_reshape(
                    self.backend.bk_sparse_dense_matmul(
                        self.backend.bk_imag(tim),
                        l_ww_real,
                    ),
                    [ndata, self.NORIENT, ishape[-1]],
                )
                ii2 = self.backend.bk_reshape(
                    self.backend.bk_sparse_dense_matmul(
                        self.backend.bk_imag(tim),
                        l_ww_imag,
                    ),
                    [ndata, self.NORIENT, ishape[-1]],
                )
                res = self.backend.bk_complex(rr1 - ii2, ii1 + rr2)
            else:
                rr = self.backend.bk_reshape(
                    self.backend.bk_sparse_dense_matmul(tim, l_ww_real),
                    [ndata, self.NORIENT, ishape[-1]],
                )
                ii = self.backend.bk_reshape(
                    self.backend.bk_sparse_dense_matmul(tim, l_ww_imag),
                    [ndata, self.NORIENT, ishape[-1]],
                )
                res = self.backend.bk_complex(rr, ii)
                
            if spin==0:
                if len(ishape) > 1:
                    return self.backend.bk_reshape(
                        res, ishape[0:-1] + [self.NORIENT, ishape[-1]]
                    )
                else:
                    return self.backend.bk_reshape(res, [self.NORIENT, ishape[-1]])
            else:
                if len(ishape) > 2:
                    return self.backend.bk_reshape(
                        res, ishape[0:-2] + [2,self.NORIENT, ishape[-1]]
                    )
                else:
                    return self.backend.bk_reshape(res, [2,self.NORIENT, ishape[-1]])
                
        return res

    # ---------------------------------------------−---------
    def smooth(self, in_image, axis=0, cell_ids=None, nside=None, spin=0):

        image = self.backend.bk_cast(in_image)

        if self.use_2D:

            ishape = list(in_image.shape)
            if len(ishape) < axis + 2:
                if not self.silent:
                    print("Use of 2D scat with data that has less than 2D")
                return None

            npix = ishape[-2]
            npiy = ishape[-1]
            
            odata = 1
            if len(ishape) > 1:
                for k in range(len(ishape)-2):
                    odata = odata * ishape[k]

            ndata = 1
            for k in range(len(ishape)-2):
                ndata = ndata * ishape[k]

            tim = self.backend.bk_reshape(
                self.backend.bk_cast(in_image), [ndata, npix, npiy]
            )

            if self.backend.bk_is_complex(tim):
                rr = self.backend.conv2d(self.backend.bk_real(tim), self.ww_SmoothT[1])
                ii = self.backend.conv2d(self.backend.bk_imag(tim), self.ww_SmoothT[1])
                res = self.backend.bk_complex(rr, ii)
            else:
                res = self.backend.conv2d(tim, self.ww_SmoothT[1])

            return self.backend.bk_reshape(res, ishape)

        elif self.use_1D:

            ishape = list(in_image.shape)

            npix = ishape[-1]
            
            ndata = 1
            for k in range(len(ishape) - 1):
                ndata = ndata * ishape[k]

            tim = self.backend.bk_reshape(self.backend.bk_cast(in_image), [ndata, npix])

            if self.backend.bk_is_complex(tim):
                rr = self.backend.conv1d(self.backend.bk_real(tim), self.ww_SmoothT[1])
                ii = self.backend.conv1d(self.backend.bk_imag(tim), self.ww_SmoothT[1])
                res = self.backend.bk_complex(rr, ii)
            else:
                res = self.backend.conv1d(tim, self.ww_SmoothT[1])
                
            return self.backend.bk_reshape(res, ishape)

        else:

            ishape = list(image.shape)
            
            if nside is None:
                nside = int(np.sqrt(image.shape[-1] // 12))

            if (spin,nside) not in self.Idx_Neighbours:
                if self.InitWave is None:
                    wr, wi, ws, widx = self.init_index(nside, cell_ids=cell_ids,spin=spin)
                else:
                    wr, wi, ws, widx = self.InitWave(nside, cell_ids=cell_ids,spin=spin)

                self.Idx_Neighbours[(spin,nside)] = 1  # self.backend.bk_constant(tmp)
                self.ww_Real[(spin,nside)] = wr
                self.ww_Imag[(spin,nside)] = wi
                self.w_smooth[(spin,nside)] = ws
                
            l_w_smooth = self.w_smooth[(spin,nside)]

            odata = 1
            for k in range(0, len(ishape) - 1):
                odata = odata * ishape[k]

            tim = self.backend.bk_reshape(image, [odata, ishape[-1]])
            if spin==0:
                if tim.dtype == self.all_cbk_type:
                    rr = self.backend.bk_sparse_dense_matmul(
                        self.backend.bk_real(tim), l_w_smooth
                    )
                    ri = self.backend.bk_sparse_dense_matmul(
                        self.backend.bk_imag(tim), l_w_smooth
                    )
                    res = self.backend.bk_complex(rr, ri)
                else:
                    res = self.backend.bk_sparse_dense_matmul(tim, l_w_smooth)
            else:
                tim=self.backend.bk_reshape(tim,[odata//2,2*tim.shape[-1]])
                if tim.dtype == self.all_cbk_type:
                    rr = self.backend.bk_sparse_dense_matmul(
                        self.backend.bk_real(tim), l_w_smooth
                    )
                    ri = self.backend.bk_sparse_dense_matmul(
                        self.backend.bk_imag(tim), l_w_smooth
                    )
                    res = self.backend.bk_complex(rr, ri)
                else:
                    res = self.backend.bk_sparse_dense_matmul(tim, l_w_smooth)
                
            if len(ishape) == 1:
                return self.backend.bk_reshape(res, [ishape[-1]])
            else:
                return self.backend.bk_reshape(res, ishape[0:-1] + [ishape[-1]])

        return res

    # ---------------------------------------------−---------
    def get_kernel_size(self):
        return self.KERNELSZ

    # ---------------------------------------------−---------
    def get_nb_orient(self):
        return self.NORIENT

    # ---------------------------------------------−---------
    def get_ww(self, nside=1):
        if self.use_2D:

            return (
                self.ww_RealT[1].reshape(self.KERNELSZ * self.KERNELSZ, self.NORIENT),
                self.ww_ImagT[1].reshape(self.KERNELSZ * self.KERNELSZ, self.NORIENT),
            )
        else:
            return (self.ww_Real[nside], self.ww_Imag[nside])

    # ---------------------------------------------−---------
    def plot_ww(self):
        c, s = self.get_ww()
        import matplotlib.pyplot as plt

        plt.figure(figsize=(16, 6))
        npt = int(np.sqrt(c.shape[0]))
        for i in range(c.shape[1]):
            plt.subplot(2, c.shape[1], 1 + i)
            plt.imshow(
                c[:, i].reshape(npt, npt), cmap="viridis", vmin=-c.max(), vmax=c.max()
            )
            plt.subplot(2, c.shape[1], 1 + i + c.shape[1])
            plt.imshow(
                s[:, i].reshape(npt, npt), cmap="viridis", vmin=-c.max(), vmax=c.max()
            )
            sys.stdout.flush()
        plt.show()
