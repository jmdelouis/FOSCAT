import numpy as np
import healpy as hp
import tensorflow as tf
import os, sys
import time
import foscat.adam as adam


class FoCUS:
    def __init__(self,
                 NORIENT=4,
                 LAMBDA=1.2,
                 KERNELSZ=3,
                 slope=1.0,
                 all_type='float64',
                 nside=32,
                 padding='SAME',
                 gpupos=0,
                 healpix=True,
                 OSTEP=0,
                 isMPI=False,
                 TEMPLATE_PATH='data'):

        print('================================================')
        print('          START FOSCAT CONFIGURATION')
        print('================================================')

        self.TEMPLATE_PATH = TEMPLATE_PATH
        self.tf = tf
        if os.path.exists(self.TEMPLATE_PATH) == False:
            print('The directory %s to store temporary information for FoCUS does not exist: Try to create it' % (
                self.TEMPLATE_PATH))
            try:
                os.system('mkdir -p %s' % (self.TEMPLATE_PATH))
                print('The directory %s is created')
            except:
                print('Impossible to create the directory %s' % (self.TEMPLATE_PATH))
                exit(0)

        self.number_of_loss = 0
        self.inpar = {}
        self.rewind = {}
        self.diff_map1 = {}
        self.diff_map2 = {}
        self.diff_mask = {}
        self.diff_weight = {}
        self.loss_type = {}
        self.loss_weight = {}

        self.MAPDIFF = 1
        self.SCATDIFF = 2
        self.NOISESTAT = 3
        self.SCATCOV = 4
        self.SCATCOEF = 5

        self.log = np.zeros([10])
        self.nlog = 0

        self.padding = padding
        self.healpix = healpix
        self.OSTEP = OSTEP
        self.nparam = 0
        self.on1 = {}
        self.on2 = {}
        self.tmpa = {}
        self.tmpb = {}
        self.tmpc = {}

        if isMPI:
            from mpi4py import MPI

            self.comm = MPI.COMM_WORLD
            self.size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()

            if all_type == 'float32':
                self.MPI_ALL_TYPE = MPI.FLOAT
            else:
                self.MPI_ALL_TYPE = MPI.DOUBLE
        else:
            self.size = 1
            self.rank = 0
        self.isMPI = isMPI

        self.tw1 = {}
        self.tw2 = {}
        self.tw3 = {}
        self.tw4 = {}
        self.tb1 = {}
        self.tb2 = {}
        self.tb3 = {}
        self.tb4 = {}

        self.ss1 = {}
        self.ss2 = {}
        self.ss3 = {}
        self.ss4 = {}

        self.os1 = {}
        self.os2 = {}
        self.os3 = {}
        self.os4 = {}
        self.is1 = {}
        self.is2 = {}
        self.is3 = {}
        self.is4 = {}

        self.NMASK = 1
        self.mask = {}
        self.all_type = all_type
        if all_type == 'float32':
            self.all_tf_type = tf.float32
            # self.MPI_ALL_TYPE=MPI.FLOAT
        else:
            if all_type == 'float64':
                self.all_type = 'float64'
                self.all_tf_type = tf.float64
                # self.MPI_ALL_TYPE=MPI.DOUBLE
            else:
                print('ERROR INIT FOCUS ', all_type, ' should be float32 or float64')
                exit(0)

        # ===========================================================================
        # INIT
        if self.rank == 0:
            print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
            sys.stdout.flush()
        tf.debugging.set_log_device_placement(False)
        tf.config.set_soft_device_placement(True)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        gpuname = 'CPU:0'
        self.gpulist = {}
        self.gpulist[0] = gpuname
        self.ngpu = 1
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                sys.stdout.flush()
                gpuname = logical_gpus[gpupos].name
                self.gpulist = {}
                self.ngpu = len(logical_gpus)
                for i in range(self.ngpu):
                    self.gpulist[i] = logical_gpus[i].name

            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        self.gpupos = (gpupos + self.rank) % self.ngpu
        print('============================================================')
        print('==                                                        ==')
        print('==                                                        ==')
        print('==     RUN ON GPU Rank %d : %s                          ==' % (
        self.rank, self.gpulist[self.gpupos % self.ngpu]))
        print('==                                                        ==')
        print('==                                                        ==')
        print('============================================================')
        sys.stdout.flush()

        self.NORIENT = NORIENT
        self.LAMBDA = LAMBDA
        self.KERNELSZ = KERNELSZ
        self.slope = slope

        wwc = np.zeros([KERNELSZ ** 2, NORIENT]).astype(all_type)
        wws = np.zeros([KERNELSZ ** 2, NORIENT]).astype(all_type)

        x = np.repeat(np.arange(KERNELSZ) - KERNELSZ // 2, KERNELSZ).reshape(KERNELSZ, KERNELSZ)
        y = x.T

        for i in range(NORIENT):
            a = i / NORIENT * np.pi
            xx = (3 / float(KERNELSZ)) * LAMBDA * (x * np.cos(a) + y * np.sin(a))
            yy = (3 / float(KERNELSZ)) * LAMBDA * (x * np.sin(a) - y * np.cos(a))

            if KERNELSZ == 5:
                w_smooth = np.exp(-4 * ((3.0 / float(KERNELSZ) * xx) ** 2 + (3.0 / float(KERNELSZ) * yy) ** 2))
            else:
                w_smooth = np.exp(-0.5 * ((3.0 / float(KERNELSZ) * xx) ** 2 + (3.0 / float(KERNELSZ) * yy) ** 2))

            tmp = np.cos(yy * np.pi) * w_smooth
            wwc[:, i] = tmp.flatten() - tmp.mean()
            tmp = np.sin(yy * np.pi) * w_smooth
            wws[:, i] = tmp.flatten() - tmp.mean()
            sigma = np.sqrt((wwc[:, i] ** 2 + wws[:, i] ** 2).mean())
            wwc[:, i] /= sigma
            wws[:, i] /= sigma

            w_smooth = w_smooth.flatten()

        self.w_smooth = tf.constant(w_smooth / w_smooth.sum())
        self.ww_Real = tf.constant(wwc)
        self.ww_Imag = tf.constant(wws)

        self.wwc = wwc
        self.wws = wws

        self.mat_avg_ang = np.zeros([NORIENT * NORIENT, NORIENT])
        for i in range(NORIENT):
            for j in range(NORIENT):
                self.mat_avg_ang[i + j * NORIENT, i] = 1.0
        self.mat_avg_ang = tf.constant(self.mat_avg_ang)

        self.Idx_Neighbours = {}
        self.pix_interp_val = {}
        self.weight_interp_val = {}
        self.ring2nest = {}

        nout = nside
        self.nout = nside
        nstep = int(np.log(nout) / np.log(2)) - self.OSTEP
        if self.rank == 0:
            print('Initialize HEALPIX synthesis NSIDE=', nout)
            sys.stdout.flush()
        self.ampnorm = {}

        for i in range(16):
            lout = (2 ** i)
            self.pix_interp_val[lout] = {}
            self.weight_interp_val[lout] = {}
            for j in range(16):
                lout2 = (2 ** j)
                self.pix_interp_val[lout][lout2] = None
                self.weight_interp_val[lout][lout2] = None
            self.ring2nest[lout] = None
            self.Idx_Neighbours[lout] = None

        self.loss = {}

    # ---------------------------------------------−---------
    # --       COMPUTE 3X3 INDEX FOR HEALPIX WORK          --
    # ---------------------------------------------−---------
    def corr_idx_wXX(self, x, y):
        idx = np.where(x == -1)[0]
        res = x
        res[idx] = y[idx]
        return (res)

    def comp_idx_w9(self, nout):

        x, y, z = hp.pix2vec(nout, np.arange(12 * nout ** 2), nest=True)
        vec = np.zeros([3, 12 * nout ** 2])
        vec[0, :] = x
        vec[1, :] = y
        vec[2, :] = z

        radius = np.sqrt(4 * np.pi / (12 * nout * nout))

        npt = 9
        outname = 'W9'

        th, ph = hp.pix2ang(nout, np.arange(12 * nout ** 2), nest=True)
        idx = hp.get_all_neighbours(nout, th, ph, nest=True)

        allidx = np.zeros([9, 12 * nout * nout], dtype='int')

        def corr(x, y):
            idx = np.where(x == -1)[0]
            res = x
            res[idx] = y[idx]
            return (res)

        allidx[4, :] = np.arange(12 * nout ** 2)
        allidx[0, :] = self.corr_idx_wXX(idx[1, :], idx[2, :])
        allidx[1, :] = self.corr_idx_wXX(idx[2, :], idx[3, :])
        allidx[2, :] = self.corr_idx_wXX(idx[3, :], idx[4, :])

        allidx[3, :] = self.corr_idx_wXX(idx[0, :], idx[1, :])
        allidx[5, :] = self.corr_idx_wXX(idx[4, :], idx[5, :])

        allidx[6, :] = self.corr_idx_wXX(idx[7, :], idx[0, :])
        allidx[7, :] = self.corr_idx_wXX(idx[6, :], idx[7, :])
        allidx[8, :] = self.corr_idx_wXX(idx[5, :], idx[6, :])

        idx = np.zeros([12 * nout * nout, npt], dtype='int')
        for iii in range(12 * nout * nout):
            idx[iii, :] = allidx[:, iii]

        np.save('%s/%s_%d_IDX.npy' % (self.TEMPLATE_PATH, outname, nout), idx)
        print('%s/%s_%d_IDX.npy COMPUTED' % (self.TEMPLATE_PATH, outname, nout))

    # ---------------------------------------------−---------
    # --       COMPUTE 5X5 INDEX FOR HEALPIX WORK          --
    # ---------------------------------------------−---------
    def comp_idx_w25(self, nout):

        x, y, z = hp.pix2vec(nout, np.arange(12 * nout ** 2), nest=True)
        vec = np.zeros([3, 12 * nout ** 2])
        vec[0, :] = x
        vec[1, :] = y
        vec[2, :] = z

        radius = np.sqrt(4 * np.pi / (12 * nout * nout))

        npt = 25
        outname = 'W25'

        th, ph = hp.pix2ang(nout, np.arange(12 * nout ** 2), nest=True)
        idx = hp.get_all_neighbours(nout, th, ph, nest=True)

        allidx = np.zeros([25, 12 * nout * nout], dtype='int')

        allidx[12, :] = np.arange(12 * nout ** 2)
        allidx[11, :] = self.corr_idx_wXX(idx[0, :], idx[1, :])
        allidx[7, :] = self.corr_idx_wXX(idx[2, :], idx[3, :])
        allidx[13, :] = self.corr_idx_wXX(idx[4, :], idx[5, :])
        allidx[17, :] = self.corr_idx_wXX(idx[6, :], idx[7, :])

        allidx[10, :] = self.corr_idx_wXX(idx[0, allidx[11, :]], idx[1, allidx[11, :]])
        allidx[6, :] = self.corr_idx_wXX(idx[2, allidx[11, :]], idx[3, allidx[11, :]])
        allidx[16, :] = self.corr_idx_wXX(idx[6, allidx[11, :]], idx[7, allidx[11, :]])

        allidx[2, :] = self.corr_idx_wXX(idx[2, allidx[7, :]], idx[3, allidx[7, :]])
        allidx[8, :] = self.corr_idx_wXX(idx[4, allidx[7, :]], idx[5, allidx[7, :]])

        allidx[14, :] = self.corr_idx_wXX(idx[4, allidx[13, :]], idx[5, allidx[13, :]])
        allidx[18, :] = self.corr_idx_wXX(idx[6, allidx[13, :]], idx[7, allidx[13, :]])

        allidx[22, :] = self.corr_idx_wXX(idx[6, allidx[17, :]], idx[7, allidx[17, :]])

        allidx[1, :] = self.corr_idx_wXX(idx[2, allidx[6, :]], idx[3, allidx[6, :]])
        allidx[5, :] = self.corr_idx_wXX(idx[0, allidx[6, :]], idx[1, allidx[6, :]])

        allidx[3, :] = self.corr_idx_wXX(idx[2, allidx[8, :]], idx[3, allidx[8, :]])
        allidx[9, :] = self.corr_idx_wXX(idx[4, allidx[8, :]], idx[5, allidx[8, :]])

        allidx[19, :] = self.corr_idx_wXX(idx[4, allidx[18, :]], idx[5, allidx[18, :]])
        allidx[23, :] = self.corr_idx_wXX(idx[6, allidx[18, :]], idx[7, allidx[18, :]])

        allidx[15, :] = self.corr_idx_wXX(idx[0, allidx[16, :]], idx[1, allidx[16, :]])
        allidx[21, :] = self.corr_idx_wXX(idx[6, allidx[16, :]], idx[7, allidx[16, :]])

        allidx[0, :] = self.corr_idx_wXX(idx[0, allidx[1, :]], idx[1, allidx[1, :]])

        allidx[4, :] = self.corr_idx_wXX(idx[4, allidx[3, :]], idx[5, allidx[3, :]])

        allidx[20, :] = self.corr_idx_wXX(idx[0, allidx[21, :]], idx[1, allidx[21, :]])

        allidx[24, :] = self.corr_idx_wXX(idx[4, allidx[23, :]], idx[5, allidx[23, :]])

        idx = np.zeros([12 * nout * nout, npt], dtype='int')
        for iii in range(12 * nout * nout):
            idx[iii, :] = allidx[:, iii]

        np.save('%s/%s_%d_IDX.npy' % (self.TEMPLATE_PATH, outname, nout), idx)
        print('%s/%s_%d_IDX.npy COMPUTED' % (self.TEMPLATE_PATH, outname, nout))

    # ---------------------------------------------−---------
    def get_rank(self):
        return (self.rank)

    # ---------------------------------------------−---------
    def get_size(self):
        return (self.size)

    # ---------------------------------------------−---------
    def barrier(self):
        if self.isMPI:
            self.comm.Barrier()

    # ---------------------------------------------−---------
    def toring(self, image, axis=0):
        lout = int(np.sqrt(image.shape[axis] // 12))

        if self.ring2nest[lout] is None:
            self.ring2nest[lout] = hp.ring2nest(lout, np.arange(12 * lout ** 2))

        return (tf.gather(image, self.ring2nest[lout], axis=axis))

    # --------------------------------------------------------
    def ud_grade_2(self, im, axis=0):

        shape = im.shape
        lout = int(np.sqrt(shape[axis] // 12))
        if im.__class__ == np.zeros([0]).__class__:
            oshape = np.zeros([len(shape) + 1], dtype='int')
            if axis > 0:
                oshape[0:axis] = shape[0:axis]
            oshape[axis] = 12 * lout * lout // 4
            oshape[axis + 1] = 4
            if len(shape) > axis:
                oshape[axis + 2:] = shape[axis + 1:]
        else:
            if axis > 0:
                oshape = shape[0:axis] + [12 * lout * lout // 4, 4]
            else:
                oshape = [12 * lout * lout // 4, 4]
            if len(shape) > axis:
                oshape = oshape + shape[axis + 1:]

        return (tf.reduce_mean(tf.reshape(im, oshape), axis=axis + 1))

    # --------------------------------------------------------
    def up_grade(self, im, nout, axis=0):
        lout = int(np.sqrt(im.shape[axis] // 12))

        if self.pix_interp_val[lout][nout] is None:
            th, ph = hp.pix2ang(nout, np.arange(12 * nout ** 2, dtype='int'), nest=True)
            p, w = hp.get_interp_weights(lout, th, ph, nest=True)
            del th
            del ph
            self.pix_interp_val[lout][nout] = p
            self.weight_interp_val[lout][nout] = w

        if lout == nout:
            imout = im
        else:
            if axis == 0:
                imout = tf.reduce_sum(tf.gather(im, self.pix_interp_val[lout][nout], axis=axis) \
                                      * self.weight_interp_val[lout][nout], axis=0)

            else:
                amap = tf.gather(im, self.pix_interp_val[lout][nout], axis=axis)

                aw = self.weight_interp_val[lout][nout]
                for k in range(axis):
                    aw = tf.expand_dims(aw, axis=0)
                for k in range(axis + 1, len(im.shape)):
                    aw = tf.expand_dims(aw, axis=-1)
                imout = tf.reduce_sum(aw * amap, axis=axis)
        return (imout)

    # ---------------------------------------------−---------
    def init_index(self, nside):
        try:
            tmp = np.load('%s/W%d_%d_IDX.npy' % (self.TEMPLATE_PATH, self.KERNELSZ ** 2, nside))
        except:
            if self.KERNELSZ ** 2 == 9:
                if self.rank == 0:
                    self.comp_idx_w9(nside)
            elif self.KERNELSZ ** 2 == 25:
                if self.rank == 0:
                    self.comp_idx_w25(nside)
            else:
                if self.rank == 0:
                    print('Only 3x3 and 5x5 kernel have been developped for Healpix and you ask for %dx%d' % (
                    KERNELSZ, KERNELSZ))
                    exit(0)
            self.barrier()
            tmp = np.load('%s/W%d_%d_IDX.npy' % (self.TEMPLATE_PATH, self.KERNELSZ ** 2, nside))

        self.Idx_Neighbours[nside] = tf.constant(tmp)

    # ---------------------------------------------−---------
    # Compute x [....,a,....] to [....,a*a,....]
    # NOT YET TESTED OR IMPLEMENTED
    def auto_cross_2(x, axis=0):
        shape = np.array(x.shape)
        if axis == 0:
            y1 = tf.reshape(x, [shape[0], 1, np.cumprod(shape[1:])])
            y2 = tf.reshape(x, [1, shape[0], np.cumprod(shape[1:])])
            oshape = np.concat([shape[0], shape[0], shape[1:]])
            return (tf.reshape(y1 * y2, oshape))

    # ---------------------------------------------−---------
    # Compute x [....,a,....,b,....] to [....,b*b,....,a*a,....]
    # NOT YET TESTED OR IMPLEMENTED
    def auto_cross_2(x, axis1=0, axis2=1):
        shape = np.array(x.shape)
        if axis == 0:
            y1 = tf.reshape(x, [shape[0], 1, np.cumprod(shape[1:])])
            y2 = tf.reshape(x, [1, shape[0], np.cumprod(shape[1:])])
            oshape = np.concat([shape[0], shape[0], shape[1:]])
            return (tf.reshape(y1 * y2, oshape))

    # ---------------------------------------------−---------
    # convert swap axes tensor x [....,a,....,b,....] to [....,b,....,a,....]
    def swapaxes(self, x, axis1, axis2):
        shape = x.shape.as_list()
        if axis1 < 0:
            laxis1 = len(shape) + axis1
        if axis2 < 0:
            laxis2 = len(shape) + axis2

        naxes = len(shape)
        thelist = [i for i in range(naxes)]
        thelist[laxis1] = laxis2
        thelist[laxis2] = laxis1
        return tf.transpose(x, thelist)

    # ---------------------------------------------−---------
    # convert tensor x [....,a,b,....] to [....,a*b,....]
    def reduce_dim(self, x, axis=0):
        shape = x.shape.as_list()
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
            oshape.extend(shape[laxis + 2:])

        return (tf.reshape(x, oshape))

    # ---------------------------------------------−---------
    def convol(self, image, axis=0):

        nside = int(np.sqrt(image.shape[axis] // 12))

        if self.Idx_Neighbours[nside] is None:
            self.init_index(nside)

        imX9 = tf.expand_dims(tf.gather(image, self.Idx_Neighbours[nside], axis=axis), -1)

        l_ww_real = self.ww_Real
        l_ww_imag = self.ww_Imag
        for i in range(axis + 1):
            l_ww_real = tf.expand_dims(l_ww_real, 0)
            l_ww_imag = tf.expand_dims(l_ww_imag, 0)

        for i in range(axis + 2, len(imX9.shape) - 1):
            l_ww_real = tf.expand_dims(l_ww_real, axis + 2)
            l_ww_imag = tf.expand_dims(l_ww_imag, axis + 2)

        rr = tf.reduce_sum(l_ww_real * imX9, axis + 1)
        ii = tf.reduce_sum(l_ww_imag * imX9, axis + 1)

        return (rr, ii)

    # ---------------------------------------------−---------
    def smooth(self, image, axis=0):

        nside = int(np.sqrt(image.shape[axis] // 12))

        if self.Idx_Neighbours[nside] is None:
            self.init_index(nside)

        imX9 = tf.gather(image, self.Idx_Neighbours[nside], axis=axis)

        l_w_smooth = self.w_smooth
        for i in range(axis + 1):
            l_w_smooth = tf.expand_dims(l_w_smooth, 0)

        for i in range(axis + 2, len(imX9.shape)):
            l_w_smooth = tf.expand_dims(l_w_smooth, axis + 2)

        res = tf.reduce_sum(l_w_smooth * imX9, axis + 1)
        return (res)

    # ---------------------------------------------−---------
    def get_kernel_size(self):
        return (self.KERNELSZ)

    # ---------------------------------------------−---------
    def get_nb_orient(self):
        return (self.NORIENT)

    # ---------------------------------------------−---------
    def get_ww(self):
        return (self.wwc, self.wws)

    # ---------------------------------------------−---------
    def plot_ww(self):
        c, s = self.get_ww()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 6))
        npt = int(np.sqrt(c.shape[0]))
        for i in range(c.shape[1]):
            plt.subplot(2, c.shape[1], 1 + i)
            plt.imshow(c[:, i].reshape(npt, npt), cmap='Greys', vmin=-0.5, vmax=1.0)
            plt.subplot(2, c.shape[1], 1 + i + c.shape[1])
            plt.imshow(s[:, i].reshape(npt, npt), cmap='Greys', vmin=-0.5, vmax=1.0)
            sys.stdout.flush()
        plt.show()

    # ---------------------------------------------−---------
    def relu(self, x):
        return tf.nn.relu(x)

    # ---------------------------------------------−---------
    def wst(self, image, mask=None, axis=0):

        # determine jmax and nside corresponding to the input map
        nside = int(np.sqrt(image.shape[axis] // 12))
        jmax = int(np.log(nside) / np.log(2)) - self.OSTEP

        if self.KERNELSZ > 3:
            # if the kernel size is bigger than 3 increase the binning before smoothing
            l_image = self.up_grade(image, nside * 2, axis=axis)
        else:
            l_image = image

        s1 = None
        s2 = None
        p00 = None
        l2_image = None

        for j1 in range(jmax):

            # Convol image along the axis defined by 'axis' using the wavelet defined at
            # the foscat initialisation
            # c_image_real is [....,Npix_j1,....,Norient]
            c_image_real, c_image_imag = self.convol(l_image, axis=axis)

            # Compute (a+ib)*(a+ib)* the last c_image column is the real and imaginary part
            conj1 = c_image_real * c_image_real + c_image_imag * c_image_imag

            # Compute l_p00 [....,....,1,Norient]
            l_p00 = tf.expand_dims(tf.reduce_mean(conj1, axis=axis), -2)

            conj1 = tf.sqrt(conj1)

            # Compute l_s1 [....,....,1,Norient]
            l_s1 = tf.expand_dims(tf.reduce_mean(conj1, axis=axis), -2)

            # Concat S1,P00 [....,....,j1,Norient]
            if s1 is None:
                s1 = l_s1
                p00 = l_p00
            else:
                s1 = tf.concat([s1, l_s1], axis=-2)
                p00 = tf.concat([p00, l_p00], axis=-2)

            # Concat l2_image [....,Npix_j1,....,j1,Norient]
            if l2_image is None:
                l2_image = tf.expand_dims(conj1, axis=-2)
            else:
                l2_image = tf.concat([tf.expand_dims(conj1, axis=-2), l2_image], axis=-2)

            # Convol l2_image [....,Npix_j1,....,j1,Norient,Norient]
            c2_image_real, c2_image_imag = self.convol(l2_image, axis=axis)

            conj2 = tf.sqrt(c2_image_real * c2_image_real + c2_image_imag * c2_image_imag)

            # Convol l_s2 [....,....,j1,Norient,Norient]
            l_s2 = tf.reduce_mean(conj2, axis=axis)

            # Concat l_s2 [....,....,j1*(j1+1)/2,Norient,Norient]
            if s2 is None:
                s2 = l_s2
            else:
                s2 = tf.concat([s2, l_s2], axis=-3)

            # Rescale l2_image [....,Npix_j1//4,....,j1,Norient]
            l2_image = self.smooth(l2_image, axis=axis)
            l2_image = self.ud_grade_2(l2_image, axis=axis)

            # Rescale l_image [....,Npix_j1//4,....]
            l_image = self.smooth(l_image, axis=axis)
            l_image = self.ud_grade_2(l_image, axis=axis)

        return (s1, p00, s2)

    def wst_cov(self, image, axis=0):
        """
        Compute the scattering covariance coefficients S1, P00, C01 and C11.
        Parameters
        ----------
        image1: tensor
            Image on which we compute the scattering coefficients [..., Npix, ...]
            Npix defines the axis where the computation is done
        Returns
        -------
        S1, P00, C01, C11
        """

        # determine jmax and nside corresponding to the input map
        nside = int(np.sqrt(image.shape[axis] // 12))
        jmax = int(np.log(nside) / np.log(2)) - self.OSTEP

        # image is [....,Npix,....] Npix is at axis
        # l_image is [....,Npix*4,....] Npix is at axis
        if self.KERNELSZ > 3:
            # if the kernel size is bigger than 3 increase the binning before smoothing
            l_image = self.up_grade(image, nside * 2, axis=axis)
        else:
            l_image = image

        S1 = None
        P00 = None
        C01 = None
        C11 = None

        I1_dic = None
        P00_dic = {}

        for j3 in range(jmax):

            # Convol image along the axis defined by 'axis' using the wavelet defined at
            # the foscat initialisation
            # c_image_real is [....,Npix_j3,....,Norient]
            c_image_real, c_image_imag = self.convol(l_image, axis=axis)

            # Compute (a+ib)*(a+ib)* the last c_image column is the real and imaginary part
            # conj1 is [....,Npix_j3,....,Norient]
            conj1 = c_image_real * c_image_real + c_image_imag * c_image_imag

            # Compute l_P00 [....,....,1,Norient]
            l_p00 = tf.expand_dims(tf.reduce_mean(conj1, axis=axis), -2)

            conj1 = tf.sqrt(conj1)

            # Compute l_s1 [....,....,1,Norient]
            l_s1 = tf.expand_dims(tf.reduce_mean(conj1, axis=axis), -2)
            if S1 is None:
                S1 = l_s1
                P00 = l_p00
            else:
                # Compute P00 [....,.....,j3,Norient]
                # Compute S1  [....,.....,j3,Norient]
                S1 = tf.concat([S1, l_s1], axis=-2)
                P00 = tf.concat([P00, l_p00], axis=-2)

            # Compute I1_dic [....,Npix_j3,.....,j3,Norient]
            if I1_dic is None:
                I1_dic = tf.expand_dims(conj1, -2)
            else:
                I1_dic = tf.concat([tf.expand_dims(conj1, -2), I1_dic], -2)

            # compute all |I*Psi_j| * Psi_j3
            # Compute I1convPsi [....,Npix_j3,.....,j3,Norient,Norient]
            I1convPsi_real, I1convPsi_imag = self.convol(I1_dic, axis=axis)

            # Convert c_image_[imag|imag] from [....,Npix_j3,....,j3,Norient] to [....,Npix_j3,.....,1,Norient,1]
            c_image_real = tf.expand_dims(c_image_real, -2)
            c_image_real = tf.expand_dims(c_image_real, -1)
            c_image_imag = tf.expand_dims(c_image_imag, -2)
            c_image_imag = tf.expand_dims(c_image_imag, -1)

            ### Compute the product (I * Psi)_j3 x (I1_j2 * Psi_j3)^*
            # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
            # Compute real_product,imag_product [....,.....,j3,Norient,Norient]
            real_product = tf.reduce_mean(I1convPsi_real * c_image_real + \
                                          I1convPsi_imag * c_image_imag, axis)
            # Compute real_product,imag_product [....,.....,j3,Norient,Norient]
            imag_product = tf.reduce_mean(I1convPsi_imag * c_image_real - \
                                          I1convPsi_real * c_image_imag, axis)

            # convert P00 from [....,.....,j3,Norient] to [....,.....,j3,Norient,1]
            P00_ori = tf.expand_dims(P00, -1)
            # convert P00 from [....,.....,j3,Norient] to [....,.....,j3,1,Norient]
            P00_ver = tf.expand_dims(P00, -2)

            # convert P00_dic [....,.....,j3,Norient,Norient]
            P00_dic = P00_ori * P00_ver
            P00_Norm = 1 / tf.sqrt(P00_dic)

            # build C01 [....,.....,j3*(j3+1),Norient,Norient]
            if C01 is None:
                C01 = tf.concat([real_product * P00_Norm, imag_product * P00_Norm], axis=-3)
            else:
                C01 = tf.concat([C01, real_product * P00_Norm, imag_product * P00_Norm], axis=-3)

            # ============================================================================
            # WARNING : tensorflow dos not know to multiply huge dimensional tensor with
            # [...,x,1,....,y,1,...]*[...,1,x,....,1,y,...]
            # thus to the next multiplication in two steps :((

            # Convert I1convPsi_[real|imag] [....,Npix_j3,.....,j3,Norient,Norient] to
            #                               [....,Npix_j3,.....,Norient,Norient,j3]
            I1convPsi_real = self.swapaxes(I1convPsi_real, -3, -1)
            I1convPsi_imag = self.swapaxes(I1convPsi_imag, -3, -1)

            # Convert I1convPsi_[real|imag] [....,Npix_j3,.....,Norient,Norient,j3] to
            #                               [....,Npix_j3,.....,Norient,Norient,1,j3,1,Norient]
            I1convPsi_real_ori = tf.expand_dims(tf.expand_dims(I1convPsi_real, -3), -2)
            I1convPsi_imag_ori = tf.expand_dims(tf.expand_dims(I1convPsi_imag, -3), -2)
            # Convert I1convPsi_[real|imag] [....,Npix_j3,.....,Norient,Norient,j3] to
            #                               [....,Npix_j3,.....,Norient,Norient,j3,1,Norient,1]
            I1convPsi_real_ver = tf.expand_dims(tf.expand_dims(I1convPsi_real, -2), -1)
            I1convPsi_imag_ver = tf.expand_dims(tf.expand_dims(I1convPsi_imag, -2), -1)

            real_product = I1convPsi_real_ori * I1convPsi_real_ver + \
                           I1convPsi_imag_ori * I1convPsi_imag_ver
            imag_product = I1convPsi_imag_ori * I1convPsi_real_ver + \
                           I1convPsi_real_ori * I1convPsi_imag_ver

            # compute product [....,Npix_j3,.....,Norient,Norient,j3,j3,Norient]
            real_product = tf.reduce_mean(real_product, axis)
            imag_product = tf.reduce_mean(imag_product, axis)

            # convert to [....,Npix_j3,.....,j3*j3,Norient,Norient,Norient]
            real_product = self.reduce_dim(
                self.swapaxes(self.swapaxes(self.swapaxes(real_product, -3, -1), -4, -2), -5, -3), -5)
            imag_product = self.reduce_dim(
                self.swapaxes(self.swapaxes(self.swapaxes(imag_product, -3, -1), -4, -2), -5, -3), -5)

            # convert P00_dic from [....,.....,j3,Norient,Norient] to  [....,.....,Norient,Norient,j3]
            P00_dic_tmp = self.swapaxes(P00_dic, -3, -1)

            # convert P00_dic_ver [....,.....,j3,j3,Norient,Norient] to  [....,.....,j3,j3,Norient,Norient,1]
            # convert P00_dic_ori [....,.....,j3,j3,Norient,Norient] to  [....,.....,j3,j3,Norient,1,Norient]
            P00_dic_ver = tf.expand_dims(tf.expand_dims(P00_dic_tmp, -3), -2)
            P00_dic_ori = tf.expand_dims(tf.expand_dims(P00_dic_tmp, -2), -1)
            P11_Norm = 1 / tf.sqrt(P00_dic_ver * P00_dic_ori)
            # convert to [....,.....,j3*j3,Norient,Norient,Norient]
            P11_Norm = self.reduce_dim(self.swapaxes(self.swapaxes(self.swapaxes(P11_Norm, -3, -1), -4, -2), -5, -3),
                                       -5)

            # compute table [....,.....,j3*j3,Norient,Norient,Norient]
            real_product = real_product * P11_Norm
            imag_product = imag_product * P11_Norm

            # build C11 [....,.....,,Norient,Norient,Norient]
            if C11 is None:
                C11 = tf.concat([real_product, imag_product], axis=-4)
            else:
                C11 = tf.concat([C11, real_product, imag_product], axis=-4)

            # build downscale the I1_dic maps [...,Npix,...,j3,Norient]
            I1_dic = self.smooth(I1_dic, axis=axis)
            I1_dic = self.ud_grade_2(I1_dic, axis=axis)

            # build downscale the l_image [...,Npix,...]
            l_image = self.smooth(l_image, axis=axis)
            l_image = self.ud_grade_2(l_image, axis=axis)

        ###### Normalize S1 and P00
        S1 = tf.math.log(S1)
        P00 = tf.math.log(P00)

        return S1, P00, C01, C11

    def get_scat_cov_coeffs(self, image1, image2=None, mask=None):
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
        Returns
        -------
        S1, P00, C01, C11 normalized
        """
        ### AUTO OR CROSS
        cross = False
        if image2 is not None:
            cross = True

        ### PARAMETERS
        BATCH_SIZE = 1
        im_shape = image1.shape
        npix = int(im_shape[0])  # Number of pixels
        n0 = int(np.sqrt(npix / 12))  # NSIDE
        J = int(np.log(np.sqrt(npix / 12)) / np.log(2))  # Number of j scales
        Jmax = J - self.OSTEP  # Number of steps for the loop on scales
        # self.KERNELSZ is the number of pixel on one side (3, 5, 7...)
        kersize2 = self.KERNELSZ ** 2  # Kernel size square (9, 25, 49...)

        ### LOCAL VARIABLES (IMAGES and MASK)
        # Check if image1 is [Npix] or [Nbatch,Npix]
        if len(image1.shape) == 1:
            # image1 is [Nbatch, Npix]
            I1 = image1[None, :]  # Local image1 [Nbatch, Npix]
            if cross:
                I2 = image2[None, :]  # Local image2 [Nbatch, Npix]
        else:
            I1 = image1
            if cross:
                I2 = image2

        # self.mask is [Nmask, Npix]
        if mask is None:
            vmask = tf.ones([1, npix], dtype=tf.float64)
        else:
            vmask = tf.constant(mask)  # [Nmask, Npix]

        if self.KERNELSZ > 3:
            # if the kernel size is bigger than 3 increase the binning before smoothing
            I1 = self.up_grade(I1, n0 * 2, axis=1)
            vmask = self.up_grade(vmask, n0 * 2, axis=1)
            if cross:
                I2 = self.up_grade(I2, n0 * 2, axis=1)
        # Normalize the masks because they have different pixel numbers
        vmask /= tf.reduce_sum(vmask, axis=1)[:, None]  # [Nmask, Npix]

        ### COEFFS INITIALIZATION
        S1, P00, C01, C11 = None, None, None, None
        M1_dic, P1_dic = {}, {}  # M stands for Module
        if cross:
            M2_dic, P2_dic = {}, {}

        #### COMPUTE S1, P00, C01 and C11
        nside_j3 = n0  # NSIDE start (nside_j3 = n0 / 2^j3)
        npix_j3 = npix  # Pixel number at each iteration on j3
        for j3 in range(Jmax):
            print(f'Nside_j3={nside_j3}')

            ### Make the convolution I1 * Psi_j3
            cconv1, sconv1, M1_square, M1 = self._compute_IconvPsi(I1, nside_j3, BATCH_SIZE, npix_j3, kersize2)
            # Store M1_j3 in a dictionary
            M1_dic[j3] = M1

            if cross:
                ### Make the convolution I2 * Psi_j3
                cconv2, sconv2, M2_square, M2 = self._compute_IconvPsi(I2, nside_j3, BATCH_SIZE, npix_j3, kersize2)
                # Store M2_j3 in a dictionary
                M2_dic[j3] = M2

            ####### S1 and P00
            ### P00_auto = < M1^2 >_pix
            #  M1_square [Nbatch,Npix,Norient3]
            #  vmask  [Nmask,Npix]

            p00 = tf.reduce_sum(vmask[None, :, :, None] * M1_square[:, None, :, :],
                                axis=2)  # [Nbatch, Nmask, Norient3]
            # We store it for normalisation of C01 and C11
            P1_dic[j3] = p00  # [Nbatch, Nmask, Norient3]

            if not cross:
                # We store P00_auto to be returned
                if P00 is None:
                    P00 = p00[:, :, None, :]  # Add a dimension for NP00
                else:
                    P00 = tf.concat([P00, p00[:, :, None, :]], axis=2)

                #### S1_auto computation
                ### Image 1 : S1 = < M1 >_pix
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                s1 = tf.reduce_sum(vmask[None, :, :, None] * M1[:, None, :, :],
                                   axis=2)  # [Nbatch, Nmask, Norient3]  # [Nbatch, Nmask, Norient3]
                ### We store S1 for image1
                if S1 is None:
                    S1 = s1[:, :, None, :]  # Add a dimension for NS1
                else:
                    S1 = tf.concat([S1, s1[:, :, None, :]], axis=2)

            else:
                ### P00_auto = < M2^2 >_pix
                p00 = tf.reduce_sum(vmask[None, :, :, None] * M2_square[:, None, :, :],
                                    axis=2)  # [Nbatch, Nmask, Norient3]
                # We store it for normalisation
                P2_dic[j3] = p00  # [Nbatch, Nmask, Norient3]

                ### P00_cross = < (I1 * Psi_j3) (I2 * Psi_j3)^* >_pix
                # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
                p00_real = cconv1 * cconv2 + sconv1 * sconv2
                p00_imag = sconv1 * cconv2 - cconv1 * sconv2
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                p00_real = tf.reduce_sum(vmask[None, :, :, None] * p00_real[:, None, :, :],
                                         axis=2)  # [Nbatch, Nmask, Norient3]
                p00_imag = tf.reduce_sum(vmask[None, :, :, None] * p00_imag[:, None, :, :],
                                         axis=2)  # [Nbatch, Nmask, Norient3]
                ### We store P00_cross
                if P00 is None:
                    P00 = tf.concat([p00_real[:, :, None, :], p00_imag[:, :, None, :]],
                                    axis=2)  # Add a dimension for NP00
                else:
                    P00 = tf.concat([P00, p00_real[:, :, None, :], p00_imag[:, :, None, :]], axis=2)

            # Initialize dictionaries for |I1*Psi_j| * Psi_j3
            cM1convPsi_dic = {}
            sM1convPsi_dic = {}
            if cross:
                # Initialize dictionaries for |I2*Psi_j| * Psi_j3
                cM2convPsi_dic = {}
                sM2convPsi_dic = {}

            ###### C01
            for j2 in range(0, j3):  # j2 <= j3
                ### C01_auto = < (I1 * Psi)_j3 x (|I1 * psi2| * Psi_j3)^* >_pix
                if not cross:
                    cc01, sc01 = self._compute_C01_auto(j2, cconv1, sconv1, vmask,
                                                        M1_dic, cM1convPsi_dic, sM1convPsi_dic,
                                                        nside_j3, BATCH_SIZE, npix_j3, kersize2)
                    ### Normalize C01 with P00 [Nbatch, Nmask, Norient]
                    cc01 /= (P1_dic[j2][:, :, :, None] *
                             P1_dic[j3][:, :, None, :]) ** 0.5  # [Nbatch, Nmask, Norient2, Norient3]
                    sc01 /= (P1_dic[j2][:, :, :, None] *
                             P1_dic[j3][:, :, None, :]) ** 0.5  # [Nbatch, Nmask, Norient2, Norient3]
                    ### Store C01
                    if C01 is None:
                        C01 = tf.concat([cc01[:, :, None, :, :], sc01[:, :, None, :, :]],
                                        axis=2)  # Add a dimension for NC01
                    else:
                        C01 = tf.concat([C01, cc01[:, :, None, :, :], sc01[:, :, None, :, :]],
                                        axis=2)  # Add a dimension for NC01

                    ### C01_cross = < (I1 * Psi)_j3 x (|I2 * psi2| * Psi_j3)^* >_pix
                    ### C01_cross_bis = < (I2 * Psi)_j3 x (|I1 * psi2| * Psi_j3)^* >_pix
                else:
                    cc01, sc01, cc01_bis, sc01_bis = self._compute_C01_cross(j2, cconv1, sconv1, cconv2, sconv2, vmask,
                                                                             M1_dic, M2_dic,
                                                                             cM1convPsi_dic, sM1convPsi_dic,
                                                                             cM2convPsi_dic, sM2convPsi_dic,
                                                                             nside_j3, BATCH_SIZE, npix_j3, kersize2)
                    ### Normalize C01 with P00 [Nbatch, Nmask, Norient]
                    cc01 /= (P2_dic[j2][:, :, :, None] *
                             P1_dic[j3][:, :, None, :]) ** 0.5  # [Nbatch, Nmask, Norient2, Norient3]
                    sc01 /= (P2_dic[j2][:, :, :, None] *
                             P1_dic[j3][:, :, None, :]) ** 0.5  # [Nbatch, Nmask, Norient2, Norient3]
                    cc01_bis /= (P1_dic[j2][:, :, :, None] *
                                 P2_dic[j3][:, :, None, :]) ** 0.5  # [Nbatch, Nmask, Norient2, Norient3]
                    sc01_bis /= (P1_dic[j2][:, :, :, None] *
                                 P2_dic[j3][:, :, None, :]) ** 0.5  # [Nbatch, Nmask, Norient2, Norient3]
                    ### Store C01
                    if C01 is None:
                        C01 = tf.concat([cc01[:, :, None, :, :], sc01[:, :, None, :, :],
                                         cc01_bis[:, :, None, :, :], sc01_bis[:, :, None, :, :]],
                                        axis=2)  # Add a dimension for NC01
                    else:
                        C01 = tf.concat([C01,
                                         cc01[:, :, None, :, :], sc01[:, :, None, :, :],
                                         cc01_bis[:, :, None, :, :], sc01_bis[:, :, None, :, :]],
                                        axis=2)  # Add a dimension for NC01

                ##### C11
                for j1 in range(0, j2):  # j1 <= j2
                    ### C11_auto = <(|I1 * psi1| * psi3)(|I1 * psi2| * psi3)^*>
                    if not cross:
                        cc11, sc11 = self._compute_C11_auto(j1, j2, vmask, cM1convPsi_dic, sM1convPsi_dic)
                        ### Normalize C11 with P00_j [Nbatch, Nmask, Norient_j]
                        cc11 /= (P1_dic[j1][:, :, None, :, None] *
                                 P1_dic[j2][:, :, None, None,
                                 :]) ** 0.5  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        sc11 /= (P1_dic[j1][:, :, None, :, None] *
                                 P1_dic[j2][:, :, None, None,
                                 :]) ** 0.5  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        # We store C11
                        if C11 is None:
                            C11 = tf.concat([cc11[:, :, None, :, :, :], sc11[:, :, None, :, :, :]],
                                            axis=2)  # Add a dimension for NC11
                        else:
                            C11 = tf.concat([C11, cc11[:, :, None, :, :, :], sc11[:, :, None, :, :, :]],
                                            axis=2)  # Add a dimension for NC11

                        ### C11_cross = <(|I1 * psi1| * psi3)(|I2 * psi2| * psi3)^*>
                    else:
                        cc11, sc11 = self._compute_C11_cross(j1, j2, vmask,
                                                             cM1convPsi_dic, sM1convPsi_dic,
                                                             cM2convPsi_dic, sM2convPsi_dic)
                        ### Normalize C11 with P00_j [Nbatch, Nmask, Norient_j]
                        cc11 /= (P1_dic[j1][:, :, None, :, None] *
                                 P2_dic[j2][:, :, None, None,
                                 :]) ** 0.5  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        sc11 /= (P1_dic[j1][:, :, None, :, None] *
                                 P2_dic[j2][:, :, None, None,
                                 :]) ** 0.5  # [Nbatch, Nmask, Norient3, Norient2, Norient1]

                        # We store C11
                        if C11 is None:
                            C11 = tf.concat([cc11[:, :, None, :, :, :], sc11[:, :, None, :, :, :]],
                                            axis=2)  # Add a dimension for NC11
                        else:
                            C11 = tf.concat([C11, cc11[:, :, None, :, :, :], sc11[:, :, None, :, :, :]],
                                            axis=2)  # Add a dimension for NC11

            ###### Reshape for next iteration on j3
            ### Image I1,
            # downscale the I1 [Nbatch,Npix_j3]
            I1_smooth = self.smooth(I1, axis=1)
            I1 = self.ud_grade_2(I1_smooth, axis=1)

            """
            I1_smooth = tf.reshape(tf.gather(I1, self.Idx_Neighbours[nside_j3], axis=1),
                               [BATCH_SIZE, npix_j3, kersize2])  # [Nbatch, Npix_j3, kersize2]
            # Convolution with self.w_smooth [1, 1, kersize2]
            I1_smooth = tf.reduce_sum(self.w_smooth[None, None, :] * I1_smooth, axis=2)  # [Nbatch, Npix_j3]
            I1 = tf.reduce_mean(tf.reshape(I1_smooth, [BATCH_SIZE, npix_j3 // 4, 4]), axis=2)  # [Nbatch, Npix_j3]
            """

            ### Image I2
            if cross:
                I2_smooth = self.smooth(I2, axis=1)
                I2 = self.ud_grade_2(I2_smooth, axis=1)
                """
                I2_smooth = tf.reshape(tf.gather(I2, self.Idx_Neighbours[nside_j3], axis=1),
                                       [BATCH_SIZE, npix_j3, kersize2])  # [Nbatch, Npix_j3, kersize2]
                # Convolution with self.w_smooth [1, 1, kersize2]
                I2_smooth = tf.reduce_sum(self.w_smooth[None, None, :] * I2_smooth, axis=2)  # [Nbatch, Npix_j3]
                I2 = tf.reduce_mean(tf.reshape(I2_smooth,
                                               [BATCH_SIZE, npix_j3 // 4, 4]), axis=2)  # [Nbatch, Npix_j3]
                """
            ### Modules
            for j2 in range(0, j3 + 1):  # j2 <= j3
                ### Dictionary M1_dic[j2]
                M1_smooth = self.smooth(M1_dic[j2], axis=1)  # [Nbatch, Npix_j3, Norient3]
                M1_dic[j2] = self.ud_grade_2(M1_smooth, axis=1)  # [Nbatch, Npix_j3, Norient3]

                """
                M1_smooth = tf.reshape(tf.gather(M1_dic[j2], self.Idx_Neighbours[nside_j3], axis=2),
                                       [BATCH_SIZE, self.NORIENT, npix_j3, kersize2])  # [Nbatch, Norient3, Npix_j3, kersize2]
                # Convolution with self.w_smooth [1, 1,1,kersize2]
                M1_smooth = tf.reduce_sum(self.w_smooth[None,None,None,:] * M1_smooth, axis=3)  # Real part [Nbatch, Norient3, Npix_j3]
                M1_dic[j2] = tf.reduce_mean(tf.reshape(M1_smooth,
                                                       [BATCH_SIZE, self.NORIENT, npix_j3 // 4, 4]), axis=3)  # [Nbatch, Norient3, Npix]
                """
                ### Dictionary M2_dic[j2]
                if cross:
                    M2_smooth = self.smooth(M2_dic[j2], axis=1)  # [Nbatch, Npix_j3, Norient3]
                    M2_dic[j2] = self.ud_grade_2(M2_smooth, axis=1)  # [Nbatch, Npix_j3, Norient3]
                    """
                    M2_smooth = tf.reshape(tf.gather(M2_dic[j2], self.Idx_Neighbours[nside_j3], axis=2),
                                           [BATCH_SIZE, self.NORIENT, npix_j3, kersize2])  # [Nbatch, Norient3, Npix_j3, kersize2]
                    # Convolution with self.w_smooth [1, 1,1,kersize2]
                    M2_smooth = tf.reduce_sum(self.w_smooth[None, None, None, :] * M2_smooth,
                                              axis=3)  # Real part [Nbatch, Norient3, Npix_j3]
                    M2_dic[j2] = tf.reduce_mean(tf.reshape(M2_smooth,
                                                           [BATCH_SIZE, self.NORIENT, npix_j3 // 4, 4]),
                                                axis=3)  # [Nbatch, Norient3, Npix_j3]
                    """
            ### Mask
            # vmask = tf.reduce_mean(tf.reshape(vmask, [self.NMASK, npix_j3 // 4, 4]), axis=2)  # [Nmask, Npix_j3]
            vmask = self.ud_grade_2(vmask, axis=1)

            ### NSIDE_j3 and npix_j3
            nside_j3 = nside_j3 // 2
            npix_j3 = 12 * nside_j3 ** 2

        #### For test
        if cross:
            print(P00.shape, C01.shape, C11.shape)
        else:
            print(S1.shape, P00.shape, C01.shape, C11.shape)

        ###### Normalize S1 and P00
        P00 = tf.math.log(P00)
        if not cross:
            S1 = tf.math.log(S1)

            return S1, P00, C01, C11
        else:
            return P00, C01, C11

    def _compute_IconvPsi(self, I, nside_j3, BATCH_SIZE, npix_j3, kersize2):
        """
        Make the convolution I * Psi_j3
        Returns
        -------
        Use convol function
        """
        """
        # self.widx2[nside_j3] is [Npix_j3 x kersize2]
        alim = tf.reshape(tf.gather(I, self.Idx_Neighbours[nside_j3], axis=1),
                          [BATCH_SIZE, 1, npix_j3, kersize2])  # [Nbatch, 1, Npix_j3, kersize2]
        # Convolution with ww_Real [1, Norient3, 1, kersize2]
        cconv = tf.reduce_sum(self.ww_Real * alim, axis=3)  # Real part [Nbatch, Norient3, Npix_j3]
        sconv = tf.reduce_sum(self.ww_Imag * alim, axis=3)  # Imag part [Nbatch, Norient3, Npix_j3]
        """
        cconv, sconv = self.convol(I, axis=1)

        # Module square |I * Psi_j3|^2
        M_square = cconv * cconv + sconv * sconv  # [Nbatch, Npix_j3, Norient3]
        # Module |I * Psi_j3|
        M = tf.sqrt(M_square)  # [Nbatch, Norient3, Npix_j3]
        return cconv, sconv, M_square, M

    def _compute_C01_auto(self, j2, cconv1, sconv1, vmask, M1_dic, cM1convPsi_dic, sM1convPsi_dic,
                          nside_j3, BATCH_SIZE, npix_j3, kersize2):
        ### Compute |I1 * psi2| * Psi_j3 = M1_j2 * Psi_j3
        # Warning: M1_dic[j2] is already at j3 resolution [Nbatch, Npix_j3, Norient3]
        # self.widx2[nside_j3] is [Npix_j3 x kersize2]
        """
        M1convPsi = tf.reshape(tf.gather(M1_dic[j2], self.Idx_Neighbours[nside_j3], axis=2),
                               [BATCH_SIZE, self.NORIENT, 1, npix_j3, kersize2])  # [Nbatch, Norient2, 1, Npix_j3, kersize2]
        # Do the convolution with wcos, wsin  [1, Norient3, 1, kersize2]
        cM1convPsi = tf.reduce_sum(self.ww_Real[None, ...] * M1convPsi,
                                   axis=4)  # Real [Nbatch, Norient2, Norient3, Npix_j3]
        sM1convPsi = tf.reduce_sum(self.ww_Imag[None, ...] * M1convPsi,
                                   axis=4)# Imag [Nbatch, Norient2, Norient3, Npix_j3]
        """
        cM1convPsi, sM1convPsi = self.convol(M1_dic[j2], axis=1)
        # Store it so we can use it in C11 computation
        cM1convPsi_dic[j2] = cM1convPsi  # [Nbatch, Npix_j3,Norient3, Norient2]
        sM1convPsi_dic[j2] = sM1convPsi  # [Nbatch, Npix_j3,Norient3, Norient2]

        ### Compute the product (I1 * Psi)_j3 x (M1_j2 * Psi_j3)^*
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        # cconv1, sconv1 are [Nbatch, Npix_j3, Norient3]
        cc01 = cconv1[:, :, :, None] * cM1convPsi + \
               sconv1[:, :, :, None] * sM1convPsi  # Real [Nbatch, Npix_j3, Norient3, Norient2]
        sc01 = sconv1[:, :, :, None] * cM1convPsi - \
               cconv1[:, :, :, None] * sM1convPsi  # Imag [Nbatch, Npix_j3, Norient3, Norient2]

        ### Sum over pixels after applying the mask [Nmask, Npix_j3]
        cc01 = tf.reduce_sum(vmask[None, :, :, None, None] *
                             cc01[:, None, :, :, :], axis=2)  # Real [Nbatch, Nmask, Norient3, Norient2]
        sc01 = tf.reduce_sum(vmask[None, :, :, None, None] *
                             sc01[:, None, :, :, :], axis=2)  # Imag [Nbatch, Nmask, Norient3, Norient2]
        return cc01, sc01

    def _compute_C01_cross(self, j2, cconv1, sconv1, cconv2, sconv2,
                           vmask, M1_dic, M2_dic,
                           cM1convPsi_dic, sM1convPsi_dic,
                           cM2convPsi_dic, sM2convPsi_dic,
                           nside_j3, BATCH_SIZE, npix_j3, kersize2):
        """
        Compute the C01 cross coefficients
        C01_cross = < (I2 * Psi)_j3 x (|I1 * psi2| * Psi_j3)^* >_pix
        C01_cross_bis = < (I1 * Psi)_j3 x (|I2 * psi2| * Psi_j3)^* >_pix
        Parameters
        ----------
        Returns
        -------
        cc01, sc01, cc01_bis, sc01_bis: real and imag parts of C01 cross coeff
        """

        ####### C01_cross
        ### Compute |I1 * psi2| * Psi_j3 = M1_j2 * Psi_j3
        # Warning: M1_dic[j2] is already at j3 resolution [Nbatch, Norient3, Npix_j3]
        # self.widx2[nside_j3] is [Npix_j3 x kersize2]
        """
        M1convPsi = tf.reshape(tf.gather(M1_dic[j2], self.Idx_Neighbours[nside_j3], axis=2),
                               [BATCH_SIZE, self.NORIENT, 1, npix_j3, kersize2])  # [Nbatch, Norient2, 1, Npix_j3, kersize2]
        # Do the convolution with wcos, wsin  [1, Norient3, 1, kersize2]
        cM1convPsi = tf.reduce_sum(self.ww_Real[None, ...] * M1convPsi,
                                   axis=4)  # Real [Nbatch, Norient2, Norient3, Npix_j3]
        sM1convPsi = tf.reduce_sum(self.ww_Imag[None, ...] * M1convPsi,
                                   axis=4)  # Imag [Nbatch, Norient2, Norient3, Npix_j3]
        """
        cM1convPsi, sM1convPsi = self.convol(M1_dic[j2], axis=1)

        # Store it so we can use it in C11 computation
        cM1convPsi_dic[j2] = cM1convPsi  # [Nbatch, Npix_j3, Norient3, Norient2]
        sM1convPsi_dic[j2] = sM1convPsi  # [Nbatch, Npix_j3, Norient3, Norient2]
        ### Compute the product (I2 * Psi)_j3 x (M1_j2 * Psi_j3)^*
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        # cconv1, sconv1 are [Nbatch, Npix_j3, Norient3]
        # cM1convPsi, sM1convPsi are [Nbatch, Npix_j3, Norient3,Norient2]

        cc01 = cconv2[:, :, :, None] * cM1convPsi + \
               sconv2[:, :, :, None] * sM1convPsi  # Real [Nbatch, Npix_j3, Norient3, Norient2]
        sc01 = sconv2[:, :, :, None] * cM1convPsi - \
               cconv2[:, :, :, None] * sM1convPsi  # Imag [Nbatch, Npix_j3, Norient3, Norient2]

        ### Sum over pixels after applying the mask [Nmask, Npix_j3]
        cc01 = tf.reduce_sum(vmask[None, :, :, None, None] *
                             cc01[:, None, :, :, :], axis=2)  # Real [Nbatch, Nmask, Norient3, Norient2]
        sc01 = tf.reduce_sum(vmask[None, :, :, None, None] *
                             sc01[:, None, :, :, :], axis=2)  # Imag [Nbatch, Nmask, Norient3, Norient2]

        ####### C01_cross_bis
        ### Compute |I2 * psi2| * Psi_j3 = M2_j2 * Psi_j3
        # Warning: M2_dic[j2] is already at j3 resolution [Nbatch, Norient3, Npix_j3]
        # self.widx2[nside_j3] is [Npix_j3 x kersize2]
        """
        M2convPsi = tf.reshape(tf.gather(M2_dic[j2], self.Idx_Neighbours[nside_j3], axis=2),
                               [BATCH_SIZE, self.NORIENT, 1, npix_j3, kersize2])  # [Nbatch, Norient2, 1, Npix_j3, kersize2]
        # Do the convolution with wcos, wsin  [1, Norient3, 1, kersize2]
        cM2convPsi = tf.reduce_sum(self.ww_Real[None, ...] * M2convPsi,
                                   axis=4)  # Real [Nbatch, Norient2, Norient3, Npix_j3]
        sM2convPsi = tf.reduce_sum(self.ww_Imag[None, ...] * M2convPsi,
                                   axis=4)  # Imag [Nbatch, Norient2, Norient3, Npix_j3]
        """
        cM2convPsi, sM2convPsi = self.convol(M2_dic[j2], axis=1)

        # Store it so we can use it in C11 computation
        cM2convPsi_dic[j2] = cM2convPsi  # [Nbatch, Npix_j3, Norient3, Norient2]
        sM2convPsi_dic[j2] = sM2convPsi  # [Nbatch, Npix_j3, Norient3, Norient2]
        ### Compute the product (I1 * Psi)_j3 x (M2_j2 * Psi_j3)^*
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        # cconv1, sconv1 are [Nbatch, Npix_j3, Norient3]
        cc01_bis = cconv1[:, :, :, None] * cM2convPsi + \
                   sconv1[:, :, :, None] * sM2convPsi  # Real [Nbatch, Npix_j3, Norient3, Norient2]
        sc01_bis = sconv1[:, :, :, None] * cM2convPsi - \
                   cconv1[:, :, :, None] * sM2convPsi  # Imag [Nbatch, Npix_j3, Norient3, Norient2]

        ### Sum over pixels after applying the mask [Nmask, Npix_j3]
        cc01_bis = tf.reduce_sum(vmask[None, :, :, None, None] *
                                 cc01_bis[:, None, :, :, :], axis=2)  # Real [Nbatch, Nmask, Norient3, Norient2]
        sc01_bis = tf.reduce_sum(vmask[None, :, :, None, None] *
                                 sc01_bis[:, None, :, :, :], axis=2)  # Imag [Nbatch, Nmask, Norient3, Norient2]
        return cc01, sc01, cc01_bis, sc01_bis

    def _compute_C11_auto(self, j1, j2, vmask, cM1convPsi_dic, sM1convPsi_dic):
        ### Compute the product (|I1 * psi1| * psi3)(|I1 * psi2| * psi3)
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        # cM1convPsi_dic[j] is [Nbatch, Npix_j3, Norient, Norient3]
        cc11 = cM1convPsi_dic[j1][:, :, :, None, :] * cM1convPsi_dic[j2][:, :, :, :, None] + \
               sM1convPsi_dic[j1][:, :, :, None, :] * sM1convPsi_dic[j2][:, :, :, :,
                                                      None]  # Real [Nbatch, Npix_j3, Norient3, Norient2, Norient1]
        sc11 = sM1convPsi_dic[j1][:, :, :, None, :] * cM1convPsi_dic[j2][:, :, :, :, None] - \
               cM1convPsi_dic[j1][:, :, :, None, :] * sM1convPsi_dic[j2][:, :, :, :,
                                                      None]  # Imag [Nbatch, Npix_j3, Norient3, Norient2, Norient1]
        ### Sum over pixels and apply the mask
        cc11 = tf.reduce_sum(vmask[None, :, :, None, None, None] *
                             cc11[:, None, :, :, :, :],
                             axis=2)  # Real [Nbatch, Nmask, Norient1, Norient2, Norient3]
        sc11 = tf.reduce_sum(vmask[None, :, :, None, None, None] *
                             sc11[:, None, :, :, :, :],
                             axis=2)  # Imag [Nbatch, Nmask, Norient1, Norient2, Norient3]
        return cc11, sc11

    def _compute_C11_cross(self, j1, j2, vmask, cM1convPsi_dic, sM1convPsi_dic, cM2convPsi_dic, sM2convPsi_dic):

        ### Compute the product (|I1 * psi1| * psi3)(|I2 * psi2| * psi3)
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        # cM1convPsi_dic[j] is [Nbatch, Norient, Norient3, Npix_j3]
        cc11 = cM1convPsi_dic[j1][:, :, :, None, :] * cM2convPsi_dic[j2][:, :, :, :, None] + \
               sM1convPsi_dic[j1][:, :, :, None, :] * sM2convPsi_dic[j2][:, :, :, :,
                                                      None]  # Real [Nbatch, Npix_j3, Norient3, Norient2, Norient1]
        sc11 = sM1convPsi_dic[j1][:, :, :, None, :] * cM2convPsi_dic[j2][:, :, :, :, None] - \
               cM1convPsi_dic[j1][:, :, :, None, :] * sM2convPsi_dic[j2][:, :, :, :,
                                                      None]  # Imag [Nbatch, Npix_j3, Norient3, Norient2, Norient1]
        ### Sum over pixels and apply the mask
        cc11 = tf.reduce_sum(vmask[None, :, :, None, None, None] *
                             cc11[:, None, :, :, :, :],
                             axis=2)  # Real [Nbatch, Nmask, Norient3, Norient2, Norient1]
        sc11 = tf.reduce_sum(vmask[None, :, :, None, None, None] *
                             sc11[:, None, :, :, :, :],
                             axis=2)  # Imag [Nbatch, Nmask, Norient3, Norient2, Norient1]
        return cc11, sc11

    # ---------------------------------------------−---------
    def init_variable(self, var):
        self.param = tf.Variable(var)
        return self.param

    # ---------------------------------------------−---------
    def check_dense(self, data, datasz):
        s = '%s' % (type(data))
        if 'Index' in s:
            data = tf.math.bincount(tf.cast(data.indices, tf.int32), \
                                    weights=data.values,
                                    minlength=datasz)
        return data

    # ---------------------------------------------−---------
    @tf.function
    def loss_wst(self, x, s1, p0, s2, axis=0):

        r1, r0, r2 = self.wst(x, axis=axis)

        l = tf.reduce_mean(tf.square(s1 - r1)) + \
            tf.reduce_mean(tf.square(s2 - r2)) + \
            tf.reduce_mean(tf.square(p0 - r0))

        if axis == 1:
            for k in range(x.shape[0]):
                l_g = self.check_dense(tf.gradients(l, x[k])[0], x.shape[axis])
                print(l_g)
                l_g = tf.expand_dims(l_g, 0)
                if k == 0:
                    g = l_g
                else:
                    g = tf.concat([g, l_g], 0)
        else:
            g = self.check_dense(tf.gradients(l, x)[0], x.shape[axis])

        return l, g

    # ---------------------------------------------−---------
    @tf.function
    def loss_wst_cov(self, x, s1, p0, c01, c11, axis=0):

        r1, r0, r01, r11 = self.wst_cov(x, axis=axis)

        l = tf.reduce_mean(tf.square(s1 - r1)) + \
            tf.reduce_mean(tf.square(p0 - r0)) + \
            tf.reduce_mean(tf.square(c01 - r01)) + \
            tf.reduce_mean(tf.square(c11 - r11))

        if axis == 1:
            for k in range(x.shape[0]):
                l_g = self.check_dense(tf.gradients(l, x[k])[0], x.shape[axis])
                print(l_g)
                l_g = tf.expand_dims(l_g, 0)
                if k == 0:
                    g = l_g
                else:
                    g = tf.concat([g, l_g], 0)
        else:
            g = self.check_dense(tf.gradients(l, x)[0], x.shape[axis])

        return l, g

    # ---------------------------------------------−---------
    def add_loss_wst(self, s1, p0, s2, axis=0):

        for i in range(axis):
            s1 = tf.expand_dims(s1, 0)
            s2 = tf.expand_dims(s2, 0)
            p0 = tf.expand_dims(p0, 0)

        self.loss[self.number_of_loss] = {'type': 'loss_wst',
                                          's1': s1,
                                          's2': s2,
                                          'p0': p0,
                                          'axis': axis}
        self.number_of_loss = self.number_of_loss + 1
        return (self.number_of_loss - 1)

    # ---------------------------------------------−---------
    def add_loss_wst_cov(self, s1, p0, c01, c11, axis=0):

        for i in range(axis):
            s1 = tf.expand_dims(s1, 0)
            p0 = tf.expand_dims(p0, 0)
            c01 = tf.expand_dims(c01, 0)
            c11 = tf.expand_dims(c11, 0)

        self.loss[self.number_of_loss] = {'type': 'loss_wst_cov',
                                          's1': s1,
                                          'p0': p0,
                                          'c01': c01,
                                          'c11': c11,
                                          'axis': axis}
        self.number_of_loss = self.number_of_loss + 1
        return (self.number_of_loss - 1)

    # ---------------------------------------------−---------
    def learnv2(self,
                NUM_EPOCHS=1000,
                DECAY_RATE=0.95,
                EVAL_FREQUENCY=100,
                DEVAL_STAT_FREQUENCY=1000,
                LEARNING_RATE=0.03,
                EPSILON=1E-7):

        opt = adam.adam(eta=LEARNING_RATE, \
                        epsilon=EPSILON, \
                        decay_rate=DECAY_RATE)

        start = time.time()

        for itt in range(NUM_EPOCHS):
            grad = None
            for k in range(self.number_of_loss):
                ltot = 0
                if self.loss[0]['type'] == 'loss_wst':
                    l, g = self.loss_wst(self.param, \
                                         self.loss[k]['s1'], \
                                         self.loss[k]['p0'], \
                                         self.loss[k]['s2'], \
                                         axis=self.loss[k]['axis'])

                if self.loss[0]['type'] == 'loss_wst_cov':
                    l, g = self.loss_wst_cov(self.param, \
                                             self.loss[k]['s1'], \
                                             self.loss[k]['p0'], \
                                             self.loss[k]['c01'], \
                                             self.loss[k]['c11'], \
                                             axis=self.loss[k]['axis'])
                if grad is None:
                    grad = g
                else:
                    grad = grad + g

                ltot = ltot + l.numpy()

            if self.nlog == self.log.shape[0]:
                new_log = np.zeros([self.log.shape[0] * 2])
                new_log[0:self.nlog] = self.log
                self.log = new_log
            self.log[self.nlog] = ltot
            self.nlog = self.nlog + 1

            self.param = self.param - opt.update(g)

            if itt % EVAL_FREQUENCY == 0:
                end = time.time()
                print('Itt %d L=%.3g %.3fs' % (itt, ltot, (end - start)))
                start = time.time()

        return (self.param)

    # ---------------------------------------------−---------
    def get_log(self):
        return (self.log[0:self.nlog])

    # ---------------------------------------------−---------
    def get_map(self, idx=0):

        return (self.param.numpy())