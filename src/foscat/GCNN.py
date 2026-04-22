import pickle

import numpy as np

import foscat.scat_cov as sc


class GCNN:

    def __init__(
        self,
        nparam=1,
        KERNELSZ=3,
        NORIENT=4,
        chanlist=[],
        in_nside=1,
        SEED=1234,
        filename=None,
    ):

        if filename is not None:
            outlist = pickle.load(open("%s.pkl" % (filename), "rb"))
            self.scat_operator = sc.funct(KERNELSZ=outlist[3], all_type=outlist[7])
            self.KERNELSZ = self.scat_operator.KERNELSZ
            self.all_type = self.scat_operator.all_type
            self.npar = outlist[2]
            self.nscale = outlist[5]
            self.chanlist = outlist[0]
            self.in_nside = outlist[4]
            self.nbatch = outlist[1]
            self.NORIENT = outlist[8]
            self.x = self.scat_operator.backend.bk_cast(outlist[6])
            self.out_nside = self.in_nside // (2**self.nscale)
        else:
            self.nscale = len(chanlist)-1
            self.npar = nparam
            self.n_chan_in = n_chan_in
            self.scat_operator = scat_operator
            if self.scat_operator is None:
                self.scat_operator = sc.funct(
                    KERNELSZ=KERNELSZ,
                    NORIENT=NORIENT)

            self.chanlist = chanlist
            self.KERNELSZ = self.scat_operator.KERNELSZ
            self.NORIENT = self.scat_operator.NORIENT
            self.all_type = self.scat_operator.all_type
            self.in_nside = in_nside
            self.out_nside = self.in_nside * (2**self.nscale)
            self.backend = self.scat_operator.backend
            np.random.seed(SEED)
            self.x = self.scat_operator.backend.bk_cast(
                np.random.rand(self.get_number_of_weights())
                / (self.KERNELSZ * (self.KERNELSZ//2+1)*self.NORIENT)
            )

    def save(self, filename):

        outlist = [
            self.chanlist,
            self.nbatch,
            self.npar,
            self.KERNELSZ,
            self.in_nside,
            self.nscale,
            self.get_weights().numpy(),
            self.all_type,
            self.NORIENT,
        ]

        myout = open("%s.pkl" % (filename), "wb")
        pickle.dump(outlist, myout)
        myout.close()

    def get_number_of_weights(self):
        totnchan = 0
        for i in range(self.nscale):
            totnchan = totnchan + self.chanlist[i] * self.chanlist[i + 1]
        return (
            self.npar * 12 * self.in_nside**2 * self.chanlist[0]
            + totnchan * self.KERNELSZ * (self.KERNELSZ//2+1)
            + self.KERNELSZ * (self.KERNELSZ//2+1) * self.chanlist[nscale]
        )

    def set_weights(self, x):
        self.x = x

    def get_weights(self):
        return self.x

    def eval(self, im, indices=None, weights=None):

        x = self.x
        
        ww = self.backend.bk_reshape(
            x[0:self.npar * 12 * self.in_nside**2 * self.chanlist[0]],
            [self.npar,12 * self.in_nside**2 * self.chanlist[0]],
        )

        im = self.scat_operator.backend.bk_matmul(im,ww)
        
        im = self.backend.bk_reshape(im,[im.shape[0],self.chanlist[0],12 * self.in_nside**2])
        
        nn = self.npar * 12 * self.in_nside**2 * self.chanlist[0]
        
        for k in range(self.nscale):
            ww = self.scat_operator.backend.bk_reshape(
                x[
                    nn : nn
                    + self.KERNELSZ
                    * (self.KERNELSZ//2+1)
                    * self.chanlist[k]
                    * self.chanlist[k + 1]
                ],
                [self.chanlist[k], self.KERNELSZ * (self.KERNELSZ//2+1),  self.chanlist[k + 1]],
            )
            nn = (
                nn
                + self.KERNELSZ
                * (self.KERNELSZ//2)
                * self.chanlist[k]
                * self.chanlist[k + 1]
            )
            if indices is None:
                im = self.scat_operator.healpix_layer(im, ww)
            else:
                im = self.scat_operator.healpix_layer(
                    im, ww, indices=indices[k], weights=weights[k]
                )
            im = self.scat_operator.backend.bk_relu(im)
            
            im = self.backend.bk_reshape(self.scat_operator.backend.bk_repeat(im,4),[im.shape[0],im.shape[1],im.shape[2]*4])

        #im = self.scat_operator.backend.bk_reshape(im, [self.npar])
        #im = self.scat_operator.backend.bk_relu(im)

        return im
