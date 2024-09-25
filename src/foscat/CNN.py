import pickle

import numpy as np

import foscat.scat_cov as sc


class CNN:

    def __init__(
        self,
        scat_operator=None,
        nparam=1,
        nscale=1,
        chanlist=[],
        in_nside=1,
        n_chan_in=1,
        nbatch=1,
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
            self.n_chan_in = outlist[8]
            self.x = self.scat_operator.backend.bk_cast(outlist[6])
            self.out_nside = self.in_nside // (2**self.nscale)
        else:
            self.nscale = nscale
            self.nbatch = nbatch
            self.npar = nparam
            self.n_chan_in = n_chan_in
            self.scat_operator = scat_operator
            if len(chanlist) != nscale + 1:
                print(
                    "len of chanlist (here %d) should of nscale+1 (here %d)"
                    % (len(chanlist), nscale + 1)
                )
                return None

            self.chanlist = chanlist
            self.KERNELSZ = scat_operator.KERNELSZ
            self.all_type = scat_operator.all_type
            self.in_nside = in_nside
            self.out_nside = self.in_nside // (2**self.nscale)

            np.random.seed(SEED)
            self.x = scat_operator.backend.bk_cast(
                np.random.randn(self.get_number_of_weights())
                / (self.KERNELSZ * self.KERNELSZ)
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
            self.n_chan_in,
        ]

        myout = open("%s.pkl" % (filename), "wb")
        pickle.dump(outlist, myout)
        myout.close()

    def get_number_of_weights(self):
        totnchan = 0
        for i in range(self.nscale):
            totnchan = totnchan + self.chanlist[i] * self.chanlist[i + 1]
        return (
            self.npar * 12 * self.out_nside**2 * self.chanlist[self.nscale]
            + totnchan * self.KERNELSZ * self.KERNELSZ
            + self.KERNELSZ * self.KERNELSZ * self.n_chan_in * self.chanlist[0]
        )

    def set_weights(self, x):
        self.x = x

    def get_weights(self):
        return self.x

    def eval(self, im, indices=None, weights=None):

        x = self.x
        ww = self.scat_operator.backend.bk_reshape(
            x[0 : self.KERNELSZ * self.KERNELSZ * self.n_chan_in * self.chanlist[0]],
            [self.KERNELSZ * self.KERNELSZ, self.n_chan_in, self.chanlist[0]],
        )
        nn = self.KERNELSZ * self.KERNELSZ * self.n_chan_in * self.chanlist[0]

        im = self.scat_operator.healpix_layer(im, ww)
        im = self.scat_operator.backend.bk_relu(im)

        for k in range(self.nscale):
            ww = self.scat_operator.backend.bk_reshape(
                x[
                    nn : nn
                    + self.KERNELSZ
                    * self.KERNELSZ
                    * self.chanlist[k]
                    * self.chanlist[k + 1]
                ],
                [self.KERNELSZ * self.KERNELSZ, self.chanlist[k], self.chanlist[k + 1]],
            )
            nn = (
                nn
                + self.KERNELSZ
                * self.KERNELSZ
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
            im = self.scat_operator.ud_grade_2(im, axis=0)

        ww = self.scat_operator.backend.bk_reshape(
            x[
                nn : nn
                + self.npar * 12 * self.out_nside**2 * self.chanlist[self.nscale]
            ],
            [12 * self.out_nside**2 * self.chanlist[self.nscale], self.npar],
        )

        im = self.scat_operator.backend.bk_matmul(
            self.scat_operator.backend.bk_reshape(
                im, [1, 12 * self.out_nside**2 * self.chanlist[self.nscale]]
            ),
            ww,
        )
        im = self.scat_operator.backend.bk_reshape(im, [self.npar])
        im = self.scat_operator.backend.bk_relu(im)

        return im
