import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import foscat.Synthesis as synthe

xsize = 32
# number of noise to learn statistics
nnoise = 10

# amplitude of the white noise
ampnoise = 0.1

from PIL import Image

SMILEY_IMAGE = "./data/smiley-face-svg-rogne.jpg"
img = Image.open(SMILEY_IMAGE)
smiley = np.zeros([512, 512])
smiley[256 - 128 : 256 + 128, 256 - 128 : 256 + 128] = 1.0 * (
    np.asarray(img)[:, :, 0] < 128
)

# de zoom at the xsize scale
smiley = (
    np.sum(np.sum(smiley.reshape(xsize, 512 // xsize, xsize, 512 // xsize), 3), 1)
    / (512 / xsize) ** 2
)

np.random.seed(1234)

noise = ampnoise * np.random.randn(nnoise, xsize, xsize)

bk_tab = ["tensorflow", "torch"]

# ============================================================================================
#
#    N_image=1
#
# ============================================================================================
import foscat.scat_cov2D as sc

for BACKEND in bk_tab:

    print("==============================================================")
    print("\n\n TEST ", BACKEND, "\n\n")
    print("==============================================================")

    scat_op = sc.funct(BACKEND=BACKEND)

    def eval_scat(sc, x, image2=None, mask=None):
        return sc.eval(x, image2=image2, mask=mask).flatten()

    # compute statistics for the smiley and the smiley+noise
    data = smiley + np.random.randn(smiley.shape[0], smiley.shape[1]) * ampnoise

    edge = False

    # \Phi(d) \simeq \Phi(u + n),
    def The_loss(x, scat_operator, args, return_all=False):

        ref = args[0]
        noise = args[1]

        learn = scat_operator.scattering_cov(x + noise, edge=edge)

        learn = scat_operator.backend.bk_reduce_mean(learn, 0)

        loss = scat_operator.reduce_mean(scat_operator.square(ref - learn))
        return loss

    # \Phi(d,u) \simeq \Phi(u + n,u),
    def The_lossX(x, scat_operator, args, return_all=False):

        im = args[0]
        noise = args[1]

        ref = scat_operator.scattering_cov(im, data2=x, edge=edge)[0]

        learn = scat_operator.scattering_cov(x + noise, data2=x, edge=edge)
        learn = scat_operator.backend.bk_reduce_mean(learn - ref, 0)

        loss = scat_operator.reduce_mean(scat_operator.square(learn))
        return loss

    ref = scat_op.scattering_cov(data, edge=edge)[0]

    in_data = scat_op.backend.bk_cast(data[None, :, :])

    loss = synthe.Loss(The_loss, scat_op, ref, scat_op.backend.bk_cast(noise))

    lossX = synthe.Loss(
        The_lossX,
        scat_op,
        scat_op.backend.bk_cast(data[None, :, :]),
        scat_op.backend.bk_cast(noise),
    )

    # define the foscat synthesis using the two previous loss
    sy = synthe.Synthesis([loss, lossX])

    clean_map = sy.run(in_data, EVAL_FREQUENCY=10, NUM_EPOCHS=100)

    clean_map = scat_op.to_numpy(clean_map)

    print((clean_map - smiley).std())
    assert (clean_map - smiley).std() < 0.07
