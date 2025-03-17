import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import foscat.scat_cov as sc
import foscat.Synthesis as synthe

# rapid test
nside = 16

import os

os.system(
    "wget -O target_map_lss.npy https://github.com/astro-informatics/s2scat/raw/main/notebooks/data/target_map_lss.npy"
)

from scipy.interpolate import RegularGridInterpolator

# convert the input data in a nside=128 healpix map
l_nside = 128

im = np.load("target_map_lss.npy")
xsize, ysize = im.shape

# Define the new row and column to be added to prepare the interpolation
new_row = im[0:1, :]  # A new row with N elements (the other longitude)
new_column = np.concatenate(
    [im[:, 0:1], im[-2:-1, 0:1]], 0
)  # A new column with N+1 elements to add previous latitude

# Add the new row to the array
im = np.vstack([im, new_row])

# Add the new column to the array with the new row

im = np.hstack([im, new_column])

# Create a grid of coordinates corresponding to the array indices
x = np.linspace(0, im.shape[0] - 1, im.shape[0])
y = np.linspace(0, im.shape[1] - 1, im.shape[1])

# Create an interpolator
interpolator = RegularGridInterpolator((x, y), im)

# List of healpix coordinate to interpol
colatitude, longitude = hp.pix2ang(l_nside, np.arange(12 * l_nside**2), nest=True)
coords = (
    np.concatenate([colatitude / np.pi * xsize, longitude / (2 * np.pi) * ysize], 0)
    .reshape(2, colatitude.shape[0])
    .T
)

# Perform the interpolation
heal_im = interpolator(coords)

# reduce the final map to the expected resolution
if nside > 128:
    th, ph = hp.pix2ang(nside, np.arange(12 * nside**2), nest=True)
    heal_im = hp.get_interp_val(heal_im, th, ph, nest=True)
else:
    heal_im = np.mean(heal_im.reshape(12 * nside**2, (l_nside // nside) ** 2), 1)
hp.mollview(heal_im, cmap="plasma", nest=True, title="LSS", min=-3, max=3)

# free memory
del coords
del interpolator
del colatitude
del longitude

bk_tab = ["tensorflow", "torch"]

for BACKEND in bk_tab:
    print("==============================================================")
    print("\n\n TEST ", BACKEND, "\n\n")
    print("==============================================================")

    scat_op = sc.funct(
        NORIENT=4,  # define the number of wavelet orientation
        KERNELSZ=5,  # KERNELSZ,  # define the kernel size
        BACKEND=BACKEND,
        all_type="float64",
    )

    # set the reference statistics
    ref = scat_op.eval(heal_im, norm="auto")
    # get the statistic and the variance
    ref, sref = scat_op.eval(heal_im, calc_var=True, norm="self")

    ref.plot(name="LSS", color="b")
    sref.plot(name="Variance", color="r", hold=False)
    plt.savefig("test1.pdf")

    def The_loss(u, scat_operator, args):
        ref = args[0]
        sref = args[1]

        # compute scattering covariance of the current synthetised map called u
        learn = scat_operator.eval(u, norm="self")

        # make the difference withe the reference coordinates
        loss = scat_operator.reduce_distance(learn, ref, sigma=sref)

        return loss

    loss1 = synthe.Loss(The_loss, scat_op, ref, sref)

    sy = synthe.Synthesis([loss1])

    np.random.seed(1234)

    imap = np.random.randn(12 * nside**2) * np.std(heal_im)
    inscat = scat_op.eval(imap, norm="self")

    # do the synthesis
    omap = scat_op.to_numpy(sy.run(imap, EVAL_FREQUENCY=10, NUM_EPOCHS=100))

    assert np.min(sy.get_history()) < 4

    def The_loss_mean(u, scat_operator, args):
        ref = args[0]
        sref = args[1]

        # compute the mean scattering covariance of the current synthetised maps called u
        learn = scat_operator.reduce_mean_batch(scat_operator.eval(u, norm="self"))

        # make the difference with the reference coordinates
        loss = scat_operator.reduce_mean(scat_operator.square((learn - ref) / sref))

        return loss

    loss_mean = synthe.Loss(The_loss_mean, scat_op, ref, sref)

    sy_mean = synthe.Synthesis([loss_mean])

    np.random.seed(1234)

    imap = np.random.randn(4, 12 * nside**2) * np.std(heal_im)

    omap = scat_op.to_numpy(sy_mean.run(imap, EVAL_FREQUENCY=10, NUM_EPOCHS=100))

    assert np.min(sy_mean.get_history()) < 1.0
