import numpy as np
import os, sys
import matplotlib.pyplot as plt
import healpy as hp
import foscat.FoCUS as FOC

# =================================================================================
# DEFINE A PATH FOR scratch data
# The data are storred using a default nside to minimize the needed storage
# =================================================================================
scratch_path = "../data"
outname = "TEST_EE"
nout = 64


# =================================================================================
# Function to reduce the data used in the FoCUS algorithm
# =================================================================================
def dodown(a, nout):
    nin = int(np.sqrt(a.shape[0] // 12))
    if nin == nout:
        return a
    return np.mean(a.reshape(12 * nout * nout, (nin // nout) ** 2), 1)


# =================================================================================
# Get data
# =================================================================================
im = dodown(np.load("Venus_256.npy"), nout)


# =================================================================================
# INITIALIZE FoCUS class
# =================================================================================

fc = FOC.FoCUS(
    NORIENT=4,  # define the number of wavelet orientation
    KERNELSZ=5,  # define the kernel size (here 3x3)
    healpix=True,  # use the healpix pixelisation
    OSTEP=0,  # get very large scale (nside=1)
    nside=nout,  # get very large scale (nside=1)
    LAMBDA=1.0,
    TEMPLATE_PATH=scratch_path,
)


# =================================================================================
# COMPUTE THE WAVELET TRANSFORM OF THE REFERENCE MAP
# =================================================================================

s1, p0, c01, c11 = fc.get_scat_cov_coeffs(im)
p0_C, c01_C, c11_C = fc.get_scat_cov_coeffs(im, image2=im)

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(s1.numpy().flatten())
plt.subplot(2, 2, 2)
plt.plot(p0.numpy().flatten())
plt.subplot(2, 2, 3)
plt.plot(c01.numpy().flatten())
plt.subplot(2, 2, 4)
plt.plot(c11.numpy().flatten())

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(s1.numpy().flatten())
plt.subplot(2, 2, 2)
plt.plot(p0_C.numpy().flatten())
plt.subplot(2, 2, 3)
plt.plot(c01_C.numpy().flatten())
plt.subplot(2, 2, 4)
plt.plot(c11_C.numpy().flatten())

plt.show()
"""
np.save('in_cov_s1.npy', s1)
np.save('in_cov_p0.npy', p0)
np.save('in_cov_c01.npy',c01)
np.save('in_cov_c11.npy',c11)


#=================================================================================
# Initialize the map to be synthesised
#=================================================================================

x=fc.init_variable(np.random.randn(12*nout*nout))

r1,r0,r01,r11=fc.wst_cov(x,axis=0)

np.save('st_cov_s1.npy', r1)
np.save('st_cov_p0.npy', r0)
np.save('st_cov_c01.npy', r01)
np.save('st_cov_c11.npy', r11)

#=================================================================================
# ADD THE LOSS CONSTRAIN
#=================================================================================
fc.add_loss_wst_cov(s1,p0,c01,c11,axis=0)

#=================================================================================
# DO THE SYNTHESIS
#=================================================================================
omap=fc.learnv2(DECAY_RATE=0.9996,
                NUM_EPOCHS = 10000,
                LEARNING_RATE = 0.3)

#=================================================================================
# STORE IT
#=================================================================================
os1,op0,oc01,oc11=fc.wst_cov(omap)

#================================================================================
# store results
#================================================================================

np.save('out_cov_s1.npy', os1)
np.save('out_cov_p0.npy', op0)
np.save('out_cov_c01.npy', oc01)
np.save('out_cov_c11.npy', oc11)

np.save('in_cov_map.npy',im)
np.save('out_cov_map.npy',omap)
np.save('out_cov_log.npy',fc.get_log())

print('Computation Done')
sys.stdout.flush()
"""
