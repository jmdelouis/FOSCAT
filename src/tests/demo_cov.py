import numpy as np
import os, sys
import matplotlib.pyplot as plt
import healpy as hp

#=================================================================================
# DEFINE A PATH FOR scratch data
# The data are storred using a default nside to minimize the needed storage
#=================================================================================
scratch_path = '../data'
outname='TEST_EE'
nout=64
#=================================================================================
# Function to reduce the data used in the FoCUS algorithm 
#=================================================================================
def dodown(a,nout):
    nin=int(np.sqrt(a.shape[0]//12))
    if nin==nout:
        return(a)
    return(np.mean(a.reshape(12*nout*nout,(nin//nout)**2),1))
#=================================================================================
# Get data
#=================================================================================
im=dodown(np.load('Venus_256.npy'),nout)

#=================================================================================
# INITIALIZE FoCUS class
#=================================================================================
import foscat.FoCUS as FOC

fc=FOC.FoCUS(NORIENT=4,   # define the number of wavelet orientation
             KERNELSZ=3,  # define the kernel size (here 3x3)
             healpix=True, # use the healpix pixelisation
             OSTEP=0,     # get very large scale (nside=1)
             TEMPLATE_PATH=scratch_path)

#=================================================================================
# This section makes all the initialization, I should think about including most
# of it inside the previous call. 
#================================================================================
#=================================================================================
# Set masking to ONE all surface is used and expected to be stationary
#================================================================================

mask=np.ones([1,12*nout**2])
fc.add_mask(mask)

#=============================================
# compute amplitude to normalize the dynamic range 
ampmap=1/im[mask[0]>0.9].std()

# convert data in tensor for focus (should be done internally)
data = fc.convimage(ampmap*(im))

# Initialize the learning and initialize the tensor to be synthesized
ldata=fc.init_synthese(np.random.randn(12*nout*nout))

#=================================================================================
# DEFINE THE LOSS, SHOULD BE DONE INSIDE INITIALISATION
#================================================================================

# Add covariance losss:
fc.add_loss_cov(data,ldata)

# initiliaze the loss
loss=fc.init_optim()

#================================================================================
# End of the initialization
#================================================================================

#================================================================================
# Compute the S1 and S2 coefficients
#================================================================================

s1,s2,s3,s4=fc.calc_stat_cov(im.reshape(1,12*nout*nout)*ampmap)

#save the input map
r1,r2,r3,r4=fc.calc_stat_cov(fc.get_map().reshape(1,12*nout*nout))

np.save('st_cov_s1.npy', r1)
np.save('st_cov_s2.npy', r2)
np.save('st_cov_s3.npy', r3)
np.save('st_cov_s4.npy', r4)

#define BIAS and WEIGTH for each scaterring coefficients
tw1={}
tw2={}
tw3={}
tw4={}
tb1={}
tb2={}
tb3={}
tb4={}
for i in range(1):
    tw1[i]=np.ones_like(s1[0])
    tw2[i]=np.ones_like(s2[0])
    tw3[i]=np.ones_like(s3[0])
    tw4[i]=np.ones_like(s4[0])
    tb1[i]=np.zeros_like(s1[0])
    tb2[i]=np.zeros_like(s2[0])
    tb3[i]=np.zeros_like(s3[0])
    tb4[i]=np.zeros_like(s4[0])

#================================================================================
# Run the synthesis
#================================================================================
omap=fc.learn(tw1,tw2,tb1,tb2,
              NUM_EPOCHS = 5000,
              EVAL_FREQUENCY = 100,
              DECAY_RATE=0.9999,
              LEARNING_RATE=1.0,
              SEQUENTIAL_ITT=10,
              ADDAPT_LEARN=200.0,
              IW3=tw3,IW4=tw4,
              IB3=tb3,IB4=tb4)

#================================================================================
# store results
#================================================================================
modd1=omap.reshape(1,12*nout**2)
os1,os2,os3,os4=fc.calc_stat_cov(modd1)

np.save('in_cov_mask.npy', mask)
np.save('in_cov_s1.npy', s1)
np.save('in_cov_s2.npy', s2)
np.save('in_cov_s3.npy', s3)
np.save('in_cov_s4.npy', s4)
np.save('out_cov_s1.npy', os1)
np.save('out_cov_s2.npy', os2)
np.save('out_cov_s3.npy', os3)
np.save('out_cov_s4.npy', os4)
np.save('in_cov_map.npy',im)
np.save('out_cov_map.npy',omap/ampmap)

print('Computation Done')
sys.stdout.flush()

