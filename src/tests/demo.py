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
nout=32
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
im=dodown(np.load('dust_64.npy'),nout)


#=================================================================================
# INITIALIZE FoCUS class
#=================================================================================
import foscat.FoCUS as FOC

fc=FOC.FoCUS(NORIENT=4,   # define the number of wavelet orientation
             KERNELSZ=3,  # define the kernel size (here 3x3)
             healpix=True, # use the healpix pixelisation
             OSTEP=-1,     # get very large scale (nside=1)
             TEMPLATE_PATH=scratch_path)

#=================================================================================
# This section makes all the initialization, I should think about including most
# of it inside the previous call. 
#================================================================================
#=================================================================================
# Set masking to define the sky region of interest
#================================================================================
tab=['MASK_GAL080_64.npy','MASK_GAL060_64.npy']
mask=np.ones([len(tab),12*nout**2])
for i in range(len(tab)):
    mask[i,:]=dodown(np.load(tab[i]),nout)
#force the first mask to be the all sky
mask[0,:]=1.0
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

#=================================================================================
# COMPUTE THE WAVELET TRANSFORM
#=================================================================================
res=fc.do_conv(im)

plt.figure(figsize=(12,8))
for i in range(4):
    hp.mollview(res[:,i].real,cmap='jet',hold=False,sub=(3,4,1+i),nest=True,title='Real Dir=%d'%(i),norm='hist')
    hp.mollview(res[:,i].imag,cmap='jet',hold=False,sub=(3,4,5+i),nest=True,title='Imag Dir=%d'%(i),norm='hist')
    hp.mollview(abs(res[:,i]),cmap='jet',hold=False,sub=(3,4,9+i),nest=True,title='Norm Dir=%d'%(i),norm='hist')

#=================================================================================
# GET WAVELET COEFFICIENTS
#=================================================================================
c,s=fc.get_ww()
kernel=int(np.sqrt(c.shape[0]))
print('Real Part of the wavelet coefficients')
print(c[:,0].reshape(kernel,kernel))
print('Imaginary Part of the wavelet coefficients')
print(s[:,0].reshape(kernel,kernel))
fc.plot_ww()

# Add losss:
fc.add_loss_healpix(data,data,ldata,ldata,imaginary=False)

# initiliaze the loss
loss=fc.init_optim()

#================================================================================
# End of the initialization
#================================================================================

#================================================================================
# Compute the S1 and S2 coefficients
#================================================================================

s1,s2=fc.calc_stat(im.reshape(1,12*nout*nout)*ampmap,im.reshape(1,12*nout*nout)*ampmap,imaginary=False)

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(s1.flatten())
plt.subplot(1,2,2)
plt.plot(s2.flatten())
plt.show()
    
#define BIAS and WEIGTH for each scaterring coefficients
tw1={}
tw2={}
tb1={}
tb2={}
for i in range(1):
    tw1[i]=1/(s1[0])
    tw2[i]=1/(s2[0])
    tb1[i]=0.0*s1[0]
    tb2[i]=0.0*s2[0]
        
#================================================================================
# Run the synthesis
#================================================================================
omap=fc.learn(tw1,tw2,tb1,tb2,
              NUM_EPOCHS = 1000,
              EVAL_FREQUENCY = 100,
              DECAY_RATE=0.9999,
              LEARNING_RATE=1.0,
              SEQUENTIAL_ITT=10,
              ADDAPT_LEARN=200.0)

#================================================================================
# store results
#================================================================================
modd1=omap.reshape(1,12*nout**2)
os1,os2=fc.calc_stat(modd1,modd1,imaginary=False)

np.save('in_mask.npy', mask)
np.save('in_s1.npy', s1)
np.save('in_s2.npy', s2)
np.save('out_s1.npy', os1)
np.save('out_s2.npy', os2)
np.save('in_map.npy',im)
np.save('out_map.npy',omap/ampmap)

print('Computation Done')
sys.stdout.flush()

