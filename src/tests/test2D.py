import numpy as np
import os, sys
import matplotlib.pyplot as plt
import foscat as FOC

#=================================================================================
# INITIALIZE foscat class
#=================================================================================
#  Change the temporary path
#=================================================================================
scratch_path = 'foscat_data'
outname='test2D'

#=================================================================================
# This test could be run till XSIZE=512 but could last for Hours if not GPUs are
# available. XSIZE=128 takes few minutes with NVIDIA T4.
#=================================================================================
XSIZE=128

#=================================================================================
#==   Make the statistic invariant by rotation => Higher dimension reduction    ==
#==   avg_ang = True                                                            ==
#=================================================================================
avg_ang=False

fc=FOC.foscat(NORIENT=8,KERNELSZ=5,OSTEP=0)

#=================================================================================
#  READ data and get data if necessary: 
#=================================================================================
#  Here the input data is a vorticity map of 512x512:
# This image shows a simulated snapshot of ocean turbulence in the North Atlantic Ocean in March 2012,
# from a groundbreaking super-high-resolution global ocean simulation (approximately 1.2 miles, 
#or 2 kilometers, horizontal resolution) developed at JPL.
# (http://wwwcvs.mitgcm.org/viewvc/MITgcm/MITgcm_contrib/llc_hires/llc_4320/). 
#=================================================================================
try:
    d=np.load(scratch_path+'/Vorticity.npy')
except:
    import imageio as iio

    os.system('wget -O '+scratch_path+'/PIA22256.tif https://photojournal.jpl.nasa.gov/tiff/PIA22256.tif')
    
    im=iio.imread(scratch_path+'/PIA22256.tif')
    im=im[1000:1512,2000:2512,0]/255.0-im[1000:1512,2000:2512,2]/255.0
    np.save(scratch_path+'/Vorticity.npy',im)
    os.system('rm '+scratch_path+'/PIA22256.tif')
    d=np.load(scratch_path+'/Vorticity.npy')


d=d[0:XSIZE,0:XSIZE]
nx=d.shape[0]
# define the level of noise of the simulation
ampnoise=1.0

#=================================================================================
# Synthesise data with the same cross statistic than the input data
#=================================================================================

# convert data in tensor for Foscat
idata = fc.convimage(d)

# define the mask where the statistic are used
x=np.repeat((np.arange(XSIZE)-XSIZE/2)/XSIZE,XSIZE).reshape(XSIZE,XSIZE)
mask=np.exp(-32*(x**4+(x.T)**4))
fc.add_mask(mask.reshape(1,nx,nx))

# Initialize the learning and initialize the tensor to be synthesized
randfield=np.random.randn(nx,nx)
ldata=fc.init_synthese(randfield)

# Build the loss:
# here Loss += (d x d - s x s - tb[0]).tw[0]
fc.add_loss_2d(idata,idata,ldata,ldata,avg_ang=avg_ang)

# initiliaze the synthesise process
loss=fc.init_optim()

tw1={}
tw2={}
tb1={}
tb2={}
# compute the weights and the bias for each loss
modd=d.reshape(1,nx,nx)
r1,r2=fc.calc_stat(modd,modd,avg_ang=avg_ang)

tw1[0]=1.0/r1[0]
tw2[0]=1.0/r2[0]
tb1[0]=0.0*r1[0]
tb2[0]=0.0*r2[0]

#save the reference statistics
np.save(scratch_path+'/%s_r1.npy'%(outname), r1)
np.save(scratch_path+'/%s_r2.npy'%(outname), r2)

# Run the learning
fc.learn(tw1,tw2,tb1,tb2,NUM_EPOCHS = 5000,DECAY_RATE=1.0)

# get the output map
omap=fc.get_map()

modd=omap.reshape(1,nx,nx)
o1,o2=fc.calc_stat(modd,modd,avg_ang=avg_ang)
#save the statistics on the synthesised data
np.save(scratch_path+'/%s_o1.npy'%(outname), o1)
np.save(scratch_path+'/%s_o2.npy'%(outname), o2)

np.save(scratch_path+'/%s_ref.npy'%(outname), d)
np.save(scratch_path+'/%s_start.npy'%(outname), randfield)
np.save(scratch_path+'/%s_result.npy'%(outname),omap)
