import numpy as np
import os, sys
import matplotlib.pyplot as plt

#=================================================================================
# INITIALIZE FoCUS class
#=================================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR)+'/FoCUS')
import FoCUS as FOC

fc=FOC.FoCUS(NORIENT=4,KERNELSZ=3)

#=================================================================================
# READ data: Here the input data is a small picture of 256x256
#=================================================================================
d  = np.load('../data/857-1_FSL_MAP.npy')
idx = np.where(np.isnan(d))[0]
while len(idx)>0:
    d[idx]=d[(idx+1)%(256*256)]
    idx = np.where(np.isnan(d))[0]
d=d.reshape(256,256)*1E3
nx=d.shape[0]
# define the level of noise of the simulation
ampnoise=1.0

# Build to half mission by adding noise to the data
d1 = d+ampnoise*np.random.randn(nx,nx)
d2 = d+ampnoise*np.random.randn(nx,nx)

# simulate the noisy data
di = d+ampnoise*np.random.randn(nx,nx)

#=================================================================================
# For real data:
# you have to give d1,d2 and d
# you have to define the value ampnoise
#=================================================================================

# convert data in tensor for focus (should be done internally)
data1 = fc.convimage(d)
data2 = fc.convimage(d2)

# All information of the map is used
fc.add_mask((1+0*d).reshape(1,nx,nx))

# Initialize the learning and initialize the tensor to be synthesized
ldata=fc.init_synthese(np.random.randn(nx,nx))

# Add losss:
# here d1 x d2 = s x s
iloss=fc.add_loss_2d(data1,data2,ldata,ldata)

# initiliaze the loss
loss=fc.init_optim()

# compute the weights and the bias using simulations
nsim=10
modd1=d.reshape(1,nx,nx)+ampnoise*np.random.randn(nsim,nx,nx)
modd2=d.reshape(1,nx,nx)+ampnoise*np.random.randn(nsim,nx,nx)
r1,r2=fc.calc_stat(modd1,modd2,iloss)
tw1={}
tw2={}
tw1[0]=1.0/(np.std(r1,0))
tw2[0]=1.0/(np.std(r2,0))

r1,r2=fc.calc_stat(modd1,modd2,iloss,imaginary=True)
modd1=d.reshape(1,nx,nx)
o1,o2=fc.calc_stat(modd1,modd1,iloss,imaginary=True)

np.save('../data/r1.npy', r1)
np.save('../data/r2.npy', r2)

tb1={}
tb2={}
tb1[0]=np.mean(r1-o1,0)
tb2[0]=np.mean(r2-o2,0)

# make the learn
fc.learn(tw1,tw2,tb1,tb2,NUM_EPOCHS = 10000,DECAY_RATE=0.99)

# get the output map
omap=fc.get_map()

np.save('../data/test2Dref.npy', d)
np.save('../data/test2Dinput1.npy',d1)
np.save('../data/test2Dinput2.npy',d2)
np.save('../data/test2Dinput.npy',di)
np.save('../data/test2Dresult.npy',omap)
