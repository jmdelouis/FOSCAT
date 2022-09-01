import numpy as np
import os, sys
import matplotlib.pyplot as plt


#export LD_LIBRARY_PATH=/export/home/jmdeloui/cuda11:/usr/local/cuda-10.0/targets/x86_64-linux/lib/


#=================================================================================
# INITIALIZE FoCUS class
#=================================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR)+'/FoCUS')
sys.path.append(os.path.dirname(CURRENT_DIR)+'demo/pattern.py')
import FoCUS as FOC
import pattern as pt
fc=FOC.FoCUS(NORIENT=4,KERNELSZ=3,gpupos=0)
#fc=FOC.FoCUS(NORIENT=12,KERNELSZ=7,gpupos=0)

#c,s=fc.get_ww()
#plt.figure(figsize=(16,6))
#npt=int(np.sqrt(c.shape[0]))
#for i in range(c.shape[1]):
#    plt.subplot(2,c.shape[1],1+i)
#    plt.imshow(c[:,i].reshape(npt,npt),cmap='jet',vmin=-0.5,vmax=1.0)
#    plt.subplot(2,c.shape[1],1+i+c.shape[1])
#    plt.imshow(s[:,i].reshape(npt,npt),cmap='jet',vmin=-0.5,vmax=1.0)
#
#plt.show()

#=================================================================================
# READ data: Here the input data is a small picture of 256x256
#=================================================================================
xsize = 48
ysize = 256

d1,d2 = pt.create_imagette(xsize,ysize)

# normalise d1 and d2
"""
a = np.min(d1)
b = np.max(d1)
d1 = (d1-a)/(b-a)
d2 = (d2-a)/(b-a)
"""
d = (d1+d2)/2


idx = np.where(np.isnan(d))[0]
while len(idx)>0:
    #d[idx]=d[(idx+1)%(256*256)]
    d[idx]=d[(idx+1)%(xsize*ysize)]
    idx = np.where(np.isnan(d))[0]

#d=d.reshape(256,256)*1E3
d=d.reshape(xsize,ysize)*1E3

nx=d.shape[0]
ny=d.shape[1]


# define the level of noise of the simulation
ampnoise=1.0
Alpha=0.0
nsim=100
avg_ang=True

d1 = d+ampnoise*np.random.randn(nx,ny)
d2 = d+ampnoise*np.random.randn(nx,ny)

# simulate the noisy data
#filter = abs(np.fft.fft2(d))
#spectre = np.fft.fft2(np.random.rand(nx,nx))
#di = np.fft.ifft2(filter/abs(spectre)*spectre).real
#di = d+ampnoise*np.random.randn(nx,nx)/np.sqrt(2)
di = d+ampnoise*np.random.randn(nx,ny)/np.sqrt(2)


#=================================================================================
# For real data:
# you have to give d1,d2 and d
# you have to define the value ampnoise
#=================================================================================

# convert data in tensor for focus (should be done internally)
data1 = fc.convimage(d1)
data2 = fc.convimage(d2)
data  = fc.convimage(d)

# All information of the map is used
mask=np.exp(-32*(np.repeat(np.arange(nx)-nx/2,ny)**4+np.tile(np.arange(nx)-nx/2,ny)**4)/(nx*ny*nx*ny))
fc.add_mask(mask.reshape(1,nx,ny))


# Initialize the learning and initialize the tensor to be synthesized
ldata=fc.init_synthese(di)

# Add losss:
# here d1 x d2 = s x s
iloss=fc.add_loss_2d(data1,data2,ldata,ldata,avg_ang=avg_ang)
iloss=fc.add_loss_2d(data1,data2,data,ldata,avg_ang=avg_ang)


# initiliaze the loss
loss=fc.init_optim()

c1,c2=fc.calc_stat(d1.reshape(1,nx,ny),d2.reshape(1,nx,ny),avg_ang=avg_ang)
lmap1=1*d1
lmap2=1*d2



for itt in range(4):
    # compute scattering to make weights
    modd1=d1.reshape(1,nx,ny)+np.random.randn(nsim,nx,ny)
    modd2=d2.reshape(1,nx,ny)+np.random.randn(nsim,nx,ny)
    o1,o2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang)


    modd1=d1.reshape(1,nx,ny)+np.random.randn(nsim,nx,ny)
    modd2=d2.reshape(1,nx,ny) + 0*np.random.randn(nsim,nx,ny)
    ob1,ob2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang)

    modd1r=d1.reshape(1,nx,ny)
    modd2r =d2.reshape(1,nx,ny)
    r1,r2=fc.calc_stat(modd1r,modd2r,avg_ang=avg_ang)

    moddr=d.reshape(1,nx,ny)
    x1,x2=fc.calc_stat(moddr,moddr,avg_ang=avg_ang)


    #calcul poids des coeffs
    np.save('../data/iB1_%d.npy'%(itt), x1)
    np.save('../data/iB2_%d.npy'%(itt), x2)
    np.save('../data/bB1_%d.npy'%(itt), r1)
    np.save('../data/bB2_%d.npy'%(itt), r2)
    np.save('../data/nB1_%d.npy'%(itt), o1)
    np.save('../data/nB2_%d.npy'%(itt), o2)

    tw1={}
    tw2={}
    tw1[0]=1/np.std(o1,0)
    tw2[0]=1/np.std(o2,0)

    tw1[1] = 1/np.std(o1-ob1,0)
    tw2[1]=1/np.std(o2-ob2,0)


    tb1={}
    tb2={}
    tb1[0]=Alpha*np.mean(o1-r1,0)
    tb2[0]=Alpha*np.mean(o2-r2,0)


    tb1[1]=Alpha*np.mean(o1-r1,0) - Alpha*np.mean(ob1-r1,0)
    tb2[1]=Alpha*np.mean(o2-r2,0)- Alpha*np.mean(ob2-r2,0)
    
        
    # make the learn
    fc.reset()
    omap=fc.learn(tw1,tw2,tb1,tb2,NUM_EPOCHS = 3000,DECAY_RATE=0.999,LEARNING_RATE=0.03)
    print('ITT ',itt,((d-omap)*mask.reshape(nx,ny)).std(),((d-di)*mask.reshape(nx,ny)).std())

    modd1=omap.reshape(1,nx,ny)
    oo1,oo2=fc.calc_stat(modd1,modd1,avg_ang=avg_ang)
    
    history = fc.get_history()

    plt.plot(history)
    plt.yscale('log')
    

    doplot=True
    lmap1=1*omap
    lmap2=1*omap
    if doplot==True:
        plt.figure(figsize=(16,8))
        plt.subplot(2,2,1)
        plt.plot(x1.flatten(),color='grey',lw=4)
        plt.plot(c1.flatten(),color='red',lw=4)
        plt.plot((oo1).flatten(),color='black')
        plt.plot(r1.flatten(),color='pink')
        plt.plot(np.mean(o1,0).flatten(),color='orange')
        plt.plot(np.mean(o1,0).flatten()+np.std(o1,0).flatten(),':',color='orange')
        plt.plot(np.mean(o1,0).flatten()-np.std(o1,0).flatten(),':',color='orange')
        plt.plot(c1.flatten()-tb1[0].flatten(),color='blue')
        plt.yscale('log')
        plt.subplot(2,2,2)
        plt.plot(x2.flatten(),color='grey',lw=4)
        plt.plot(c2.flatten(),color='red',lw=4)
        plt.plot((oo2).flatten(),color='black')
        plt.plot(r2.flatten(),color='pink')
        plt.plot(np.mean(o2,0).flatten(),color='orange')
        plt.plot(np.mean(o2,0).flatten()+np.std(o2,0).flatten(),':',color='orange')
        plt.plot(np.mean(o2,0).flatten()-np.std(o2,0).flatten(),':',color='orange')
        plt.plot(c2.flatten()-tb2[0].flatten(),color='blue')
        plt.yscale('log')
        plt.subplot(2,4,5)
        plt.imshow(di,cmap='Greys',vmin=0.0,vmax=1)
        plt.subplot(2,4,6)
        plt.imshow(d,cmap='Greys',vmin=0,vmax=1)
        plt.subplot(2,4,7)
        plt.imshow(lmap1,cmap='Greys',vmin=0,vmax=1)
        plt.subplot(2,4,8)
        plt.imshow(d-lmap1,cmap='Greys',vmin=-0.5,vmax=0.5)
        plt.show()
        
    np.save('../data/oB1_%d.npy'%(itt), oo1)
    np.save('../data/oB2_%d.npy'%(itt), oo2)

    np.save('../data/testBref_%d.npy'%(itt), d)
    np.save('../data/testBinput_%d.npy'%(itt),di)
    np.save('../data/testBresult_%d.npy'%(itt),omap)

