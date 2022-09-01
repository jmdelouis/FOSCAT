import numpy as np
import os, sys
import matplotlib.pyplot as plt

#=================================================================================
# INITIALIZE FoCUS class
#=================================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR)+'/FoCUS')
import FoCUS as FOC

fc=FOC.FoCUS(NORIENT=8,KERNELSZ=5,gpupos=2)

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

import imageio as iio
im = iio.imread("../data/Bloom.png")

def compspec(im):
    r=np.sqrt(np.mean((im[:,:,0:3].astype('float'))**2,2))
    xz=r.shape[0]
    yz=r.shape[1]
    x=np.repeat(np.roll(np.arange(xz)-xz//2,xz//2),yz).reshape(xz,yz)
    y=np.tile(np.roll(np.arange(yz)-yz//2,yz//2),xz).reshape(xz,yz)
    ir=np.sqrt(x**2+y**2).astype('int')
    tf=np.fft.fft2(r)
    cl=np.bincount(ir.flatten(),weights=abs(tf.flatten()))
    clw=np.bincount(ir.flatten())
    return(r,cl/clw,np.arange(cl.shape[0])/cl.shape[0])

d,cl1,xx1=compspec(im)

#=================================================================================
# READ data: Here the input data is a small picture of 256x256
#=================================================================================
d=d[0:256,0:256]/255.0
d=(d-d.min())/(d.max()-d.min())
nx=d.shape[0]
# define the level of noise of the simulation
ampnoise=0.1
Alpha=1.0
nsim=1000
avg_ang=True

# Build to half mission by adding noise to the data
d1 = d+ampnoise*np.random.randn(nx,nx)
d2 = d+ampnoise*np.random.randn(nx,nx)

# simulate the noisy data
#filter = abs(np.fft.fft2(d))
#spectre = np.fft.fft2(np.random.rand(nx,nx))
#di = np.fft.ifft2(filter/abs(spectre)*spectre).real
di = d+ampnoise*np.random.randn(nx,nx)/np.sqrt(2)

#=================================================================================
# For real data:
# you have to give d1,d2 and d
# you have to define the value ampnoise
#=================================================================================

# convert data in tensor for focus (should be done internally)
data1 = fc.convimage(d1)
data2 = fc.convimage(d2)
data  = fc.convimage(di)

# All information of the map is used
mask=np.exp(-32*(np.repeat(np.arange(nx)-nx/2,nx)**4+np.tile(np.arange(nx)-nx/2,nx)**4)/(nx*nx*nx*nx))
fc.add_mask(mask.reshape(1,nx,nx))

# Initialize the learning and initialize the tensor to be synthesized
ldata=fc.init_synthese(di)

# Add losss:
# here d1 x d2 = s x s
iloss=fc.add_loss_2d(data1,data2,ldata,ldata,avg_ang=avg_ang)

# initiliaze the loss
loss=fc.init_optim()

c1,c2=fc.calc_stat(d1.reshape(1,nx,nx),d2.reshape(1,nx,nx),avg_ang=avg_ang)
lmap1=1*d1
lmap2=1*d2

doplot=False

for itt in range(10):
    # compute scattering to make weights
    modd1=lmap1.reshape(1,nx,nx)+ampnoise*np.random.randn(nsim,nx,nx)
    modd2=lmap2.reshape(1,nx,nx)+ampnoise*np.random.randn(nsim,nx,nx)
    o1,o2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang)

    modd1=lmap1.reshape(1,nx,nx)
    modd2=lmap2.reshape(1,nx,nx)
    r1,r2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang)
    
    moddr=d.reshape(1,nx,nx)
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
    
    tb1={}
    tb2={}
    tb1[0]=Alpha*np.mean(o1-r1,0)
    tb2[0]=Alpha*np.mean(o2-r2,0)
    
        
    # make the learn
    fc.reset()
    omap=fc.learn(tw1,tw2,tb1,tb2,NUM_EPOCHS = 3000,DECAY_RATE=0.98,LEARNING_RATE=0.01)
    print('ITT ',itt,((d-omap)*mask.reshape(nx,nx)).std(),((d-di)*mask.reshape(nx,nx)).std())

    modd1=omap.reshape(1,nx,nx)
    oo1,oo2=fc.calc_stat(modd1,modd1,avg_ang=avg_ang)
    doplot=False
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

