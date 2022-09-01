import numpy as np
import os, sys
import healpy as hp
import matplotlib.pyplot as plt

nitt=1
outname='HM'
plt.figure(figsize=(8,8))
#reference
i1=np.load('../data/i%s1_%d.npy'%(outname,0))
i2=np.load('../data/i%s2_%d.npy'%(outname,0))
#mesure
c1=np.load('../data/c%s1_%d.npy'%(outname,0))
c2=np.load('../data/c%s2_%d.npy'%(outname,0))

for itt in range(nitt):
    
    #Focus map
    o1=np.load('../data/o%s1_%d.npy'%(outname,itt))
    o2=np.load('../data/o%s2_%d.npy'%(outname,itt))

    plt.subplot(nitt,2,1+itt*2)
    plt.plot(i1.flatten(),color='blue')
    plt.plot(c1.flatten(),color='orange')
    plt.plot(o1.flatten(),color='red')
    plt.yscale('log')
    plt.ylabel(r'$S_1$')
    plt.xlabel('Itt=%d'%(itt))
    plt.subplot(nitt,2,2+itt*2)
    plt.plot(i2.flatten(),color='blue')
    plt.plot(c2.flatten(),color='orange')
    plt.plot(o2.flatten(),color='red')
    plt.yscale('log')
    plt.ylabel(r'$S_2$')
    plt.xlabel('Itt=%d'%(itt))
    
for itt in range(0,nitt):
    d  = np.load('../data/test%sref_%d.npy'%(outname,itt))
    d1 = np.load('../data/test%sinput1_%d.npy'%(outname,itt))
    d2 = np.load('../data/test%sinput2_%d.npy'%(outname,itt))
    di = np.load('../data/test%sinput_%d.npy'%(outname,itt))
    s  = np.load('../data/test%sresult_%d.npy'%(outname,itt)).flatten()

    amp=1
    
    plt.figure(figsize=(12,6))
    hp.mollview(d,cmap='jet',    nest=True,hold=False,sub=(2,3,1),min=-amp,max=amp,title='Model')
    hp.mollview(di,cmap='jet',   nest=True,hold=False,sub=(2,3,2),min=-amp,max=amp,title='Noisy')
    hp.mollview(s,cmap='jet',    nest=True,hold=False,sub=(2,3,3),min=-amp,max=amp,title='Cleanned')
    hp.mollview(di-d,cmap='jet', nest=True,hold=False,sub=(2,3,4),min=-amp,max=amp,title='Noisy-Model')
    hp.mollview(di-s,cmap='jet', nest=True,hold=False,sub=(2,3,5),min=-amp,max=amp,title='Noisy-Cleanned')
    hp.mollview(s-d,cmap='jet',  nest=True,hold=False,sub=(2,3,6),min=-amp,max=amp,title='Cleanned-Model')
plt.show()
