import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def dodown(a,nout):
    nin=int(np.sqrt(a.shape[0]//12))
    if nin==nout:
        return(a)
    return(np.mean(a.reshape(12*nout*nout,(nin//nout)**2),1))

s1 = np.load('in_demo_s1.npy')
s2 = np.load('in_demo_s2.npy')
os1= np.load('out_demo_s1.npy')
os2= np.load('out_demo_s2.npy')
im = np.load('in_demo_map.npy')
om = np.load('out_demo_map.npy')

#make the galactic ridge positive
if abs(om).max()>om.max():
    om=-om
    
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(s1.flatten(),color='blue',label=r'Model $S_1$')
plt.plot(os1.flatten(),color='red',label=r'Output $S_1$')
plt.yscale('log')
plt.legend()
plt.subplot(1,2,2)
plt.plot(s2.flatten(),color='blue',label=r'Model $S_2$')
plt.plot(os2.flatten(),color='red',label=r'Output $S_2$')
plt.yscale('log')
plt.legend()

nside=int(np.sqrt(im.shape[0]//12))

idx=hp.ring2nest(nside,np.arange(12*nside**2))
plt.figure(figsize=(8,8))
hp.mollview(im[idx],cmap='jet',norm='hist',hold=False,sub=(2,1,1),nest=False,title='Model')
hp.mollview(om,cmap='jet',norm='hist',hold=False,sub=(2,1,2),nest=True,title='Output')

cli=hp.anafast((im-np.median(im))[idx])
clo=hp.anafast((om-np.median(om))[idx])

plt.figure(figsize=(6,6))
plt.plot(cli,color='blue',label=r'Model')
plt.plot(clo,color='red',label=r'Output')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.xlabel('Multipoles')
plt.ylabel('C(l)')
plt.show()
