import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def dodown(a,nout):
    nin=int(np.sqrt(a.shape[0]//12))
    if nin==nout:
        return(a)
    return(np.mean(a.reshape(12*nout*nout,(nin//nout)**2),1))

s1 = np.load('in_cov_s1.npy')
s2 = np.load('in_cov_s2.npy')
s3 = np.load('in_cov_s3.npy')
s4 = np.load('in_cov_s4.npy')
r1 = np.load('st_cov_s1.npy')
r2 = np.load('st_cov_s2.npy')
r3 = np.load('st_cov_s3.npy')
r4 = np.load('st_cov_s4.npy')

os1= np.load('out_cov_s1.npy')
os2= np.load('out_cov_s2.npy')
os3= np.load('out_cov_s3.npy')
os4= np.load('out_cov_s4.npy')

im = np.load('in_map.npy')
om = np.load('out_map.npy')

#make the galactic ridge positive
if abs(om).max()>om.max():
    om=-om
    
plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.plot(r1.flatten(),color='orange',label=r'Init $\log{S_1}$')
plt.plot(s1.flatten(),color='blue',label=r'Model $\log{S_1}$',lw=4)
plt.plot(os1.flatten(),color='red',label=r'Output $\log{S_1}$')
#plt.yscale('log')
plt.legend()
plt.subplot(2,2,2)
plt.plot(r2.flatten(),color='orange',label=r'Init $\log{P_{00}}$')
plt.plot(s2.flatten(),color='blue',label=r'Model $\log{P_{00}}$',lw=4)
plt.plot(os2.flatten(),color='red',label=r'Output $\log{P_{00}}$')
#plt.yscale('log')
plt.legend()

plt.subplot(2,2,3)
plt.plot(abs(r3).flatten(),color='orange',label=r'Init $C_{01}}$')
plt.plot(abs(s3).flatten(),color='blue',label=r'Model $C_{01}$',lw=4)
plt.plot(abs(os3).flatten(),color='red',label=r'Output $C_{01}$')
plt.yscale('log')
plt.legend()

plt.subplot(2,2,4)
plt.plot(abs(r4).flatten(),color='orange',label=r'Init $C_{11}}$')
plt.plot(abs(s4).flatten(),color='blue',label=r'Model $C_{11}$',lw=4)
plt.plot(abs(os4).flatten(),color='red',label=r'Output $C_{11}$')
plt.yscale('log')
plt.legend()

nside=int(np.sqrt(im.shape[0]//12))

idx=hp.ring2nest(nside,np.arange(12*nside**2))
plt.figure(figsize=(6,8))
hp.mollview(im[idx],cmap='jet',hold=False,sub=(2,1,1),nest=False,title='Input',min=-3,max=3)
hp.mollview(om,cmap='jet',hold=False,sub=(2,1,2),nest=True,title='Output',min=-3,max=3)

cli=hp.anafast((im -np.median(im))[idx])
clo=hp.anafast((om -np.median(om))[idx])

plt.figure(figsize=(6,6))
plt.plot(cli,color='blue',label=r'Input')
plt.plot(clo,color='red',label=r'Output')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.xlabel('Multipoles')
plt.ylabel('C(l)')
plt.show()
