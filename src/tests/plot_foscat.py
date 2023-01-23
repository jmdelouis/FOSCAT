import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

nout=64
idx=hp.ring2nest(nout,np.arange(12*nout**2))
outpath = '/home1/scratch/jmdeloui/heal_cnn/'
outname='FOCUS%s%d'%('EE0256',nout)

td=np.load(outpath+'/%std.npy'%(outname))
di=np.load(outpath+'/%sdi.npy'%(outname))
d1=np.load(outpath+'/%sd1.npy'%(outname))
d2=np.load(outpath+'/%sd2.npy'%(outname))
rr=np.load(outpath+'/%sresult_%d.npy'%(outname,0))

amp=300
plt.figure(figsize=(6,12))
hp.mollview(1E6*di[idx],cmap='jet',nest=False,min=-amp,max=amp,hold=False,sub=(3,1,1))
hp.mollview(1E6*rr,cmap='jet',nest=True,min=-amp,max=amp,hold=False,sub=(3,1,2))
hp.mollview(1E6*(di-rr),cmap='jet',nest=True,min=-amp,max=amp,hold=False,sub=(3,1,3))

clin=hp.anafast(1E6*di[idx])
clout=hp.anafast(1E6*rr[idx])
cldiff=hp.anafast(1E6*(di[idx]-rr[idx]))

plt.figure(figsize=(6,6))
plt.plot(clin,color='blue')
plt.plot(clout,color='red')
plt.xscale('log')
plt.yscale('log')
plt.show()
