import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import sys        
import foscat.scat as sc

if len(sys.argv)<4:
    print('\nhwst_foscat usage:\n')
    print('python plot_foscat <in> <scratch_path> <out> <nside> <step>')
    print('============================================')
    print('<in>           : name of the 3 input data files: <in>_MONO.npy,<in>_HM1_MONO.npy,<in>_HM2_MONO.npy')
    print('<scratch_path> : name of the directory with all the input files (noise, TT,etc.) and also use for FOSCAT temporary files')
    print('<out>          : name of the directory where the computed data are stored')
    print('<nside>        : nside of the synthesised map')
    print('<step>         : iteration number to plot')
    print('============================================')
    exit(0)

scratch_path = sys.argv[2]
datapath     = scratch_path
outpath      = sys.argv[3]
nside        = int(sys.argv[4])
step         = int(sys.argv[5])

idx=hp.ring2nest(nside,np.arange(12*nside**2))
outname='FOCUS%s%d'%(sys.argv[1],nside)

print(outname)

try:
    ref=np.mean(np.load(datapath+'%s_REF.npy'%(sys.argv[1])).reshape(12*nside**2,(256//nside)**2),1)
except:
    ref=None
td=np.load(outpath+'/%std.npy'%(outname))
di=np.load(outpath+'/%sdi.npy'%(outname))
d1=np.load(outpath+'/%sd1.npy'%(outname))
d2=np.load(outpath+'/%sd2.npy'%(outname))
rr=np.load(outpath+'/%sresult_%d.npy'%(outname,step))

try:
    smod=sc.read(outpath+'/%s_cross_%d.npy'%(outname,step))
    sin=sc.read(outpath+'/%s_in_%d.npy'%(outname,step))
    sout=sc.read(outpath+'/%s_out_%d.npy'%(outname,step))

    smod.plot(name='Model',lw=4)
    sin.plot(name='In',hold=False,color='red')
    sout.plot(name='Out',hold=False,color='yellow')
    (smod-sin).plot(name='Diff In',hold=False,color='grey')
    (smod-sout).plot(name='Diff Out',hold=False,color='darkgrey')
except:
    print('no scat computed')
    
amp=300
amp2=amp/5

plt.figure(figsize=(12,4))
hp.mollview(1E6*di[idx],cmap='jet',nest=False,min=-amp,max=amp,hold=False,sub=(2,3,1),title='d')
if ref is not None:
    hp.mollview(1E6*ref[idx],cmap='jet',nest=False,min=-amp,max=amp,hold=False,sub=(2,3,2),title='s')
hp.mollview(1E6*rr,cmap='jet',nest=True,min=-amp,max=amp,hold=False,sub=(2,3,3),title='u')
if ref is not None:
    hp.mollview(1E6*(di-ref),cmap='jet',nest=True,min=-amp2,max=amp2,hold=False,sub=(2,3,4),title='d-s')
hp.mollview(1E6*(di-rr),cmap='jet',nest=True,min=-amp2,max=amp2,hold=False,sub=(2,3,5),title='d-u')
if ref is not None:
    hp.mollview(1E6*(ref-rr),cmap='jet',nest=True,min=-amp2,max=amp2,hold=False,sub=(2,3,6),title='s-y')

tab=['08','06','04','02']

plt.figure(figsize=(12,12))

for i in range(len(tab)):
    mm=np.mean(np.load(scratch_path+'/MASK_GAL%s_256.npy'%(tab[i])).reshape(12*nside**2,(256//nside)**2),1)

    clin=hp.anafast(1E6*(mm*di)[idx])
    clout=hp.anafast(1E6*(mm*rr)[idx])
    cldiff=hp.anafast(1E6*(mm*(di[idx]-rr[idx])))
    clx=hp.anafast(1E6*(mm*d1)[idx],map2=1E6*(mm*d2)[idx])
    clox=hp.anafast(1E6*(mm*di)[idx],map2=1E6*(mm*rr)[idx])
    if ref is not None:
        clr=hp.anafast(1E6*(mm*(ref-rr))[idx])
    cln=hp.anafast(1E6*(mm*(di-rr))[idx])
    if ref is not None:
        clauto=hp.anafast(1E6*(mm*(ref))[idx])

    plt.subplot(2,2,1+i)
    plt.plot(clin,   color='blue',  label='d*d fsky=%.2f'%(mm.mean()))
    plt.plot(clx,    color='grey',  label='d1*d2')
    plt.plot(cln,    color='lightgrey',  label='d-s')
    if ref is not None:
        plt.plot(clauto, color='black', label='s*s',lw=6)
    plt.plot(clox,   color='orange',label='d*s')
    plt.plot(clout,  color='purple',label='u*u')
    if ref is not None:
        plt.plot(clr  ,  color='red',   label='s-u')
    plt.xscale('log')
    plt.yscale('log')
    if i==0:
        plt.legend(frameon=0)
plt.show()
