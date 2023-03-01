import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import sys        

if len(sys.argv)<4:
    print('\nhwst_foscat usage:\n')
    print('python plot_foscat <in> <scratch_path> <out> <nside> <step>')
    print('============================================')
    print('<in>           : name of the 3 input data files: <in>_MONO.npy,<in>_HM1_MONO.npy,<in>_HM2_MONO.npy')
    print('<scratch_path> : name of the directory with all the input files (noise, TT,etc.) and also use for FOSCAT temporary files')
    print('<out>          : name of the directory where the computed data are stored')
    print('<nside>        : nside of the synthesised map')
    print('<step>         : iteration number to plot')
    print('<cov>          : if Y use scat_cov istead of scat')
    print('<kernelsz>     : kernelsz of the convolution')
    print('============================================')
    exit(0)

scratch_path = sys.argv[2]
datapath     = scratch_path
outpath      = sys.argv[3]
nside        = int(sys.argv[4])
step         = int(sys.argv[5])
docov        = (sys.argv[6]=='Y')
kernelsz     = int(sys.argv[7])

if docov:
    import foscat.scat_cov as sc
else:
    import foscat.scat as sc
    
idx=hp.ring2nest(nside,np.arange(12*nside**2))
if kernelsz==5:
    outname='FOCUS_5x5%s%d'%(sys.argv[1],nside)
else:
    outname='FOCUS%s%d'%(sys.argv[1],nside)

print(outname)

ampmap=36522.24321517236

try:
    ref=np.mean(np.load(datapath+'%s_REF.npy'%(sys.argv[1])).reshape(12*nside**2,(256//nside)**2),1)
except:
    ref=None

if kernelsz==3:
    lam=1.2
else:
    lam=1.0
    
scat_op=sc.funct(NORIENT=4,   # define the number of wavelet orientation
                 KERNELSZ=kernelsz,  # define the kernel size (here 5x5)
                 OSTEP=-1,     # get very large scale (nside=1)
                 LAMBDA=lam,
                 all_type='float64',
                 use_R_format=True)

#=================================================================================
# Function to reduce the data used in the FoCUS algorithm 
#=================================================================================
def dodown(a,nout):
    nin=int(np.sqrt(a.shape[0]//12))
    if nin==nout:
        return(a)
    return(np.mean(a.reshape(12*nout*nout,(nin//nout)**2),1))

nin=256
tab=['MASK_GAL11_%d.npy'%(nin),'MASK_GAL09_%d.npy'%(nin),'MASK_GAL08_%d.npy'%(nin),'MASK_GAL06_%d.npy'%(nin),'MASK_GAL04_%d.npy'%(nin)]
mask=np.ones([len(tab),12*nside**2])
for i in range(len(tab)):
    mask[i,:]=dodown(np.load(datapath+tab[i]),nside)

#set the first mask to 1
mask[0,:]=1.0
for i in range(1,len(tab)):
    mask[i,:]=mask[i,:]*mask[0,:].sum()/mask[i,:].sum()

if ref is not None:
    scref=scat_op.eval((ref+6.981021657074907e-05)*ampmap,mask=mask)
else:
    scref=None
    
td=np.load(outpath+'/%std.npy'%(outname))
di=np.load(outpath+'/%sdi.npy'%(outname))
d1=np.load(outpath+'/%sd1.npy'%(outname))
d2=np.load(outpath+'/%sd2.npy'%(outname))
rr=np.load(outpath+'/%sresult_%d.npy'%(outname,step))
print(outpath+'/%sresult_%d.npy'%(outname,step))

try:
    smod=sc.read(outpath+'/%s_cross_%d.npy'%(outname,step))
    sin=sc.read(outpath+'/%s_in_%d.npy'%(outname,step))
    sout=sc.read(outpath+'/%s_out_%d.npy'%(outname,step))
    b1=sc.read(outpath+'/%s_bias1_%d.npy'%(outname,step))
    b2=sc.read(outpath+'/%s_bias2_%d.npy'%(outname,step))
    b3=sc.read(outpath+'/%s_bias3_%d.npy'%(outname,step))

    if ref is not None:
        scref.plot(name='Model',lw=4)
        
    smod.plot(name='cross',hold=False,color='purple')
    (smod-b1).plot(name='cross debias',hold=False,color='orange')
    sin.plot(name='In',hold=False,color='red')
    sout.plot(name='Out',hold=False,color='yellow')
    
    if ref is not None:
        (scref-sout).plot(name='Diff Out',hold=False,color='black')
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
    mm=np.mean(np.load(scratch_path+'/MASK_GAL%s_256.npy'%(tab[i])).reshape(12*nside**2,(256//nside)**2),1)[idx]

    mm=hp.smoothing(mm,5/180.*np.pi)
    
    clin=hp.anafast(1E6*mm*di[idx])
    clout=hp.anafast(1E6*mm*rr[idx])
    cldiff=hp.anafast(1E6*mm*(di[idx]-rr[idx]))
    clx=hp.anafast(1E6*mm*d1[idx],map2=1E6*mm*d2[idx])
    clox=hp.anafast(1E6*mm*di[idx],map2=1E6*mm*rr[idx])
    if ref is not None:
        clr=hp.anafast(1E6*mm*(ref-rr)[idx])
    cln=hp.anafast(1E6*mm*(di-rr)[idx])
    if ref is not None:
        clauto=hp.anafast(1E6*mm*(ref)[idx])

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
