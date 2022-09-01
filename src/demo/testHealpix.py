import numpy as np
import os, sys
import matplotlib.pyplot as plt

#=================================================================================
# INITIALIZE FoCUS class
#=================================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR)+'/FoCUS')
import FoCUS as FOC

fc=FOC.FoCUS(NORIENT=4,KERNELSZ=3,healpix=True,gpupos=0,OSTEP=0,slope=1.0)

#=================================================================================
# READ data: Here the input data is a small picture of nside=64
#=================================================================================
def dodown(a,nout):
    nin=int(np.sqrt(a.shape[0]//12))
    if nin==nout:
        return(a)
    return(np.mean(a.reshape(12*nout*nout,(nin//nout)**2),1))

nout=256

d=dodown(np.load('../data/EE_vansingel_512.npy'),nout)
d=d/d.std()
td=dodown(np.load('../data/TT857_256.npy'),nout)
td=td/td.std()
# define the level of noise of the simulation
ampnoise=1.0
outname='HM'

# define the level of noise of the simulation
ampnoise=0.3
Alpha=1.0
nsim=100
avg_ang=True

# Build to half mission by adding noise to the data
d1 = d+ampnoise*np.random.randn(12*nout**2)
d2 = d+ampnoise*np.random.randn(12*nout**2)

# simulate the noisy data
di = d+ampnoise*np.random.randn(12*nout**2)/np.sqrt(2)

#=================================================================================
# For real data:
# you have to give d1,d2 and d
# you have to define the value ampnoise
#=================================================================================


# All information of the map is used
nin=256
tab=['MASK_GAL11_%d.npy'%(nin),'MASK_GAL09_%d.npy'%(nin),'MASK_GAL08_%d.npy'%(nin),'MASK_GAL06_%d.npy'%(nin),'MASK_GAL04_%d.npy'%(nin)]
mask=np.ones([len(tab),12*nout**2])
for imask in range(len(tab)):
    mask[imask]=dodown(np.load('/export/home/jmdeloui/heal_cnn/'+tab[imask]),nout)
mask[0,:]=1.0
fc.add_mask(mask)

# convert data in tensor for focus (should be done internally)
data1 = fc.convimage(d1)
data2 = fc.convimage(d2)
data  = fc.convimage(di)
tdata = fc.convimage(td)

# Initialize the learning and initialize the tensor to be synthesized
ldata=fc.init_synthese(di)

# Add losss:
# here d1 x d2 = s x s
iloss=fc.add_loss_healpix(data1,data2,ldata,ldata,avg_ang=avg_ang)
# here d1 x d2 = d x s
iloss=fc.add_loss_healpix(data,ldata,ldata,ldata,avg_ang=avg_ang)
# here T x d = T x s
iloss=fc.add_loss_healpix(tdata,data,tdata,ldata,avg_ang=avg_ang,imaginary=True)

# initiliaze the loss
loss=fc.init_optim()

moddr=d.reshape(1,12*nout**2)
x1,x2=fc.calc_stat(moddr,moddr,avg_ang=avg_ang)
np.save('../data/i%s1_%d.npy'%(outname,0), x1)
np.save('../data/i%s2_%d.npy'%(outname,0), x2)

c1,c2=fc.calc_stat(d1.reshape(1,12*nout**2),d2.reshape(1,12*nout**2),avg_ang=avg_ang)
np.save('../data/c%s1_%d.npy'%(outname,0), c1)
np.save('../data/c%s2_%d.npy'%(outname,0), c2)
lmap1=1*d1
lmap2=1*d2

for itt in range(5):
    # compute scattering to make weights
    modd1=lmap1.reshape(1,12*nout**2)+ampnoise*np.random.randn(nsim,12*nout**2)
    modd2=lmap2.reshape(1,12*nout**2)+ampnoise*np.random.randn(nsim,12*nout**2)
    o1,o2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang)
    modd1=lmap1.reshape(1,12*nout**2)+ampnoise*np.random.randn(nsim,12*nout**2)
    modd2=lmap2.reshape(1,12*nout**2)+0*np.random.randn(nsim,12*nout**2)
    of1,of2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang)
    modd1=lmap1.reshape(1,12*nout**2)
    modd2=lmap2.reshape(1,12*nout**2)
    r1,r2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang)
    
    modd1=td.reshape(1,12*nout**2)+0*np.random.randn(nsim,12*nout**2)
    modd2=((lmap1+lmap2)/2).reshape(1,12*nout**2)+ampnoise*np.random.randn(nsim,12*nout**2)
    onx1,onx2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,imaginary=True)
    modd1=td.reshape(1,12*nout**2)
    modd2=((lmap1+lmap2)/2).reshape(1,12*nout**2)
    ox1,ox2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,imaginary=True)
    

    #calcul poids des coeffs
    np.save('../data/b%s1_%d.npy'%(outname,itt), r1)
    np.save('../data/b%s2_%d.npy'%(outname,itt), r2)
    np.save('../data/n%s1_%d.npy'%(outname,itt), o1)
    np.save('../data/n%s2_%d.npy'%(outname,itt), o2)

    tw1={}
    tw2={}
    tw1[0]=1/np.std(o1,0)
    tw2[0]=1/np.std(o2,0)
    tw1[1]=1/np.std(of1,0)
    tw2[1]=1/np.std(of2,0)
    tw1[2]=1/np.std(onx1,0)
    tw2[2]=1/np.std(onx2,0)
    
    tb1={}
    tb2={}
    tb1[0]=Alpha*np.mean(o1-r1,0)
    tb2[0]=Alpha*np.mean(o2-r2,0)
    tb1[1]=Alpha*np.mean(of1-r1,0)
    tb2[1]=Alpha*np.mean(of2-r2,0)
    tb1[2]=Alpha*np.mean(onx1-ox1,0)
    tb2[2]=Alpha*np.mean(onx2-ox2,0)
    
    # make the learn
    fc.reset()
    omap=fc.learn(tw1,tw2,tb1,tb2,NUM_EPOCHS = 3000,DECAY_RATE=0.9995,LEARNING_RATE=0.03)
    print('ITT ',itt,((d-omap)*mask[1].reshape(12*nout**2)).std(),((d-di)*mask[1].reshape(12*nout**2)).std())

    modd1=omap.reshape(1,12*nout**2)
    oo1,oo2=fc.calc_stat(modd1,modd1,avg_ang=avg_ang)
    lmap1=1*omap
    lmap2=1*omap
        
    np.save('../data/o%s1_%d.npy'%(outname,itt), oo1)
    np.save('../data/o%s2_%d.npy'%(outname,itt), oo2)

    np.save('../data/test%sref_%d.npy'%(outname,itt), d)
    np.save('../data/test%sinput_%d.npy'%(outname,itt),di)
    np.save('../data/test%sinput1_%d.npy'%(outname,itt),d1)
    np.save('../data/test%sinput2_%d.npy'%(outname,itt),d2)
    np.save('../data/test%sresult_%d.npy'%(outname,itt),omap)

