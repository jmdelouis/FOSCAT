import numpy as np
import os, sys
import matplotlib.pyplot as plt
import healpy as hp

#=================================================================================
# DEFINE A PATH FOR scratch data
# The data are storred using a default nside to minimize the needed storage
#=================================================================================
scratch_path = '/home1/scratch/jmdeloui/data_foscat/'
data_path='/home1/datawork/jmdeloui/heal_cnn/'

outname='TEST2_EE'
Default_nside=512

#=================================================================================
# INITIALIZE FoCUS class
#=================================================================================
import foscat.FoCUS as FOC

fc=FOC.FoCUS(NORIENT=4,
             KERNELSZ=3,
             healpix=True,
             OSTEP=0,
             slope=1.0,
             isMPI=True,
             TEMPLATE_PATH=scratch_path)


#=================================================================================
# DENOISE NSIDE=32 MAP
#=================================================================================

nout=256

#=================================================================================
# Function to reduce the data used in the FoCUS algorithm 
#=================================================================================
def dodown(a,nout):
    nin=int(np.sqrt(a.shape[0]//12))
    if nin==nout:
        return(a)
    return(np.mean(a.reshape(12*nout*nout,(nin//nout)**2),1))

#=================================================================================
# Get data
#=================================================================================

# define the level of noise of the simulation
ampnoise=0.4
Alpha=1.0
DAlpha=0.8
nsim=100
avg_ang=False

# Read data from disk
di=dodown(np.load(datapath+'%s_MONO.npy'%(sys.argv[1])),nout)
td=dodown(np.load(datapath+'TT857_%d.npy'%(Default_nside)),nout)
d=dodown(np.load(datapath+'%s_vansingel_512.npy'%(sys.argv[1][0:2])),nout)
aa=np.median(d*di)/np.median(d*d)
d=aa*d
d1=dodown(np.load(data+'%s_HM1_MONO.npy'%(sys.argv[1])),nout)
d2=dodown(np.load(datapath+'%s_HM2_MONO.npy'%(sys.argv[1])),nout)
ooo=1E-4
#di=d+np.random.randn(12*nout*nout)*ooo
#d1=d+np.sqrt(2)*np.random.randn(12*nout*nout)*ooo
#d2=d+np.sqrt(2)*np.random.randn(12*nout*nout)*ooo

if fc.get_rank()==0%fc.get_size():
    np.save(scratch_path+'/test%sdi.npy'%(outname),di)
    np.save(scratch_path+'/test%sd.npy'%(outname),d)
    np.save(scratch_path+'/test%sd1.npy'%(outname),d1)
    np.save(scratch_path+'/test%sd2.npy'%(outname),d2)

fc.barrier()
#=================================================================================
# For real data:
# you have to give d1,d2 and d
# you have to define the value ampnoise
#=================================================================================

di=np.load(scratch_path+'/test%sdi.npy'%(outname))
d=np.load(scratch_path+'/test%sd.npy'%(outname))
d1=np.load(scratch_path+'/test%sd1.npy'%(outname))
d2=np.load(scratch_path+'/test%sd2.npy'%(outname))

# All information of the map is used
nin=256

tab=['MASK_GAL11_%d.npy'%(nin),'MASK_GAL09_%d.npy'%(nin),'MASK_GAL08_%d.npy'%(nin),'MASK_GAL06_%d.npy'%(nin),'MASK_GAL04_%d.npy'%(nin)]
mask=np.ones([len(tab),12*nout**2])
for i in range(len(tab)):
    mask[i,:]=dodown(np.load(datapath+tab[i]),nout)
mask[0,:]=1.0
for i in range(1,len(tab)):
    mask[i,:]=mask[i,:]*mask[0,:].sum()/mask[i,:].sum()
    
fc.add_mask(mask)

#=============================================
# fill empty pixels

off=np.median(di[di>-1E10])
d[di<-1E10]=off
d1[di<-1E10]=off
d2[di<-1E10]=off
di[di<-1E10]=off

#=============================================
# compute amplitude to normalize the dynamic range
ampmap=1/di[mask[1]>0.9].std()

# convert data in tensor for focus (should be done internally)
data1 = fc.convimage(ampmap*(d1-off))
data2 = fc.convimage(ampmap*(d2-off))
data  = fc.convimage(ampmap*(di-off))
tdata = fc.convimage(ampmap*(td))

#compute all noise map statistics
noise=np.zeros([nsim,12*nout*nout])
noise1=np.zeros([nsim,12*nout*nout])
noise2=np.zeros([nsim,12*nout*nout])
for i in range(nsim):
    noise[i] =ampmap*dodown(np.load(scratch_path+'%s_NOISE%03d_full.npy'%(sys.argv[1],i)).flatten(),nout)
    noise1[i]=ampmap*dodown(np.load(scratch_path+'%s_NOISE%03d_hm1.npy'%(sys.argv[1],i)).flatten(),nout)
    noise2[i]=ampmap*dodown(np.load(scratch_path+'%s_NOISE%03d_hm2.npy'%(sys.argv[1],i)).flatten(),nout)

for i in range(nsim):
    noise1[i]-=np.mean(noise[i])
    noise2[i]-=np.mean(noise[i])
    noise[i] -=np.mean(noise[i])

if fc.get_rank()==0%fc.get_size():
    np.save(scratch_path+'/test%snoise.npy'%(outname),noise)
    np.save(scratch_path+'/test%snoise1.npy'%(outname),noise1)
    np.save(scratch_path+'/test%snoise2.npy'%(outname),noise2)

fc.barrier()

#============

noise=np.load(scratch_path+'test%snoise.npy'%(outname))
noise1=np.load(scratch_path+'test%snoise1.npy'%(outname))
noise2=np.load(scratch_path+'test%snoise2.npy'%(outname))
    
# Initialize the learning and initialize the tensor to be synthesized
ldata=fc.init_synthese(ampmap*(di-off),ampmap*(d1-off),ampmap*(d2-off))

# Add losss:
# here d1 x d2 = s x s
if fc.get_rank()==0%fc.get_size():
    fc.add_loss_healpix(data1,data2,ldata,ldata,avg_ang=avg_ang,imaginary=True)
# here T x d = T x s
if fc.get_rank()==1%fc.get_size():
    fc.add_loss_healpix(tdata,data,tdata,ldata,avg_ang=avg_ang,imaginary=True)
# here d1 x d2 = d x s
if fc.get_rank()==2%fc.get_size():
    fc.add_loss_healpix(data1,data2,data,ldata,avg_ang=avg_ang,imaginary=True)
    
# Add losss:
# here (d-s)^2 = avg(noise^2)
if fc.get_rank()==1%fc.get_size():
    fc.add_loss_noise_stat(mask,noise,data-ldata,weight=100)

# initiliaze the loss
loss=fc.init_optim()
    
# Use a reference map to compute bias
rmap  = ampmap*(d-off)
rmap1 = ampmap*(d1-off)
rmap2 = ampmap*(d2-off)

tw1={}
tw2={}
tb1={}
tb2={}
for i in range(3):
    tw1[i]=0.0
    tw2[i]=0.0
    tb1[i]=0.0
    tb2[i]=0.0

for itt in range(6):
    # compute bias and variance using noise model
    if fc.get_rank()==0%fc.get_size() or fc.get_rank()==2%fc.get_size():
        # compute scattering to make weights
        modd1=rmap1.reshape(1,12*nout**2)+noise1
        modd2=rmap2.reshape(1,12*nout**2)+noise2
        o1,o2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,gpupos=0,imaginary=True)
        modd1=rmap1.reshape(1,12*nout**2)
        modd2=rmap2.reshape(1,12*nout**2)
        r1,r2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,gpupos=0,imaginary=True)
        
    if fc.get_rank()==1%fc.get_size():
        modd1=(ampmap*td).reshape(1,12*nout**2)+0*noise
        modd2=rmap.reshape(1,12*nout**2)+noise
        onx1,onx2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,imaginary=True,gpupos=2)
        modd1=(ampmap*td).reshape(1,12*nout**2)
        modd2=rmap.reshape(1,12*nout**2)
        ox1,ox2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,imaginary=True,gpupos=2)
        
    if fc.get_rank()==2%fc.get_size():
        modd1=rmap1.reshape(1,12*nout**2)+noise
        modd2=rmap2.reshape(1,12*nout**2)+0*noise
        of1,of2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,gpupos=1,imaginary=True)

    if itt==0:
        # compute de variance only at the first itteration, otherwise the loss tend to increase the variance
        if fc.get_rank()==0%fc.get_size():
            tw1[0]=1/np.std(o1-r1,0)
            tw2[0]=1/np.std(o2-r2,0)

        if fc.get_rank()==1%fc.get_size():
            ii=0
            if fc.get_size()==1:
                ii=1
            tw1[ii]=1/np.std(onx1-ox1,0)
            tw2[ii]=1/np.std(onx2-ox2,0)

        if fc.get_rank()==2%fc.get_size():
            ii=0
            if fc.get_size()==1:
                ii=2
            tw1[ii]=1/np.std(o1-r1,0)
            tw2[ii]=1/np.std(o2-r2,0)

    #compute the bias at each iteration
    if fc.get_rank()==0%fc.get_size():
        tb1[0]=Alpha*(np.mean(o1-r1,0)-tb1[0])+tb1[0]
        tb2[0]=Alpha*(np.mean(o2-r2,0)-tb2[0])+tb2[0]

    if fc.get_rank()==1%fc.get_size():
        ii=0
        if fc.get_size()==1:
            ii=1
        tb1[ii]=Alpha*(np.mean(onx1-ox1,0)-tb1[ii])+tb1[ii]
        tb2[ii]=Alpha*(np.mean(onx2-ox2,0)-tb2[ii])+tb2[ii]

    if fc.get_rank()==2%fc.get_size():
        ii=0
        if fc.get_size()==1:
            ii=2
        tb1[ii]=Alpha*(np.mean(o1-r1,0)-np.mean(of1-r1,0)-tb1[ii])+tb1[ii]
        tb2[ii]=Alpha*(np.mean(o2-r2,0)-np.mean(of2-r2,0)-tb2[ii])+tb2[ii]
    
    Alpha*=DAlpha

    # reset the correction to 0
    fc.reset()
        
    omap=fc.learn(tw1,tw2,tb1,tb2,
                  NUM_EPOCHS = 1000,
                  EVAL_FREQUENCY = 10,
                  DECAY_RATE=0.995,
                  LEARNING_RATE=0.03,
                  SEQUENTIAL_ITT=10,
                  ADDAPT_LEARN=10.0)

    # save the reult
    if fc.get_rank()==0%fc.get_size():
        print('ITT %d fsky=%.2f std_origin=%9.3f std_corr=%9.3f'%(itt,np.mean(mask[3]),
                                                        ((d-di)*mask[3].reshape(12*nout**2)).std(),
                                                        ((d-(omap/ampmap+off))*mask[3].reshape(12*nout**2)).std()))
        sys.stdout.flush()
        modd1=omap.reshape(1,12*nout**2)
        oo1,oo2=fc.calc_stat(modd1,modd1,avg_ang=avg_ang)
        np.save(scratch_path+'o%s1_%d.npy'%(outname,itt), oo1)
        np.save(scratch_path+'o%s2_%d.npy'%(outname,itt), oo2)
        np.save(scratch_path+'test%sresult_%d.npy'%(outname,itt),omap/ampmap+off)

    rmap=1*omap
    rmap1=rmap
    rmap2=rmap

print('Computation Done')
sys.stdout.flush()

