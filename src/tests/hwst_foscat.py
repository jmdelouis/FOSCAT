import numpy as np
import os, sys
import matplotlib.pyplot as plt
import healpy as hp

#=================================================================================
# DEFINE A PATH FOR scratch data
# The data are storred using a default nside to minimize the needed storage
#=================================================================================

if len(sys.argv)<4:
    print('\nhwst_foscat usage:\n')
    print('python hwst_foscat <in> <scratch_path> <out>')
    print('============================================')
    print('<in>           : name of the 3 input data files: <in>_MONO.npy,<in>_HM1_MONO.npy,<in>_HM2_MONO.npy')
    print('<scratch_path> : name of the directory with all the input files (noise, TT,etc.) and also use for FOSCAT temporary files')
    print('<out>          : name of the directory where the computed data are stored')
    print('============================================')
    exit(0)

scratch_path = sys.argv[2]
datapath = scratch_path
outpath = sys.argv[3]

#set the nside of input data
Default_nside=256

#=================================================================================
# DEFINE THE WORKING NSIDE
#=================================================================================

nout=32
# set the default name
outname='FOCUS%s%d'%(sys.argv[1],nout)

#=================================================================================
# INITIALIZE FoCUS class
#=================================================================================
import foscat.FoCUS as FOC

fc=FOC.FoCUS(NORIENT=4,
             KERNELSZ=3,
             healpix=True,
             OSTEP=0,
             slope=1.2,
             TEMPLATE_PATH=scratch_path)

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
# coeficient to control bias iteration
Alpha=1.0
DAlpha=1.0

#number of simulations used as reference
nsim=100

#work with angle invariant statistics 
avg_ang=False

# Read data from disk
td=dodown(np.load(datapath+'TT857_%d.npy'%(Default_nside)),nout)
di=dodown(np.load(datapath+'%s_MONO.npy'%(sys.argv[1])),nout)
d1=dodown(np.load(datapath+'%s_HM1_MONO.npy'%(sys.argv[1])),nout)
d2=dodown(np.load(datapath+'%s_HM2_MONO.npy'%(sys.argv[1])),nout)

# save data for analysis notebook
if fc.get_rank()==0%fc.get_size():
    np.save(outpath+'/%sdi.npy'%(outname),di)
    np.save(outpath+'/%sd1.npy'%(outname),d1)
    np.save(outpath+'/%sd2.npy'%(outname),d2)

fc.barrier()

di=np.load(outpath+'/%sdi.npy'%(outname))
d1=np.load(outpath+'/%sd1.npy'%(outname))
d2=np.load(outpath+'/%sd2.npy'%(outname))

# All information of the map is used
nin=Default_nside

tab=['MASK_GAL11_%d.npy'%(nin),'MASK_GAL09_%d.npy'%(nin),'MASK_GAL08_%d.npy'%(nin),'MASK_GAL06_%d.npy'%(nin),'MASK_GAL04_%d.npy'%(nin)]
mask=np.ones([len(tab),12*nout**2])
for i in range(len(tab)):
    mask[i,:]=dodown(np.load(datapath+tab[i]),nout)

#set the first mask to 1
mask[0,:]=1.0
for i in range(1,len(tab)):
    mask[i,:]=mask[i,:]*mask[0,:].sum()/mask[i,:].sum()
    
fc.add_mask(mask)

#=============================================
# fill empty pixels

off=np.median(di[di>-1E10])
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
    noise[i] =ampmap*dodown(np.load(scratch_path+'%s_NOISE%03d_full.npy'%(sys.argv[1][0:6],i)).flatten(),nout)
    noise1[i]=ampmap*dodown(np.load(scratch_path+'%s_NOISE%03d_hm1.npy'%(sys.argv[1][0:6],i)).flatten(),nout)
    noise2[i]=ampmap*dodown(np.load(scratch_path+'%s_NOISE%03d_hm2.npy'%(sys.argv[1][0:6],i)).flatten(),nout)

for i in range(nsim):
    noise1[i]-=np.mean(noise[i])
    noise2[i]-=np.mean(noise[i])
    noise[i] -=np.mean(noise[i])

fc.barrier()

#============
    
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
rmap  = ampmap*(di-off)
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

for itt in range(10):
    #=============================================
    # compute bias and variance using noise model
    #=============================================
    # here (r1+n1)x(r2+n2) and (r1)x(r2)
    if fc.get_rank()==0%fc.get_size() or fc.get_rank()==2%fc.get_size():
        # compute scattering to make weights
        modd1=rmap1.reshape(1,12*nout**2)+noise1
        modd2=rmap2.reshape(1,12*nout**2)+noise2
        o1,o2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,gpupos=0,imaginary=True)
        modd1=rmap1.reshape(1,12*nout**2)
        modd2=rmap2.reshape(1,12*nout**2)
        r1,r2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,gpupos=0,imaginary=True)
        
    # here (t)x(r+n) and (t)x(r)
    if fc.get_rank()==1%fc.get_size():
        modd1=(ampmap*td).reshape(1,12*nout**2)+0*noise
        modd2=rmap.reshape(1,12*nout**2)+noise
        onx1,onx2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,imaginary=True,gpupos=2)
        modd1=(ampmap*td).reshape(1,12*nout**2)
        modd2=rmap.reshape(1,12*nout**2)
        ox1,ox2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,imaginary=True,gpupos=2)
        
    # here (r+n)x(r)
    if fc.get_rank()==2%fc.get_size():
        modd1=rmap1.reshape(1,12*nout**2)+noise
        modd2=rmap2.reshape(1,12*nout**2)+0*noise
        of1,of2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,gpupos=1,imaginary=True)

    #=============================================
    # refresh variance
    #=============================================
    
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

    #=============================================
    # refresh bias
    #=============================================
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
    
    #decrease the correction after several iterations
    Alpha*=DAlpha

    # reset the correction to 0
    fc.reset()
        
    omap=fc.learn(tw1,tw2,tb1,tb2,
                  NUM_EPOCHS = 1000,
                  EVAL_FREQUENCY = 10,
                  DECAY_RATE=0.998,
                  LEARNING_RATE=0.03,
                  SEQUENTIAL_ITT=10,
                  ADDAPT_LEARN=10.0)

    # save the intermediate results
    if fc.get_rank()==0%fc.get_size():
        print('ITT %d DONE'%(itt))
        sys.stdout.flush()
        modd1=omap.reshape(1,12*nout**2)
        oo1,oo2=fc.calc_stat(modd1,modd1,avg_ang=avg_ang)
        np.save(outpath+'o%s1_%d.npy'%(outname,itt), oo1)
        np.save(outpath+'o%s2_%d.npy'%(outname,itt), oo2)
        np.save(outpath+'%sresult_%d.npy'%(outname,itt),omap/ampmap+off)

    rmap=1*omap
    rmap1=rmap
    rmap2=rmap

# save the intermediate results
if fc.get_rank()==0%fc.get_size():
    print('SAVE FINAL RESULT')
    sys.stdout.flush()
    modd1=omap.reshape(1,12*nout**2)
    oo1,oo2=fc.calc_stat(modd1,modd1,avg_ang=avg_ang)
    np.save(outpath+'o%s1.npy'%(outname), oo1)
    np.save(outpath+'o%s2.npy'%(outname), oo2)
    np.save(outpath+'%sresult.npy'%(outname),omap/ampmap+off)
print('Computation Done')
sys.stdout.flush()

