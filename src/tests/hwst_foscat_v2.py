import numpy as np
import os, sys
import matplotlib.pyplot as plt
import healpy as hp

#=================================================================================
# DEFINE A PATH FOR scratch data
# The data are storred using a default nside to minimize the needed storage
#=================================================================================
#python hwst_foscat_v2.py EE0256 /export/home/jmdeloui/heal_cnn/ /home1/scratch/jmdeloui/heal_cnn/
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

nout=256
# set the default name
outname='FOCUS%s%d'%(sys.argv[1],nout)

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
try:
    di=np.load(outpath+'/%sdi.npy'%(outname)) 
    d1=np.load(outpath+'/%sd1.npy'%(outname)) 
    d2=np.load(outpath+'/%sd2.npy'%(outname)) 
    td=np.load(outpath+'/%std.npy'%(outname)) 
except:
    td=dodown(np.load(datapath+'TT857_%d.npy'%(Default_nside)),nout)
    di=dodown(np.load(datapath+'%s_MONO.npy'%(sys.argv[1])),nout)
    d1=dodown(np.load(datapath+'%s_HM1_MONO.npy'%(sys.argv[1])),nout)
    d2=dodown(np.load(datapath+'%s_HM2_MONO.npy'%(sys.argv[1])),nout)

    np.save(outpath+'/%std.npy'%(outname),td)
    np.save(outpath+'/%sdi.npy'%(outname),di)
    np.save(outpath+'/%sd1.npy'%(outname),d1)
    np.save(outpath+'/%sd2.npy'%(outname),d2)
    
    td=np.load(outpath+'/%std.npy'%(outname)) 
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

    
off=np.median(di[di>-1E10])
d1[di<-1E10]=off
d2[di<-1E10]=off
di[di<-1E10]=off

#=============================================

# compute amplitude to normalize the dynamic range
ampmap=1/dodown(np.load(scratch_path+'%s_NOISE%03d_full.npy'%(sys.argv[1][0:6],0)).flatten(),nout).std()

# rescale maps to ease the convergence
d1=ampmap*(d1-off)
d2=ampmap*(d2-off)
di=ampmap*(di-off)
td=ampmap*(td)

#compute all noise map statistics
noise=np.zeros([nsim,12*nout*nout],dtype='float64')
noise1=np.zeros([nsim,12*nout*nout],dtype='float64')
noise2=np.zeros([nsim,12*nout*nout],dtype='float64')
for i in range(nsim):
    noise[i] =ampmap*dodown(np.load(scratch_path+'%s_NOISE%03d_full.npy'%(sys.argv[1][0:6],i)).flatten(),nout)
    noise1[i]=ampmap*dodown(np.load(scratch_path+'%s_NOISE%03d_hm1.npy'%(sys.argv[1][0:6],i)).flatten(),nout)
    noise2[i]=ampmap*dodown(np.load(scratch_path+'%s_NOISE%03d_hm2.npy'%(sys.argv[1][0:6],i)).flatten(),nout)

for i in range(nsim):
    noise1[i]-=np.mean(noise[i])
    noise2[i]-=np.mean(noise[i])
    noise[i] -=np.mean(noise[i])


import foscat.scat as sc
import foscat.Synthesis as synthe

scat_op=sc.funct(NORIENT=4,   # define the number of wavelet orientation
                 KERNELSZ=5,  # define the kernel size (here 5x5)
                 OSTEP=-1,     # get very large scale (nside=1)
                 LAMBDA=1.0,
                 all_type='float32',
                 TEMPLATE_PATH=scratch_path)

#compute d1xd2
refH=scat_op.eval(d1,image2=d2,Imaginary=False,mask=mask)

#compute Tdxdi
refX=scat_op.eval(td,image2=di,mask=mask)

def loss_fct1(x,args):

    ref  = args[0]
    mask = args[1]
    isig = args[2]
    
    b=scat_op.eval(x,mask=mask)

    l_val=scat_op.reduce_sum(isig*scat_op.reduce_mean(scat_op.square(ref-b)))
    
    return(l_val)

def loss_fct2(x,args):

    ref  = args[0]
    TT   = args[1]
    mask = args[2]
    isig = args[3]
    
    b=scat_op.eval(TT,image2=x,mask=mask)
    
    l_val=scat_op.reduce_sum(scat_op.reduce_mean(scat_op.square((ref-b))))
    
    return(l_val)

def loss_fct3(x,args):

    im   = args[0]
    bias = args[1]
    refH = args[2]
    mask = args[3]
    isig = args[4]
    
    a=scat_op.eval(im,image2=x,mask=mask,Imaginary=False)-bias
    
    l_val=scat_op.reduce_sum(scat_op.reduce_mean(scat_op.square((a-refH))))
    
    return(l_val)

i1=d1
i2=d2
imap=di
init_map=1*di

for itt in range(1):

    #loss1 : d1xd2 = (u+n1)x(u+n2)
    stat1_p_noise=scat_op.eval(i1+noise1[0],image2=i2+noise2[0],mask=mask,Imaginary=False)
    stat1 =scat_op.eval(i1,image2=i2,mask=mask,Imaginary=False)
    
    #bias1 = mean(F((d1+n1)*(d2+n2))-F(d1*d2))
    bias1 = stat1_p_noise-stat1
    isig1 = scat_op.square(stat1_p_noise-stat1)
    for k in range(1,nsim):
        stat1_p_noise=scat_op.eval(i1+noise1[k],image2=i2+noise2[k],mask=mask,Imaginary=False)
        bias1 = bias1 + stat1_p_noise-stat1
        isig1 = isig1 + scat_op.square(stat1_p_noise-stat1)

    bias1=bias1/nsim
    isig1=nsim/isig1
    
    #loss2 : Txd = Tx(u+n)
    #bias2 = mean(F((T*(d+n))-F(T*d))
    stat2_p_noise=scat_op.eval(td,image2=imap+noise[0],mask=mask,Imaginary=True)
    stat2 =scat_op.eval(td,image2=imap,mask=mask,Imaginary=True)
    
    bias2 = stat2_p_noise-stat2
    isig2 = scat_op.square(stat2_p_noise-stat2)
    for k in range(1,nsim):
        stat2_p_noise=scat_op.eval(td,image2=imap+noise[k],mask=mask,Imaginary=True)
        bias2 = bias2 + stat2_p_noise-stat2
        isig2 = isig2 + scat_op.square(stat2_p_noise-stat2)

    bias2=bias2/nsim
    isig2=nsim/isig2

    #loss3 : dxu = (u+n)xu
    stat3_p_noise=scat_op.eval(di,image2=imap+noise[0],mask=mask,Imaginary=False)
    stat3 =scat_op.eval(di,image2=imap,mask=mask,Imaginary=False)
    
    bias3 = stat3_p_noise-stat3
    isig3 = scat_op.square(stat3_p_noise-stat3)
    for k in range(1,nsim):
        stat3_p_noise=scat_op.eval(di,image2=imap+noise[k],mask=mask,Imaginary=False)
        bias3 = bias3 + stat3_p_noise-stat3
        isig3 = isig3 + scat_op.square(stat3_p_noise-stat3)

    bias3=bias3/nsim
    isig3=nsim/isig3

    loss1=synthe.Loss(loss_fct1,refH-bias1,mask,isig1)
    loss2=synthe.Loss(loss_fct2,refX-bias2,td,mask,isig2)
    loss3=synthe.Loss(loss_fct3,di,bias3,refH-bias1,mask,isig3)
    
    sy = synthe.Synthesis([loss1,loss2,loss3])

    omap=sy.run(init_map,
                EVAL_FREQUENCY = 10,
                DECAY_RATE=0.9998,
                NUM_EPOCHS = 1000,
                LEARNING_RATE = 0.03,
                EPSILON = 1E-16)

    i1=omap
    i2=omap
    imap=omap

    # save the intermediate results
    print('ITT %d DONE'%(itt))
    sys.stdout.flush()
    np.save(outpath+'%sresult_%d.npy'%(outname,itt),omap/ampmap+off)

print('Computation Done')
sys.stdout.flush()

