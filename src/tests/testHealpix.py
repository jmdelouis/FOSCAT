import numpy as np
import os, sys
import matplotlib.pyplot as plt
import healpy as hp
import foscat as FOC

#=================================================================================
# DEFINE A PATH FOR scratch data
# The data are storred using a default nside to minimize the needed storage
#=================================================================================
scratch_path = 'foscat_data'
outname='TEST'
Default_nside=512

#=================================================================================
# INITIALIZE foscat class
#=================================================================================
fc=FOC.foscat(NORIENT=4,
             KERNELSZ=3,
             healpix=True,
             OSTEP=0,
             slope=1.0,
             TEMPLATE_PATH=scratch_path)


#=================================================================================
# DENOISE NSIDE=32 MAP
#=================================================================================
nout=32
#=================================================================================
# Function to reduce the data used in the foscat algorithm 
#=================================================================================
def dodown(a,nout):
    nin=int(np.sqrt(a.shape[0]//12))
    if nin==nout:
        return(a)
    return(np.mean(a.reshape(12*nout*nout,(nin//nout)**2),1))

#=================================================================================
# Get data from web for demo
#=================================================================================

#=================================================================================
# Get dust simulated map from PySM2
#=================================================================================
try:
    d=dodown(np.load(scratch_path+'/dust_512.npy'),nout)
except:
    os.system('wget -O '+scratch_path+'/dust2comp_I1_ns512_545.fits https://portal.nersc.gov/project/cmb/pysm-data/pysm_2/dust2comp_I1_ns512_545.fits')

    print('wget -O '+scratch_path+'/dust2comp_I1_ns512_545.fits https://portal.nersc.gov/project/cmb/pysm-data/pysm_2/dust2comp_I1_ns512_545.fits')
    im=hp.ud_grade(hp.read_map(scratch_path+'/dust2comp_I1_ns512_545.fits'),512)
    idx=hp.nest2ring(512,np.arange(12*512**2))
    im=10*im/im.std()
    np.save(scratch_path+'/dust_512.npy',im[idx])
    os.system('rm '+scratch_path+'/dust2comp_I1_ns512_545.fits')
    d=dodown(np.load(scratch_path+'/dust_512.npy'),nout)

#=================================================================================
# Get HI experimetal map
#=================================================================================
try:
    td=dodown(np.load(scratch_path+'/TH1_512.npy'),nout)
except:
    os.system('wget -O '+scratch_path+'/haslam408_dsds_Remazeilles2014.fits "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=haslam408_dsds_Remazeilles2014.fits"')
    h1=hp.ud_grade(hp.read_map(scratch_path+'/haslam408_dsds_Remazeilles2014.fits'),512)
    idx=hp.nest2ring(512,np.arange(12*512**2))
    h1=h1/h1.std()
    np.save(scratch_path+'/TH1_512.npy',h1[idx])
    os.system('rm '+scratch_path+'/haslam408_dsds_Remazeilles2014.fits')
    td=dodown(np.load(scratch_path+'/TH1_512.npy'),nout)

# define the level of noise of the simulation
ampnoise=0.3
Alpha=0.8
nsim=100
avg_ang=False

# Build to half mission by adding noise to the data on the first process
d1 = d+ampnoise*np.random.randn(12*nout**2)
d2 = d+ampnoise*np.random.randn(12*nout**2)
# simulate the noisy data
di = d+ampnoise*np.random.randn(12*nout**2)/np.sqrt(2)
np.save(scratch_path+'/test%sref.npy'%(outname), d)
np.save(scratch_path+'/test%sinput.npy'%(outname),di)
np.save(scratch_path+'/test%sinput1.npy'%(outname),d1)
np.save(scratch_path+'/test%sinput2.npy'%(outname),d2)

#=================================================================================
# For real data:
# you have to give d1,d2 and d
# you have to define the value ampnoise
#=================================================================================


# All information of the map is used
nin=512

tab=['MASK_GAL097_%d.npy'%(nin),
     'MASK_GAL090_%d.npy'%(nin),
     'MASK_GAL080_%d.npy'%(nin),
     'MASK_GAL060_%d.npy'%(nin),
     'MASK_GAL040_%d.npy'%(nin)]

mask=np.ones([len(tab),12*nout**2])
for imask in range(len(tab)):
    try:
        mask[imask]=dodown(np.load(scratch_path+'/'+tab[imask]),nout)
    except:
        print("==========================================================")
        print("")
        print("Get offical Planck Galactic masks...may take few minutes..")
        print("")
        print("==========================================================")

        os.system('wget -O '+scratch_path+'/HFI_Mask_GalPlane-apo5_2048_R2.00.fits https://irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/masks/HFI_Mask_GalPlane-apo5_2048_R2.00.fits')
        tab_data=['GAL020','GAL040','GAL060','GAL070','GAL080','GAL090','GAL097','GAL099']
        
        idx=hp.nest2ring(512,np.arange(12*512**2))
        for i in range(10):
            mask=hp.ud_grade(hp.read_map(scratch_path+'/HFI_Mask_GalPlane-apo5_2048_R2.00.fits',i),512)
            np.save(scratch_path+'/MASK_%s_512.npy'%(tab_data[i]),mask[idx])
            print('Save '+scratch_path+'/MASK_%s_512.npy'%(tab_data[i]))
        os.system('rm '+scratch_path+'/HFI_Mask_GalPlane-apo5_2048_R2.00.fits')
        mask[imask]=dodown(np.load(scratch_path+'/'+tab[imask]),nout)
        
mask[0,:]=1.0
fc.add_mask(mask)

# convert data in tensor for foscat (should be done internally)
data1 = fc.convimage(d1)
data2 = fc.convimage(d2)
data  = fc.convimage(di)
tdata = fc.convimage(td)

# Initialize the learning and initialize the tensor to be synthesized
ldata=fc.init_synthese(di)

# Add losss:
# here d1 x d2 = s x s
fc.add_loss_healpix(data1,data2,ldata,ldata,avg_ang=avg_ang)
# here d1 x d2 = d x s
fc.add_loss_healpix(data,ldata,ldata,ldata,avg_ang=avg_ang)
# here T x d = T x s
fc.add_loss_healpix(tdata,data,tdata,ldata,avg_ang=avg_ang,imaginary=True)

# initiliaze the loss
loss=fc.init_optim()

moddr=d.reshape(1,12*nout**2)
x1,x2=fc.calc_stat(moddr,moddr,avg_ang=avg_ang)
np.save(scratch_path+'/i%s1_%d.npy'%(outname,0), x1)
np.save(scratch_path+'/i%s2_%d.npy'%(outname,0), x2)

c1,c2=fc.calc_stat(d1.reshape(1,12*nout**2),d2.reshape(1,12*nout**2),avg_ang=avg_ang)
np.save(scratch_path+'/c%s1_%d.npy'%(outname,0), c1)
np.save(scratch_path+'/c%s2_%d.npy'%(outname,0), c2)
lmap1=1*d1
lmap2=1*d2

tw1={}
tw2={}
tb1={}
tb2={}
for i in range(3):
    tw1[i]=0.0
    tw2[i]=0.0
    tb1[i]=0.0
    tb2[i]=0.0
    
for itt in range(5):
    # compute scattering to make weights
    modd1=lmap1.reshape(1,12*nout**2)+ampnoise*np.random.randn(nsim,12*nout**2)
    modd2=lmap2.reshape(1,12*nout**2)+ampnoise*np.random.randn(nsim,12*nout**2)
    o1,o2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,gpupos=0)
    modd1=lmap1.reshape(1,12*nout**2)
    modd2=lmap2.reshape(1,12*nout**2)
    r1,r2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,gpupos=0)
    #calcul poids des coeffs
    np.save(scratch_path+'/b%s1_%d.npy'%(outname,itt), r1)
    np.save(scratch_path+'/b%s2_%d.npy'%(outname,itt), r2)
    np.save(scratch_path+'/n%s1_%d.npy'%(outname,itt), o1)
    np.save(scratch_path+'/n%s2_%d.npy'%(outname,itt), o2)
        
    modd1=lmap1.reshape(1,12*nout**2)+ampnoise*np.random.randn(nsim,12*nout**2)
    modd2=lmap2.reshape(1,12*nout**2)+0*np.random.randn(nsim,12*nout**2)
    of1,of2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,gpupos=1)
    
    modd1=td.reshape(1,12*nout**2)+0*np.random.randn(nsim,12*nout**2)
    modd2=((lmap1+lmap2)/2).reshape(1,12*nout**2)+ampnoise*np.random.randn(nsim,12*nout**2)
    onx1,onx2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,imaginary=True,gpupos=2)
    modd1=td.reshape(1,12*nout**2)
    modd2=((lmap1+lmap2)/2).reshape(1,12*nout**2)
    ox1,ox2=fc.calc_stat(modd1,modd2,avg_ang=avg_ang,imaginary=True,gpupos=2)
    
    tw1[0]=1/np.std(o1,0)
    tw2[0]=1/np.std(o2,0)
    tw1[1]=1/np.std(of1,0)
    tw2[1]=1/np.std(of2,0)
    tw1[2]=1/np.std(onx1,0)
    tw2[2]=1/np.std(onx2,0)
    
    tb1[0]=Alpha*(np.mean(o1-r1,0)-tb1[0])+tb1[0]
    tb2[0]=Alpha*(np.mean(o2-r2,0)-tb2[0])+tb2[0]
    tb1[1]=Alpha*(np.mean(of1-r1,0)-tb1[1])+tb1[1]
    tb2[1]=Alpha*(np.mean(of2-r2,0)-tb2[1])+tb2[1]
    tb1[2]=Alpha*(np.mean(onx1-ox1,0)-tb1[2])+tb1[2]
    tb2[2]=Alpha*(np.mean(onx2-ox2,0)-tb2[2])+tb2[2]
    
    # make the learn
    fc.reset()
    
    omap=fc.learn(tw1,tw2,tb1,tb2,NUM_EPOCHS = 500,DECAY_RATE=0.95,LEARNING_RATE=0.03)

    print('ITT ',itt,((d-omap)*mask[1].reshape(12*nout**2)).std(),((d-di)*mask[1].reshape(12*nout**2)).std())
    sys.stdout.flush()
    modd1=omap.reshape(1,12*nout**2)
    oo1,oo2=fc.calc_stat(modd1,modd1,avg_ang=avg_ang)
    lmap1=1*omap
    lmap2=1*omap
    np.save(scratch_path+'/o%s1_%d.npy'%(outname,itt), oo1)
    np.save(scratch_path+'/o%s2_%d.npy'%(outname,itt), oo2)
    np.save(scratch_path+'/test%sresult_%d.npy'%(outname,itt),omap)


