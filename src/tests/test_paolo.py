import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import foscat.scat_cov as sc
import foscat.Synthesis as synthe

nside = 4

nstepmap = 4
ndata = 10
KERNELSZ = 5

NORIENT = 4

nstep = 1000

np.random.seed(1234)
im=np.random.randn(ndata,2,12*nside**2)

heal_im=np.zeros([ndata,2,12*nside**2])

for k in range(ndata):
    for l in range(2):
        heal_im[k,l]=hp.reorder(hp.ud_grade(im[k,l],nside),r2n=True)
        
heal_im/=heal_im.std()

mean_data=np.mean(heal_im,0)
std_data=np.std(heal_im,0)
del im

bk_tab=['tensorflow','torch']

for BACKEND in bk_tab:
    print("==============================================================")
    print("\n\n TEST ",BACKEND,"\n\n")
    print("==============================================================")
    
    scat_op=sc.funct(BACKEND=BACKEND,
                 NORIENT=NORIENT,          # define the number of wavelet orientation
                 KERNELSZ=KERNELSZ,         #KERNELSZ,  # define the kernel size
                 )
                 
    import foscat.alm as foscat_alm

    alm=foscat_alm.alm(nside=nside,backend=scat_op)


    def dospec(im):
        return alm.anafast(im,nest=True)
        
    im=np.random.randn(2,12*nside**2)
    cl2,cl1=dospec(im)
    cl2=scat_op.backend.to_numpy(cl2)
    
    print('mean_cl2 ',abs(np.mean(cl2[2])))
    assert np.mean(cl2[2])<0.01
    
    norm_noise = (heal_im - np.expand_dims(mean_data,0)) / np.expand_dims(std_data,0)

    cmat_0,cmat2_0=scat_op.stat_cfft(norm_noise[0],upscale=KERNELSZ==5,smooth_scale=1)
    cmat_1,cmat2_1=scat_op.stat_cfft(norm_noise[1],upscale=KERNELSZ==5,smooth_scale=1)

    mask=np.ones([1,12*nside**2])
    print(nside)
    ref1={}
    ref2={}
    c_l1=np.zeros([heal_im.shape[0],3,3*nside])
    c_l2=np.zeros([heal_im.shape[0],3,3*nside])
    for k in range(heal_im.shape[0]):
        ref1[k] = scat_op.eval((heal_im[k,0]-mean_data[0])/std_data[0], norm='self',cmat=cmat_0,cmat2=cmat2_0)
        ref2[k] = scat_op.eval((heal_im[k,1]-mean_data[1])/std_data[1], norm='self',cmat=cmat_1,cmat2=cmat2_1)
        tp_l2,tp_l1=dospec(heal_im[k])
        c_l1[k]=scat_op.backend.to_numpy(tp_l1)
        c_l2[k]=scat_op.backend.to_numpy(tp_l2)

    mref1,vref1=scat_op.moments(ref1)
    mref2,vref2=scat_op.moments(ref2)

    r_c_l1=np.mean(c_l1,0)
    r_c_l2=np.mean(c_l2,0)

    d_c_l1=np.std(c_l1,0)
    d_c_l2=np.std(c_l2,0)
    d_c_l1[:,0:2]=1.0
    d_c_l2[:,0:2]=1.0
    
    def The_loss_spec(x, scat_operator, args):
        mean_val = args[0]
        std_val  = args[1]
        r_c_l1 = args[2]
        r_c_l2 = args[3]
        d_c_l1 = args[4]
        d_c_l2 = args[5]
        alm    = args[6]

        tp_c_l2,tp_c_l1=dospec(x[0]*std_val+mean_val)
        c_l1=tp_c_l1-r_c_l1
        c_l2=tp_c_l2-r_c_l2
        
        for k in range(1,x.shape[0]):
            tp_c_l2,tp_c_l1=dospec(x[k]*std_val+mean_val)
            c_l1=c_l1+tp_c_l1-r_c_l1
            c_l2=c_l2+tp_c_l2-r_c_l2
            
        loss = scat_operator.backend.bk_reduce_mean(scat_operator.backend.bk_square(c_l1/d_c_l1))+ \
            scat_operator.backend.bk_reduce_mean(scat_operator.backend.bk_square(c_l2/d_c_l2))
        return loss
        
    def The_loss(x, scat_operator, args):
        ref1  = args[0]
        sref1 = args[1]
        ref2  = args[2]
        sref2 = args[3]
        cmat11= args[4]
        cmat12= args[5]
        cmat21= args[6]
        cmat22= args[7]
                                                                                               
        # compute scattering covariance of the current synthetised maps called
        learn = scat_operator.eval(x[:,0], norm='self',cmat=cmat11,cmat2=cmat12)
        # Reduce the coordinate to the sum to keep the proper weight on the chi2
        learn = scat_operator.reduce_sum_batch(learn)
        loss = scat_operator.reduce_mean(scat_operator.square((learn-x.shape[0]*ref1)/sref1))
        
        learn = scat_operator.eval(x[:,1], norm='self',cmat=cmat11,cmat2=cmat12)
        learn = scat_operator.reduce_sum_batch(learn)
        loss = loss+scat_operator.reduce_mean(scat_operator.square((learn-x.shape[0]*ref2)/sref2))

        return loss*10
        
    loss = synthe.Loss(The_loss,scat_op,
                   mref1,vref1,
                   mref2,vref2,
                   cmat_0,cmat2_0,
                   cmat_1,cmat2_1)

    loss_sp = synthe.Loss(The_loss_spec,scat_op,
                    scat_op.backend.bk_cast(mean_data),scat_op.backend.bk_cast(std_data),
                    scat_op.backend.bk_cast(r_c_l1), scat_op.backend.bk_cast(r_c_l2),
                    scat_op.backend.bk_cast(d_c_l1), scat_op.backend.bk_cast(d_c_l2),alm)

    sy = synthe.Synthesis([loss,loss_sp])
    
    np.random.seed(7)
    omap=np.random.randn(nstepmap,2,12*nside**2)*np.std(heal_im)

    omap=scat_op.backend.to_numpy(sy.run(omap,
                EVAL_FREQUENCY=10,
                NUM_EPOCHS = 100))
                
    assert np.min(sy.get_history())<1.0
                
    # Plot scalar power spectrum of the Q map
    lmax=3*nside
    ps = np.zeros((omap.shape[0],6, lmax))
    rps= np.zeros((heal_im.shape[0],6, lmax))
    map=np.zeros([3,12*nside**2])
    for k in range(0, omap.shape[0]):
        for p in range(2):
            map[1+p] = hp.reorder(omap[k,p]*std_data[p]+mean_data[p], n2r=True)
        ps[k] = hp.anafast(map)
    for k in range(0, heal_im.shape[0]):
        for p in range(2):
            map[1+p] = hp.reorder(heal_im[k,p], n2r=True)
        rps[k]= hp.anafast(map)
    mean_ps = np.mean(ps, axis=0)
    mean_rps = np.mean(rps, axis=0)
    
    print(np.mean((mean_ps-mean_rps)**2))
    
    assert  np.mean((mean_ps-mean_rps)**2)<1E-5