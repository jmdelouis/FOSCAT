import numpy as np
import os, sys
import matplotlib.pyplot as plt
import healpy as hp
import getopt

#=================================================================================
# INITIALIZE FoCUS class
#=================================================================================
import foscat.Synthesis as synthe

def usage():
    print(' This software is a demo of the foscat library:')
    print('>python demo.py -n=8 [-c|--cov][-s|--steps=3000][-S=1234|--seed=1234][-x|--xstat][-p|--p00][-g|--gauss][-k|--k5x5][-d|--data][-o|--out][-K|--k128][-r|--orient]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov (optional): use scat_cov instead of scat.')
    print('--steps (optional): number of iteration, if not specified 1000.')
    print('--seed  (optional): seed of the random generator.')
    print('--xstat (optional): work with cross statistics.')
    print('--p00   (optional): Loss only computed on p00.')
    print('--gauss (optional): convert Venus map in gaussian field.')
    print('--k5x5  (optional): Work with a 5x5 kernel instead of a 3x3.')
    print('--k128  (optional): Work with 128 pixel kernel reproducing wignercomputation instead of a 3x3.')
    print('--data  (optional): If not specified use Venu_256.npy.')
    print('--out   (optional): If not specified save in *_demo_*.')
    print('--orient(optional): If not specified use 4 orientation')
    exit(0)
    
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:cS:s:xpgkd:o:Kr:", \
                                   ["nside", "cov","seed","steps","xstat","p00","gauss","k5x5","data","out","k128","orient"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    nstep=1000
    docross=False
    dop00=False
    dogauss=False
    KERNELSZ=3
    dok128=False
    seed=1234
    outname='demo'
    data="Venus_256.npy"
    instep=16
    norient=4
    
    for o, a in opts:
        if o in ("-c","--cov"):
            cov = True
        elif o in ("-n", "--nside"):
            nside=int(a[1:])
        elif o in ("-s", "--steps"):
            nstep=int(a[1:])
        elif o in ("-S", "--seed"):
            seed=int(a[1:])
            print('Use SEED = ',seed)
        elif o in ("-o", "--out"):
            outname=a[1:]
            print('Save data in ',outname)
        elif o in ("-d", "--data"):
            data=a[1:]
            print('Read data from ',data)
        elif o in ("-x", "--xstat"):
            docross=True
        elif o in ("-g", "--gauss"):
            dogauss=True
        elif o in ("-k", "--k5x5"):
            KERNELSZ=5
        elif o in ("-K", "--k64"):
            KERNELSZ=64
            instep=8
        elif o in ("-r", "--orient"):
            norient=int(a[1:])
            print('Use %d orientations'%(norient))
        elif o in ("-p", "--p00"):
            dop00=True
        else:
            assert False, "unhandled option"

    if nside<2 or nside!=2**(int(np.log(nside)/np.log(2))) or (nside>256 and KERNELSZ<=5) or (nside>2**instep and KERNELSZ>5) :
        print('nside should be a power of 2 and in [2,...,256] or [2,...,%d] if -K|-k128 option has been choosen'%(2**instep))
        usage()
        exit(0)

    print('Work with nside=%d'%(nside))

    if cov:
        import foscat.scat_cov as sc
        print('Work with ScatCov')
    else:
        import foscat.scat as sc
        print('Work with Scat')
        
    #=================================================================================
    # DEFINE A PATH FOR scratch data
    # The data are storred using a default nside to minimize the needed storage
    #=================================================================================
    scratch_path = '../data'

    #=================================================================================
    # Function to reduce the data used in the FoCUS algorithm 
    #=================================================================================
    def dodown(a,nside):
        nin=int(np.sqrt(a.shape[0]//12))
        if nin==nside:
            return(a)
        return(np.mean(a.reshape(12*nside*nside,(nin//nside)**2),1))

    #=================================================================================
    # Get data
    #=================================================================================
    im=dodown(np.load(data),nside)
    mask=np.ones([1,im.shape[0]])
    mask[0,:]=(im!=hp.UNSEEN)
    im[im==hp.UNSEEN]=0.0

    #=================================================================================
    # Generate a random noise with the same coloured than the input data
    #=================================================================================

    idx=hp.ring2nest(nside,np.arange(12*nside*nside))
    idx1=hp.nest2ring(nside,np.arange(12*nside*nside))
    cl=hp.anafast(im[idx])
    
    if dogauss:
        np.random.seed(seed+1)
        im=hp.synfast(cl,nside)[idx1]
        hp.mollview(im,cmap='jet',nest=True)
        plt.show()
        
    np.random.seed(seed)
    imap=hp.synfast(cl,nside)[idx1]

    lam=1.2
    if KERNELSZ==5:
        lam=1.0

    l_slope=1.0
    r_format=True
    if KERNELSZ==64:
        r_format=False
        l_slope=4.0
    #=================================================================================
    # COMPUTE THE WAVELET TRANSFORM OF THE REFERENCE MAP
    #=================================================================================
    scat_op=sc.funct(NORIENT=4,          # define the number of wavelet orientation
                     KERNELSZ=KERNELSZ,  #KERNELSZ,  # define the kernel size
                     OSTEP=-1,           # get very large scale (nside=1)
                     LAMBDA=lam,
                     TEMPLATE_PATH=scratch_path,
                     slope=l_slope,
                     gpupos=2,
                     use_R_format=r_format,
                     all_type='float32',
                     nstep_max=instep)
    
    #=================================================================================
    # DEFINE A LOSS FUNCTION AND THE SYNTHESIS
    #=================================================================================
    
    def lossX(x,scat_operator,args):
        
        ref = args[0]
        im  = args[1]
        mask = args[2]

        if docross:
            learn=scat_operator.eval(im,image2=x,Imaginary=True,mask=mask)
        else:
            learn=scat_operator.eval(x,mask=mask)
            
        if dop00:
            loss=scat_operator.bk_reduce_mean(scat_operator.bk_square(ref.P00[0,0,:]-learn.P00[0,0,:]))
        else:
            loss=scat_operator.reduce_sum(scat_operator.square(ref-learn))
            

        return(loss)

    if docross:
        refX=scat_op.eval(im,image2=im,Imaginary=True,mask=mask)
    else:
        refX=scat_op.eval(im,mask=mask)
    
    loss1=synthe.Loss(lossX,scat_op,refX,im,mask)
        
    sy = synthe.Synthesis([loss1])
    #=================================================================================
    # RUN ON SYNTHESIS
    #=================================================================================

    omap=sy.run(imap,
                DECAY_RATE=0.9995,
                NUM_EPOCHS = nstep,
                LEARNING_RATE = 0.03,
                EPSILON = 1E-15,
                SHOWGPU=True)

    #=================================================================================
    # STORE RESULTS
    #=================================================================================
    if docross:
        start=scat_op.eval(im,image2=imap,mask=mask)
        out =scat_op.eval(im,image2=omap,mask=mask)
    else:
        start=scat_op.eval(imap,mask=mask)
        out =scat_op.eval(omap,mask=mask)
    
    np.save('in_%s_map_%d.npy'%(outname,nside),im)
    np.save('mm_%s_map_%d.npy'%(outname,nside),mask[0])
    np.save('st_%s_map_%d.npy'%(outname,nside),imap)
    np.save('out_%s_map_%d.npy'%(outname,nside),omap)
    np.save('out_%s_log_%d.npy'%(outname,nside),sy.get_history())

    refX.save('in_%s_%d'%(outname,nside))
    start.save('st_%s_%d'%(outname,nside))
    out.save('out_%s_%d'%(outname,nside))

    print('Computation Done')
    sys.stdout.flush()

if __name__ == "__main__":
    main()


    
