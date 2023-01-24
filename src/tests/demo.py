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
    print('>python demo.py -n=8 [-c|--cov][-s|--steps=3000]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov (optional): use scat_cov instead of scat')
    print('--steps (optional): number of iteration, if not specified 1000')
    exit(0)
    
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:cs:", ["nside", "cov","steps"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    nstep=1000
    for o, a in opts:
        if o in ("-c","--cov"):
            cov = True
        elif o in ("-n", "--nside"):
            nside=int(a[1:])
        elif o in ("-s", "--steps"):
            nstep=int(a[1:])
        else:
            assert False, "unhandled option"

    if nside<2 or nside!=2**(int(np.log(nside)/np.log(2))) or nside>256:
        print('nside should be a power of 2 and in [2,...,256]')
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
    outname='TEST_EE'

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
    im=dodown(np.load('Venus_256.npy'),nside)

    #=================================================================================
    # COMPUTE THE WAVELET TRANSFORM OF THE REFERENCE MAP
    #=================================================================================
    scat_op=sc.funct(NORIENT=4,   # define the number of wavelet orientation
                     KERNELSZ=3,  # define the kernel size (here 5x5)
                     OSTEP=-1,     # get very large scale (nside=1)
                     LAMBDA=1.0,
                     TEMPLATE_PATH=scratch_path,
                     all_type='float32')

    refX=scat_op.eval(im)

    #=================================================================================
    # DEFINE A LOSS FUNCTION AND THE SYNTHESIS
    #=================================================================================

    def lossX(x,args):

        ref = args[0]
        im  = args[1]

        b=scat_op.eval(x)

        loss=scat_op.reduce_sum(scat_op.square((ref-b)))

        return(loss)

    loss1=synthe.Loss(lossX,refX,im.astype('float32'))

    sy = synthe.Synthesis([loss1])

    #=================================================================================
    # RUN ON SYNTHESIS
    #=================================================================================

    imap=np.random.randn(12*nside*nside).astype('float32')

    omap=sy.run(imap,
                DECAY_RATE=0.995,
                NUM_EPOCHS = nstep,
                LEARNING_RATE = 0.03,
                EPSILON = 1E-16)

    #=================================================================================
    # STORE RESULTS
    #=================================================================================
    start=scat_op.eval(im.astype('float32'),image2=imap.astype('float32'))
    out =scat_op.eval(im.astype('float32'),image2=omap.numpy().astype('float32'))

    np.save('in_demo_map_%d.npy'%(nside),im)
    np.save('st_demo_map_%d.npy'%(nside),imap)
    np.save('out_demo_map_%d.npy'%(nside),omap)
    np.save('out_demo_log_%d.npy'%(nside),sy.get_history())

    refX.save('in_demo_%d'%(nside))
    start.save('st_demo_%d'%(nside))
    out.save('out_demo_%d'%(nside))

    print('Computation Done')
    sys.stdout.flush()

if __name__ == "__main__":
    main()


    
