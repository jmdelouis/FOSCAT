import numpy as np
import os, sys
import healpy as hp
import matplotlib.pyplot as plt
import getopt

def usage():
    print(' This software plots the demo results:')
    print('>python plotdemo.py -n=8 [-c|cov] [-o|--out] [-c|--cmap] [-g|--geo]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov     (optional): use scat_cov instead of scat')
    print('--out     (optional): If not specified save in *_demo_*.')
    print('--map=jet (optional): If not specified use cmap=jet')
    print('--geo     (optional): If specified use cartview')
    exit(0)
    
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:co:m:g", ["nside", "cov","out","map","geo"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    outname='demo'
    cmap='jet'
    docart=False
    
    for o, a in opts:
        if o in ("-c","--cov"):
            cov = True
        elif o in ("-g","--geo"):
            docart=True
        elif o in ("-m","--map"):
            cmap=a[1:]
        elif o in ("-n", "--nside"):
            nside=int(a[1:])
        elif o in ("-o", "--out"):
            outname=a[1:]
        else:
            print(o,a)
            assert False, "unhandled option"

    if nside<2 or nside!=2**(int(np.log(nside)/np.log(2))) or nside>256:
        print('nside should be a pwer of 2 and in [2,...,256]')
        exit(0)

    print('Work with nside=%d'%(nside))

    if cov:
        import foscat.scat_cov as sc
    else:
        import foscat.scat as sc

    refX  = sc.read('in_%s_%d'%(outname,nside))
    start = sc.read('st_%s_%d'%(outname,nside))
    out   = sc.read('out_%s_%d'%(outname,nside))

    log= np.load('out_%s_log_%d.npy'%(outname,nside))
    plt.figure(figsize=(6,6))
    plt.plot(np.arange(log.shape[0])+1,log,color='black')
    plt.xscale('log')
    plt.yscale('log')

    refX.plot(name='Model',lw=6)
    start.plot(name='Input',color='orange',hold=False)
    out.plot(name='Output',color='red',hold=False)
    (refX-out).plot(name='Diff',color='purple',hold=False)

    im = np.load('in_%s_map_%d.npy'%(outname,nside))
    sm = np.load('st_%s_map_%d.npy'%(outname,nside))
    om = np.load('out_%s_map_%d.npy'%(outname,nside))

    idx=hp.ring2nest(nside,np.arange(12*nside**2))
    plt.figure(figsize=(10,6))
    if docart:
        hp.cartview(im[idx],cmap=cmap,min=-3,max=3,hold=False,sub=(2,2,1),nest=False,title='Model')
        hp.cartview(sm,cmap=cmap,min=-3,max=3,hold=False,sub=(2,2,2),nest=True,title='Start')
        hp.cartview(om,cmap=cmap,min=-3,max=3,hold=False,sub=(2,2,3),nest=True,title='Output')
        hp.cartview(im-om,cmap=cmap,min=-3,max=3,hold=False,sub=(2,2,4),nest=True,title='Diff')
    else:
        hp.mollview(im[idx],cmap=cmap,min=-3,max=3,hold=False,sub=(2,2,1),nest=False,title='Model')
        hp.mollview(sm,cmap=cmap,min=-3,max=3,hold=False,sub=(2,2,2),nest=True,title='Start')
        hp.mollview(om,cmap=cmap,min=-3,max=3,hold=False,sub=(2,2,3),nest=True,title='Output')
        hp.mollview(im-om,cmap=cmap,min=-3,max=3,hold=False,sub=(2,2,4),nest=True,title='Diff')

    cli=hp.anafast((im-np.median(im))[idx])
    clo=hp.anafast((om-np.median(om))[idx])
    cldiff=hp.anafast((im-om-np.median(om))[idx])

    plt.figure(figsize=(6,6))
    plt.plot(cli,color='blue',label=r'Model',lw=6)
    plt.plot(clo,color='orange',label=r'Output')
    plt.plot(cldiff,color='red',label=r'Diff')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.xlabel('Multipoles')
    plt.ylabel('C(l)')
    plt.show()

if __name__ == "__main__":
    main()
