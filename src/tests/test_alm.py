import numpy as np
import healpy as hp
import foscat.alm as foscat_alm
import foscat.scat_cov as sc1

bk_tab=['tensorflow','torch']
nside_tab=[1,2,32]

#============================================================================================
#
#    N_image=1
#
#============================================================================================

for BACKEND in bk_tab:
    for nside in nside_tab:
        
        alm1=foscat_alm.alm(nside=nside,backend=sc1.funct(BACKEND=BACKEND))

        im=np.random.randn(3,12*nside**2)

        cl1,cl1_l1=alm1.anafast(im)
        cl1=alm1.backend.to_numpy(cl1)
        cl3=hp.anafast(im,iter=0)
        
        print('TT,EE,BB,TE,EB,TB',nside,BACKEND)
        
        for k in range(6):
            print('foscat ml',(cl1[k]-cl3[k]).std())
            assert  (cl1[k]-cl3[k]).std()<1E-6/nside

        cl1,cl1_l1=alm1.anafast(im[1:])
        cl1=alm1.backend.to_numpy(cl1)
        cl3=hp.anafast(im,iter=0)
        
        print('EE,BB,EB',nside,BACKEND)
        
        print('foscat ml',(cl1[0]-cl3[1]).std())
        print('foscat ml',(cl1[1]-cl3[2]).std())
        print('foscat ml',(cl1[2]-cl3[4]).std())
        assert  (cl1[0]-cl3[1]).std()<1E-6/nside
        assert  (cl1[1]-cl3[2]).std()<1E-6/nside
        assert  (cl1[2]-cl3[4]).std()<1E-6/nside
        
        cl1,cl1_l1=alm1.anafast(im[0])
        cl1=alm1.backend.to_numpy(cl1)
        cl3=hp.anafast(im[0],iter=0)
        
        print('TT',nside,BACKEND)
        
        print('foscat ml',(cl1-cl3).std())
        assert  (cl1-cl3).std()<1E-6/nside
        
#============================================================================================
#
#    N_image=3
#
#============================================================================================

for BACKEND in bk_tab:
    for nside in nside_tab:
        
        alm1=foscat_alm.alm(nside=nside,backend=sc1.funct(BACKEND=BACKEND))

        im=np.random.randn(3,3,12*nside**2)

        cl1,cl1_l1=alm1.anafast(im,axes=1)
        cl1=alm1.backend.to_numpy(cl1)
        cl3=0*cl1
        print('TT,EE,BB,TE,EB,TB',nside,BACKEND)
        for l in range(3):
            cl3=hp.anafast(im[l],iter=0)
            for k in range(6):
                print('foscat ml',l,(cl1[l,k]-cl3[k]).std())
                assert  (cl1[l,k]-cl3[k]).std()<1E-6/nside

        cl1,cl1_l1=alm1.anafast(im[:,1:],axes=1)
        cl1=alm1.backend.to_numpy(cl1)
        
        print('EE,BB,EB',nside,BACKEND)
        for l in range(3):
            cl3=hp.anafast(im[l],iter=0)
        
            print('foscat ml',l,(cl1[l,0]-cl3[1]).std())
            print('foscat ml',l,(cl1[l,1]-cl3[2]).std())
            print('foscat ml',l,(cl1[l,2]-cl3[4]).std())
            assert  (cl1[l,0]-cl3[1]).std()<1E-6/nside
            assert  (cl1[l,1]-cl3[2]).std()<1E-6/nside
            assert  (cl1[l,2]-cl3[4]).std()<1E-6/nside
        
        cl1,cl1_l1=alm1.anafast(im[:,0],axes=1)
        cl1=alm1.backend.to_numpy(cl1)
        print('TT',nside,BACKEND)
        for l in range(3):
            cl3=hp.anafast(im[l,0],iter=0)
        
            print('foscat ml',l,(cl1[l]-cl3).std())
            assert  (cl1[l]-cl3).std()<1E-6/nside