import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os, sys
import time

class FoCUS:
    def __init__(self,
                 NORIENT=4,
                 LAMBDA=1.2,
                 KERNELSZ=3,
                 slope=1.0,
                 all_type='float64',
                 padding='SAME',
                 gpupos=0,
                 healpix=True,
                 OSTEP=0,
                 isMPI=False,
                 TEMPLATE_PATH='data'):

        print('================================================')
        print('          START FOSCAT CONFIGURATION')
        print('================================================')

        self.TEMPLATE_PATH=TEMPLATE_PATH
        self.tf=tf
        if os.path.exists(self.TEMPLATE_PATH)==False:
            print('The directory %s to store temporary information for FoCUS does not exist: Try to create it'%(self.TEMPLATE_PATH))
            try:
                os.system('mkdir -p %s'%(self.TEMPLATE_PATH))
                print('The directory %s is created')
            except:
                print('Impossible to create the directory %s'%(self.TEMPLATE_PATH))
                exit(0)
                
        self.nloss=0
        self.inpar={}
        self.rewind={}
        self.diff_map1={}
        self.diff_map2={}
        self.diff_mask={}
        self.diff_weight={}
        self.loss_type={}
        self.loss_weight={}
        
        self.MAPDIFF=1
        self.SCATDIFF=2
        self.NOISESTAT=3
        self.SCATCOV = 4

        self.log=np.zeros([10])
        self.nlog=0
        
        self.padding=padding
        self.healpix=healpix
        self.OSTEP=OSTEP
        self.nparam=0
        self.on1={}
        self.on2={}
        self.tmpa={}
        self.tmpb={}
        self.tmpc={}
        
        if isMPI:
            from mpi4py import MPI

            self.comm = MPI.COMM_WORLD
            self.size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
            
            if all_type=='float32':
                self.MPI_ALL_TYPE=MPI.FLOAT
            else:
                self.MPI_ALL_TYPE=MPI.DOUBLE
        else:
            self.size = 1
            self.rank = 0
        self.isMPI=isMPI
        
        self.tw1={}
        self.tw2={}
        self.tw3={}
        self.tw4={}
        self.tb1={}
        self.tb2={}
        self.tb3={}
        self.tb4={}

        self.ss1={}
        self.ss2={}
        self.ss3={}
        self.ss4={}
        
        self.os1 = {}
        self.os2 = {}
        self.os3 = {}
        self.os4 = {}
        self.is1 = {}
        self.is2 = {}
        self.is3 = {}
        self.is4 = {}
        
        self.NMASK=1
        self.mask={}
        self.all_type=all_type
        if all_type=='float32':
            self.all_tf_type=tf.float32
            #self.MPI_ALL_TYPE=MPI.FLOAT
        else:
            if all_type=='float64':
                self.all_type='float64'
                self.all_tf_type=tf.float64
                #self.MPI_ALL_TYPE=MPI.DOUBLE
            else:
                print('ERROR INIT FOCUS ',all_type,' should be float32 or float64')
                exit(0)
                
        #===========================================================================
        # INIT 
        if self.rank==0:
            print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
            sys.stdout.flush()
        tf.debugging.set_log_device_placement(False)
        tf.config.set_soft_device_placement(True)
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        gpuname='CPU:0'
        self.gpulist={}
        self.gpulist[0]=gpuname
        self.ngpu=1
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                sys.stdout.flush()
                gpuname=logical_gpus[gpupos].name
                self.gpulist={}
                self.ngpu=len(logical_gpus)
                for i in range(self.ngpu):
                    self.gpulist[i]=logical_gpus[i].name
                    
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
                
        self.gpupos=(gpupos+self.rank)%self.ngpu
        print('============================================================')
        print('==                                                        ==')
        print('==                                                        ==')
        print('==     RUN ON GPU Rank %d : %s                          =='%(self.rank,self.gpulist[self.gpupos%self.ngpu]))
        print('==                                                        ==')
        print('==                                                        ==')
        print('============================================================')
        sys.stdout.flush()
        
        self.NORIENT=NORIENT
        self.LAMBDA=LAMBDA
        self.KERNELSZ=KERNELSZ
        self.slope=slope

        wwc=np.zeros([KERNELSZ**2,NORIENT]).astype(all_type)
        wws=np.zeros([KERNELSZ**2,NORIENT]).astype(all_type)

        x=np.repeat(np.arange(KERNELSZ)-KERNELSZ//2,KERNELSZ).reshape(KERNELSZ,KERNELSZ)
        y=x.T

        for i in range(NORIENT):
            a=i/NORIENT*np.pi
            xx=LAMBDA*(x*np.cos(a)+y*np.sin(a))
            yy=LAMBDA*(x*np.sin(a)-y*np.cos(a))
            ww=np.exp(-0.5*yy**2)
            tmp=np.cos(yy*np.pi)*np.exp(-0.5*(4/float(NORIENT)*xx**2))*ww
            ww[tmp<0]-=tmp.sum()/(np.cos(yy[tmp<0]*np.pi)*np.exp(-0.5*(4/float(NORIENT)*xx[tmp<0]**2))).sum()
            tmp=np.cos(yy*np.pi)*np.exp(-0.5*(4/float(NORIENT)*xx**2))*ww
            tmp-=tmp.mean()
            wwc[:,i]=tmp.flatten()
            tmp=np.sin(yy*np.pi)*np.exp(-0.5*(4/float(NORIENT)*xx**2))*ww
            tmp-=tmp.mean()
            wws[:,i]=tmp.flatten()
            sigma=np.sqrt((wwc[:,i]**2).mean())
            wwc[:,i]/=sigma
            wws[:,i]/=sigma

        self.np_wwc=wwc
        self.np_wws=wws
        self.wwc=tf.constant(wwc)
        self.wws=tf.constant(wws)
        self.mat_avg_ang=np.zeros([NORIENT*NORIENT,NORIENT])
        for i in range(NORIENT):
            for j in range(NORIENT):
                self.mat_avg_ang[i+j*NORIENT,i]=1.0
        self.mat_avg_ang=tf.constant(self.mat_avg_ang)
    
        
    # ---------------------------------------------−---------
    # --       COMPUTE 3X3 INDEX FOR HEALPIX WORK          --
    # ---------------------------------------------−---------
    def corr_idx_wXX(self,x,y):
        idx=np.where(x==-1)[0]
        res=x
        res[idx]=y[idx]
        return(res)
    
    def comp_idx_w9(self,max_nside=1024):
        import healpy as hp
        nout=max_nside
        while nout>0:
            x,y,z=hp.pix2vec(nout,np.arange(12*nout**2),nest=True)
            vec=np.zeros([3,12*nout**2])
            vec[0,:]=x
            vec[1,:]=y
            vec[2,:]=z

            radius=np.sqrt(4*np.pi/(12*nout*nout))
            
            npt=9
            outname='W9'
            
            th,ph=hp.pix2ang(nout,np.arange(12*nout**2),nest=True)
            idx=hp.get_all_neighbours(nout,th,ph,nest=True)
    
            allidx=np.zeros([9,12*nout*nout],dtype='int')

            def corr(x,y):
                idx=np.where(x==-1)[0]
                res=x
                res[idx]=y[idx]
                return(res)

            allidx[4,:] = np.arange(12*nout**2)
            allidx[0,:] = self.corr_idx_wXX(idx[1,:],idx[2,:])
            allidx[1,:] = self.corr_idx_wXX(idx[2,:],idx[3,:])
            allidx[2,:] = self.corr_idx_wXX(idx[3,:],idx[4,:])
            
            allidx[3,:] = self.corr_idx_wXX(idx[0,:],idx[1,:])
            allidx[5,:] = self.corr_idx_wXX(idx[4,:],idx[5,:])
            
            allidx[6,:] = self.corr_idx_wXX(idx[7,:],idx[0,:])
            allidx[7,:] = self.corr_idx_wXX(idx[6,:],idx[7,:])
            allidx[8,:] = self.corr_idx_wXX(idx[5,:],idx[6,:])
            
            idx=np.zeros([12*nout*nout,npt],dtype='int')
            for iii in range(12*nout*nout):
                idx[iii,:]=allidx[:,iii]

            np.save('%s/%s_%d_IDX.npy'%(self.TEMPLATE_PATH,outname,nout),idx)
            print('%s/%s_%d_IDX.npy COMPUTED'%(self.TEMPLATE_PATH,outname,nout))
            nout=nout//2
            
    # ---------------------------------------------−---------
    # --       COMPUTE 5X5 INDEX FOR HEALPIX WORK          --
    # ---------------------------------------------−---------
    def comp_idx_w25(self,max_nside=1024):
        import healpy as hp
        nout=max_nside
        while nout>0:
            x,y,z=hp.pix2vec(nout,np.arange(12*nout**2),nest=True)
            vec=np.zeros([3,12*nout**2])
            vec[0,:]=x
            vec[1,:]=y
            vec[2,:]=z

            radius=np.sqrt(4*np.pi/(12*nout*nout))
            
            npt=25
            outname='W25'
            
            th,ph=hp.pix2ang(nout,np.arange(12*nout**2),nest=True)
            idx=hp.get_all_neighbours(nout,th,ph,nest=True)
    
            allidx=np.zeros([25,12*nout*nout],dtype='int')

            allidx[12,:] = np.arange(12*nout**2)
            allidx[11,:] = self.corr_idx_wXX(idx[0,:],idx[1,:])
            allidx[ 7,:] = self.corr_idx_wXX(idx[2,:],idx[3,:])
            allidx[13,:] = self.corr_idx_wXX(idx[4,:],idx[5,:])
            allidx[17,:] = self.corr_idx_wXX(idx[6,:],idx[7,:])
            
            allidx[10,:] = self.corr_idx_wXX(idx[0,allidx[11,:]],idx[1,allidx[11,:]])
            allidx[ 6,:] = self.corr_idx_wXX(idx[2,allidx[11,:]],idx[3,allidx[11,:]])
            allidx[16,:] = self.corr_idx_wXX(idx[6,allidx[11,:]],idx[7,allidx[11,:]])
            
            allidx[2,:]  = self.corr_idx_wXX(idx[2,allidx[7,:]],idx[3,allidx[7,:]])
            allidx[8,:]  = self.corr_idx_wXX(idx[4,allidx[7,:]],idx[5,allidx[7,:]])
            
            allidx[14,:]  = self.corr_idx_wXX(idx[4,allidx[13,:]],idx[5,allidx[13,:]])
            allidx[18,:]  = self.corr_idx_wXX(idx[6,allidx[13,:]],idx[7,allidx[13,:]])
            
            allidx[22,:]  = self.corr_idx_wXX(idx[6,allidx[17,:]],idx[7,allidx[17,:]])
            
            allidx[1,:]   = self.corr_idx_wXX(idx[2,allidx[6,:]],idx[3,allidx[6,:]])
            allidx[5,:]   = self.corr_idx_wXX(idx[0,allidx[6,:]],idx[1,allidx[6,:]])
            
            allidx[3,:]   = self.corr_idx_wXX(idx[2,allidx[8,:]],idx[3,allidx[8,:]])
            allidx[9,:]   = self.corr_idx_wXX(idx[4,allidx[8,:]],idx[5,allidx[8,:]])
            
            allidx[19,:]  = self.corr_idx_wXX(idx[4,allidx[18,:]],idx[5,allidx[18,:]])
            allidx[23,:]  = self.corr_idx_wXX(idx[6,allidx[18,:]],idx[7,allidx[18,:]])
            
            allidx[15,:]  = self.corr_idx_wXX(idx[0,allidx[16,:]],idx[1,allidx[16,:]])
            allidx[21,:]  = self.corr_idx_wXX(idx[6,allidx[16,:]],idx[7,allidx[16,:]])
            
            allidx[0,:]   = self.corr_idx_wXX(idx[0,allidx[1,:]],idx[1,allidx[1,:]])
            
            allidx[4,:]   = self.corr_idx_wXX(idx[4,allidx[3,:]],idx[5,allidx[3,:]])
            
            allidx[20,:]   = self.corr_idx_wXX(idx[0,allidx[21,:]],idx[1,allidx[21,:]])
            
            allidx[24,:]   = self.corr_idx_wXX(idx[4,allidx[23,:]],idx[5,allidx[23,:]])
            
            idx=np.zeros([12*nout*nout,npt],dtype='int')
            for iii in range(12*nout*nout):
                idx[iii,:]=allidx[:,iii]

            np.save('%s/%s_%d_IDX.npy'%(self.TEMPLATE_PATH,outname,nout),idx)
            print('%s/%s_%d_IDX.npy COMPUTED'%(self.TEMPLATE_PATH,outname,nout))
            nout=nout//2
    # ---------------------------------------------−---------
    def get_rank(self):
        return(self.rank)
    # ---------------------------------------------−---------
    def get_size(self):
        return(self.size)
    
    # ---------------------------------------------−---------
    def barrier(self):
        if self.isMPI:
            self.comm.Barrier()

    
    # ---------------------------------------------−---------
    def do_conv(self,image):
        c,s=self.get_ww()
        lout=int(np.sqrt(image.shape[0]//12))
        idx=np.load('%s/W%d_%d_IDX.npy'%(self.TEMPLATE_PATH,self.KERNELSZ**2,lout))
        res=np.zeros([12*lout*lout,self.NORIENT],dtype='complex')
        
        res[:,:].real=np.sum(image[idx].reshape(12*lout*lout,self.KERNELSZ**2,1)
                             *c.reshape(1,self.KERNELSZ**2,self.NORIENT),1)
        res[:,:].imag=np.sum(image[idx].reshape(12*lout*lout,self.KERNELSZ**2,1)
                             *s.reshape(1,self.KERNELSZ**2,self.NORIENT),1)
        return(res)
            
    # ---------------------------------------------−---------
    def get_ww(self):
        return(self.np_wwc,self.np_wws)
    # ---------------------------------------------−---------
    def plot_ww(self):
        c,s=self.get_ww()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16,6))
        npt=int(np.sqrt(c.shape[0]))
        for i in range(c.shape[1]):
            plt.subplot(2,c.shape[1],1+i)
            plt.imshow(c[:,i].reshape(npt,npt),cmap='Greys',vmin=-0.5,vmax=1.0)
            plt.subplot(2,c.shape[1],1+i+c.shape[1])
            plt.imshow(s[:,i].reshape(npt,npt),cmap='Greys',vmin=-0.5,vmax=1.0)
            sys.stdout.flush()
        plt.show()
    # ---------------------------------------------−---------
    def relu(self,x):
        return tf.nn.relu(x)
    # ---------------------------------------------−---------
    def hpwst_2(self,image1,mask,doL1=True,doL2=False):
            
        BATCH_SIZE=1
        im_shape = image1.get_shape().as_list()
        nout=int(im_shape[1])
        norient=self.NORIENT
            
        nstep=int(np.log(np.sqrt(nout/12))/np.log(2))-self.OSTEP
        lim1=image1
        tshape=mask.get_shape().as_list()
        
        vmask=mask

        n0=int(np.sqrt(nout/12))

        vscale=1.0
        all_nstep=0
        wshape=self.widx2[n0].get_shape().as_list()
        npt=wshape[0]//(12*n0*n0)
        s1=[]
        
        for iscale in range(nstep):
            vnorm=2.0/(tf.math.reduce_sum(vmask))
            im_shape = lim1.get_shape().as_list()
            alim1=tf.reshape(tf.gather(lim1,self.widx2[n0],axis=1),[BATCH_SIZE*norient,1,12*n0*n0,npt])
            cconv1 = tf.reduce_sum(self.wcos*alim1,3)
            sconv1 = tf.reduce_sum(self.wsin*alim1,3)

            tconvc1=cconv1*cconv1+sconv1*sconv1
            tmp1=self.L1(tconvc1)
            l_shape=tconvc1.get_shape().as_list()

            if doL1:
                vals=vnorm*tf.math.reduce_sum(tf.reshape(vmask,[self.NMASK,1,1,12*n0*n0])*tf.reshape(tmp1,[1,l_shape[0],l_shape[1],l_shape[2]]),3)
                ts1=vscale*tf.reshape(vals,[BATCH_SIZE*self.NMASK,norient,norient])
                s1.append(ts1)
                
            if doL2:
                vals=vnorm*tf.math.reduce_sum(tf.reshape(vmask,[self.NMASK,1,1,12*n0*n0])*tf.reshape(tconvc1,[1,l_shape[0],l_shape[1],l_shape[2]]),3)
                ts1=vscale*tf.reshape(vals,[BATCH_SIZE*self.NMASK,norient,norient])
                s1.append(ts1)
                
            lim1=tf.math.reduce_mean(tf.reshape(lim1,[BATCH_SIZE*norient,12*n0*n0//4,4,1,1]),2) 
            #lim1=tf.reshape(tf.nn.avg_pool(tf.reshape(lim1,[BATCH_SIZE*norient,12*n0*n0,1,1]),
#                                           ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1],padding='SAME'),[BATCH_SIZE*norient,12*n0*n0//4])
                
            tshape=vmask.get_shape().as_list()
            vmask=tf.math.reduce_mean(tf.reshape(vmask,[self.NMASK,tshape[1]//4,4]),2) 
            n0=n0//2
            #vscale=vscale*self.slope
        
        return(tf.concat(s1,2))


    def hpwst_2_cov(self, image1):

        BATCH_SIZE = 1
        im_shape = image1.get_shape().as_list()
        nout = int(im_shape[1])  # Number of pixels
        n0 = int(np.sqrt(nout / 12))  # NSIDE
        norient = self.NORIENT

        # Number of j scales
        J = int(np.log(np.sqrt(nout / 12)) / np.log(2))
        # Number of steps for the loop on scales
        nstep = J - self.OSTEP

        lim1 = image1  # [Nbatch, Npix, 1, 1]
        tshape = self.mask.get_shape().as_list()

        vmask = self.mask
        vscale = 1.0
        all_nstep = 0
        wshape = self.widx2[n0].get_shape().as_list()
        npt = wshape[0] // (12 * n0 * n0)
        S1, P00, C01, C11 = [], [], [], []
        I1_dic = {}

        #### Loop on each scale
        for j3 in range(nstep):
            vnorm = 2.0 / (tf.math.reduce_sum(vmask))
            im_shape = lim1.get_shape().as_list()

            #### Make the convolution
            alim1 = tf.reshape(tf.gather(lim1, self.widx2[n0], axis=1),
                               [BATCH_SIZE, 1, 12 * n0 * n0, npt])
            cconv1 = tf.reduce_sum(self.wcos * alim1, 3)  # Real part [Nbatch, Norient, Npix]
            sconv1 = tf.reduce_sum(self.wsin * alim1, 3)  # Imag part [Nbatch, Norient, Npix]
            # Take the module
            I1_square = cconv1 * cconv1 + sconv1 * sconv1  # Square module [Nbatch, Norient, Npix]
            I1 = tf.sqrt(I1_square)  # Module [Nbatch, Norient, Npix]
            I1_dic[j3] = I1
            l_shape = I1.get_shape().as_list()

            ### S1: L1 norm
            # Apply the mask on the convolved map and average over pixels
            vals = vnorm * tf.math.reduce_sum(tf.reshape(vmask, [self.NMASK, 1, 1, 12 * n0 * n0]) *
                                              tf.reshape(I1,[1, l_shape[0], l_shape[1], l_shape[2]]), 3)
            s1 = vscale * tf.reshape(vals, [BATCH_SIZE * self.NMASK, norient])  # [Nbatch x Nmask, Norient]
            S1.append(s1)

            ### P00: L2 norm
            vals_p00 = vnorm * tf.math.reduce_sum(tf.reshape(vmask, [self.NMASK, 1, 1, 12 * n0 * n0]) *
                                                  tf.reshape(I1_square,[1, l_shape[0], l_shape[1], l_shape[2]]), 3)
            p00 = vscale * tf.reshape(vals_p00, [BATCH_SIZE * self.NMASK, norient])  # [Nbatch x Nmask, Norient]
            P00.append(p00)

            ### C01
            for j2 in range(0, j3):  # j2 <= j3
                I1convPsi2 = tf.reshape(tf.gather(I1_dic[j2], self.widx2[n0], axis=2),
                                       [BATCH_SIZE, norient, 12 * n0 * n0, npt])  # [Nbatch, Norient, Npix, Npt]
                cI1convPsi2 = tf.reduce_sum(self.wcos * I1convPsi2, 3)  # Real part [Nbatch, Norient, Npix]
                sI1convPsi2 = tf.reduce_sum(self.wsin * I1convPsi2, 3)  # Imag part [Nbatch, Norient, Npix]

                product = cconv1[:, None, :, :] * cI1convPsi2[:, :, None, :] +\
                          sconv1[:, None, :, :] * sI1convPsi2[:, :, None, :]  # [Nbatch, Norient, Norient, Npix]
                # Sum over pixels and apply the mask
                c01 = vscale * tf.reduce_sum(tf.reshape(vmask, [1, self.NMASK, 1, 1, 12*n0*n0]) *
                                             product[:, None, :, :, :], axis=4)  # [Nmask, Nbatch, Norient, Norient]
                C01.append(c01)

                # ### C11
                C11 = C01
                # for j1 in range(0, j2 + 1):  # j1 <= j2
                #     c11 =
                #     C11.append(c11)

            ### Reshape the image and the mask for next iteration
            lim1=tf.math.reduce_mean(tf.reshape(lim1,[BATCH_SIZE,12*n0*n0//4,4,1,1]),2) 
            #lim1 = tf.reshape(tf.nn.avg_pool(tf.reshape(lim1, [BATCH_SIZE, 12 * n0 * n0, 1, 1]),
            #                                 ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME'),
            #                  [BATCH_SIZE, 12 * n0 * n0 // 4])
            for j2 in range(0, j3+1):  # j2 <= j3
                I1_dic[j2] = tf.reshape(tf.nn.avg_pool(tf.reshape(I1_dic[j2],
                                                                  [BATCH_SIZE, norient, 12 * n0 * n0, 1]),
                                             ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='SAME'),
                              [BATCH_SIZE, norient,  12 * n0 * n0 // 4])
            # Update the mask for next iteration
            tshape = vmask.get_shape().as_list()
            vmask = tf.math.reduce_mean(tf.reshape(vmask, [self.NMASK, tshape[1] // 4, 4]), 2)
            # Update NSIDE for next iteration
            n0 = n0 // 2

            vscale=vscale*self.slope

        return tf.concat(S1, 0), tf.concat(P00, 0), tf.concat(C01, 0), tf.concat(C11, 0)

    # ---------------------------------------------−---------
    #  Compute the CWST on the sphere
    # 
    # ---------------------------------------------−---------
    def hpwst1(self,image1,image2,imaginary=False,avg_ang=False,doL1=True,doL2=False):
            
        BATCH_SIZE=1
        im_shape = image1.get_shape().as_list()
        nout=int(im_shape[1]) 
        norient=self.NORIENT  

        # compute the number of step : OSTEP can be configured
        nstep=int(np.log(np.sqrt(nout/12))/np.log(2))-self.OSTEP

        # set temporal tensor with downgraded image
        lim1=image1
        lim2=image2
        tshape=self.mask.get_shape().as_list()

        # same for mask
        vmask=self.mask

        # compute the input nside
        n0=int(np.sqrt(nout/12))

        # information for internal loop
        vscale=1.0
        all_nstep=0
        wshape=self.widx2[n0].get_shape().as_list()

        # number of wieghts in the convolution
        npt=wshape[0]//(12*n0*n0)

        # loop on all scales
        for iscale in range(nstep):

            vnorm=2.0/(tf.math.reduce_sum(vmask))
            im_shape = lim1.get_shape().as_list()

            # build table (BATCHSIZE)x(1)x(12*nside*nside)x(npt) of the two input maps
            alim1=tf.reshape(tf.gather(lim1,self.widx2[n0],axis=1),[BATCH_SIZE,1,12*n0*n0,npt])
            alim2=tf.reshape(tf.gather(lim2,self.widx2[n0],axis=1),[BATCH_SIZE,1,12*n0*n0,npt])
            
            # compute convolution for the real and imaginary part of each image
            # sum (1)x(NORIENT)x(1)x(npt) * (BATCHSIZE)x(NORIENT)x(12*nside*nside)x(npt) sum sur (npt)
            # => (BATCHSIZE)x(NORIENT)x(12*nside*nside)
            cconv1 = tf.reduce_sum(self.wcos*alim1,3) 
            cconv2 = tf.reduce_sum(self.wcos*alim2,3)
            sconv1 = tf.reduce_sum(self.wsin*alim1,3)
            sconv2 = tf.reduce_sum(self.wsin*alim2,3)

            # compute the L2 norm
            tconvc1=self.ampnorm[iscale]*(cconv1*cconv2+sconv1*sconv2)
            # compute L1 norm
            tmpc1=tf.reshape(self.L1(tconvc1),[BATCH_SIZE*norient,12*n0*n0])
            
            l_shape=tconvc1.get_shape().as_list()
                
            if doL1:
                #do the sum using mask
                valc=vnorm*tf.math.reduce_sum(tf.reshape(vmask,[self.NMASK,1,1,12*n0*n0])*tf.reshape(tmpc1    ,[1,l_shape[0],l_shape[1],l_shape[2]]),3)
                # add to outputed stats
                ts1=vscale*tf.reshape(valc,[BATCH_SIZE*self.NMASK,1,norient])
                if iscale==0:
                    s1=ts1
                else:
                    s1=tf.concat([s1,ts1],2)
            if doL2:
                
                valc=vnorm*tf.math.reduce_sum(tf.reshape(vmask,[self.NMASK,1,1,12*n0*n0])*tf.reshape(tconvc1,[1,l_shape[0],l_shape[1],l_shape[2]]),3)
                ts1=vscale*tf.reshape(valc,[BATCH_SIZE*self.NMASK,1,norient])
                if iscale==0 and doL1==False:
                    s1=ts1
                else:
                    s1=tf.concat([s1,ts1],2)

            if imaginary:
                # do the same thing on imaginary part of the cross norm
                tconvs1=self.ampnorm[iscale]*(cconv1*sconv2-sconv1*cconv2)
                tmps1=tf.reshape(self.L1(tconvs1),[BATCH_SIZE*norient,12*n0*n0])
                    
                if doL1:
                    vals=vnorm*tf.math.reduce_sum(tf.reshape(vmask,[self.NMASK,1,1,12*n0*n0])*tf.reshape(tmps1    ,[1,l_shape[0],l_shape[1],l_shape[2]]),3)
                    ts1=vscale*tf.reshape(vals,[BATCH_SIZE*self.NMASK,1,norient])
                    if iscale==0:
                        c1=ts1
                    else:
                        c1=tf.concat([c1,ts1],2)
                if doL2:
                    vals=vnorm*tf.math.reduce_sum(tf.reshape(vmask,[self.NMASK,1,1,12*n0*n0])*tf.reshape(tconvs1,[1,l_shape[0],l_shape[1],l_shape[2]]),3)
                    ts1=vscale*tf.reshape(vals,[BATCH_SIZE*self.NMASK,1,norient])
                    if iscale==0 and doL1==False:
                        c1=ts1
                    else:
                        c1=tf.concat([c1,ts1],2)

            if iscale<nstep-1:
                # si l'echelle le permet calcul les coefs S2 callin hpwst2
                val2c=self.hpwst_2(self.relu(tmpc1),vmask,doL1=doL1,doL2=doL2)-self.hpwst_2(self.relu(-tmpc1),vmask,doL1=doL1,doL2=doL2)
                ts2= vscale*val2c
                if iscale==0:
                    s2=ts2
                else:
                    s2=tf.concat([s2,ts2],2)
                if imaginary:
                    val2s=self.hpwst_2(self.relu(tmps1),vmask,doL1=doL1,doL2=doL2)-self.hpwst_2(self.relu(-tmps1),vmask,doL1=doL1,doL2=doL2)
                
                    ts2= vscale*val2s
                    if iscale==0:
                        c2=ts2
                    else:
                        c2=tf.concat([c2,ts2],2)

            lim1=tf.math.reduce_mean(tf.reshape(lim1,[BATCH_SIZE,12*n0*n0//4,4,1,1]),2) 
            lim2=tf.math.reduce_mean(tf.reshape(lim2,[BATCH_SIZE,12*n0*n0//4,4,1,1]),2) 
            #lim1=tf.nn.avg_pool(lim1, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1],padding='SAME')
            #lim2=tf.nn.avg_pool(lim2, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1],padding='SAME')
                
            tshape=vmask.get_shape().as_list()
            vmask=tf.math.reduce_mean(tf.reshape(vmask,[self.NMASK,tshape[1]//4,4]),2) 
            n0=n0//2
            vscale=vscale*self.slope
            
        if imaginary:
            s1=tf.concat([s1,c1],0)
            s2=tf.concat([s2,c2],0)

        if avg_ang:
            lshape=s1.get_shape().as_list()
            s1=tf.math.reduce_mean(tf.reshape(s1,[lshape[0],lshape[2]//self.NORIENT,self.NORIENT]),2)
            lshape=s2.get_shape().as_list()
            s2=tf.reshape(s2,[lshape[0],self.NORIENT,lshape[2]//self.NORIENT,self.NORIENT])
            s2=tf.reshape(tf.transpose(s2,[0,2,1,3]),[lshape[0],lshape[2]//self.NORIENT,self.NORIENT*self.NORIENT])
            s2=tf.reshape(tf.matmul(s2,self.mat_avg_ang),[lshape[0],lshape[2]])
            
        return(s1,s2)
    # ---------------------------------------------−---------      
    def cwst_2(self,image1,mask,doL1=True,doL2=False):

        BATCH_SIZE=1
        slope=self.slope
        norient=self.NORIENT
        
        im_shape = image1.get_shape().as_list()
        n0=int(im_shape[1])
        n1=int(im_shape[2])
        nstep=int(np.log(n0)/np.log(2))-1
        lim1=tf.reshape(tf.transpose(image1,[0,3,1,2]),[BATCH_SIZE*self.NORIENT,n0,n1,1])
        
        vmask=mask
        

        vscale=1.0
        all_nstep=0
        iwwc=tf.reshape(self.wwc,[self.KERNELSZ,self.KERNELSZ,1,self.NORIENT])
        iwws=tf.reshape(self.wws,[self.KERNELSZ,self.KERNELSZ,1,self.NORIENT])
        for iscale in range(nstep):
            convc1 = tf.nn.conv2d(lim1,iwwc,strides=[1, 1, 1, 1],
                                  padding=self.padding,name='cconv1_%d'%(iscale))
            convs1 = tf.nn.conv2d(lim1,iwws,strides=[1, 1, 1, 1],
                                  padding=self.padding,name='sconv1_%d'%(iscale))
            
            tconvc1_l2=convc1*convc1+convs1*convs1
            tconvc1=self.L1(tconvc1_l2)
            if doL1:
                valc=tf.math.reduce_sum(tf.reshape(tf.reshape(vmask,[BATCH_SIZE,1,self.NMASK,n0,n1,1])*
                                                   tf.reshape(tconvc1,[BATCH_SIZE,self.NORIENT,1,n0,n1,self.NORIENT]),
                                                   [BATCH_SIZE*self.NMASK*self.NORIENT,n0*n1,self.NORIENT]),1)
                ts1=vscale*tf.reshape(valc,[BATCH_SIZE*self.NMASK,self.NORIENT,self.NORIENT])
                if iscale==0:
                    s1=ts1
                else:
                    s1=tf.concat([s1,ts1],2)
            if doL2:
                valc=tf.math.reduce_sum(tf.reshape(tf.reshape(vmask,[BATCH_SIZE,1,self.NMASK,n0,n1,1])*
                                                   tf.reshape(tconvc1_l2,[BATCH_SIZE,self.NORIENT,1,n0,n1,self.NORIENT]),
                                                   [BATCH_SIZE*self.NMASK*self.NORIENT,n0*n1,self.NORIENT]),1)
                ts1=vscale*tf.reshape(valc,[BATCH_SIZE*self.NMASK,self.NORIENT,self.NORIENT])
                if iscale==0 and doL1==False:
                    s1=ts1
                else:
                    s1=tf.concat([s1,ts1],2)
                
            lim1=2*tf.nn.avg_pool(lim1,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
            vmask=tf.reshape(tf.nn.avg_pool(tf.reshape(vmask,[self.NMASK,n0,n1,1]), ksize=[1, 2, 2, 1],
                                            strides=[1, 2, 2, 1],padding='SAME'),
                             [self.NMASK,n0//2,n1//2,1])
            
            n0=n0//2
            n1=n1//2
            vscale=vscale*self.slope
        return(s1)
    
    # ---------------------------------------------−---------
    def L1(self,tmp):
        return(tf.sign(tmp)*tf.sqrt(tf.sign(tmp)*tmp))
    
    # ---------------------------------------------−---------    
    def cwst1(self,image1,image2,imaginary=False,avg_ang=False,doL1=True,doL2=False):

        BATCH_SIZE=1
        slope=self.slope
        im_shape = image1.get_shape().as_list()
        nout=int(im_shape[1])
        norient=self.NORIENT
        nstep=int(np.log(nout)/np.log(2))-1
        lim1=image1
        lim2=image2
        
        vmask=self.mask
        
        n0=1*nout
        n1 = int(im_shape[2])
        vscale=1.0
        all_nstep=0
        iwwc=tf.reshape(self.wwc,[self.KERNELSZ,self.KERNELSZ,1,self.NORIENT])
        iwws=tf.reshape(self.wws,[self.KERNELSZ,self.KERNELSZ,1,self.NORIENT])
        
        for iscale in range(nstep):
            convc1 = tf.nn.conv2d(lim1,iwwc,strides=[1, 1, 1, 1],padding=self.padding,name='cconv1_%d'%(iscale))
            convc2 = tf.nn.conv2d(lim2,iwwc,strides=[1, 1, 1, 1],padding=self.padding,name='cconv2_%d'%(iscale))
            convs1 = tf.nn.conv2d(lim1,iwws,strides=[1, 1, 1, 1],padding=self.padding,name='sconv1_%d'%(iscale))
            convs2 = tf.nn.conv2d(lim2,iwws,strides=[1, 1, 1, 1],padding=self.padding,name='sconv2_%d'%(iscale))
            
            tconvc1_l2=convc1*convc2+convs1*convs2
            tconvc1=self.L1(tconvc1_l2)

            if doL1:
                valc=tf.math.reduce_sum(tf.reshape(tf.reshape(vmask,[BATCH_SIZE,self.NMASK,n0,n1,1])*
                                                   tf.reshape(tconvc1,[BATCH_SIZE,1,n0,n1,norient]),
                                                   [BATCH_SIZE*self.NMASK,n0*n1,norient]),1)
                ts1=vscale*tf.reshape(valc,[BATCH_SIZE*self.NMASK,1,norient])
                if iscale==0:
                    s1=ts1
                else:
                    s1=tf.concat([s1,ts1],2)
            if doL2:
                valc=tf.math.reduce_sum(tf.reshape(tf.reshape(vmask,[BATCH_SIZE,self.NMASK,n0,n1,1])*
                                                   tf.reshape(tconvc1_l2,[BATCH_SIZE,1,n0,n1,norient]),
                                                   [BATCH_SIZE*self.NMASK,n0*n1,norient]),1)
                ts1=vscale*tf.reshape(valc,[BATCH_SIZE*self.NMASK,1,norient])
                if iscale==0 and doL1==False:
                    s1=ts1
                else:
                    s1=tf.concat([s1,ts1],2)

            if imaginary:
                tconvs1_l2=convc1*convs2-convc2*convs1
                tconvs1=self.L1(tconvs1_l2)
                
                if doL1:
                    vals=tf.math.reduce_sum(tf.reshape(tf.reshape(vmask,[BATCH_SIZE,self.NMASK,n0,n1,1])*
                                                       tf.reshape(tconvs1,[BATCH_SIZE,1,n0,n1,norient]),
                                                       [BATCH_SIZE*self.NMASK,n0*n1,norient]),1)
                    ts1=vscale*tf.reshape(vals,[BATCH_SIZE*self.NMASK,1,norient])
                    if iscale==0:
                        c1=ts1
                    else:
                        c1=tf.concat([c1,ts1],2)
                if doL2:
                    vals=tf.math.reduce_sum(tf.reshape(tf.reshape(vmask,[BATCH_SIZE,self.NMASK,n0,n1,1])*
                                                       tf.reshape(tconvs1_l2,[BATCH_SIZE,1,n0,n1,norient]),
                                                       [BATCH_SIZE*self.NMASK,n0*n1,norient]),1)
                    ts1=vscale*tf.reshape(vals,[BATCH_SIZE*self.NMASK,1,norient])
                    if iscale==0 and doL1==False:
                        c1=ts1
                    else:
                        c1=tf.concat([c1,ts1],2)

            if iscale<nstep-1:
                val2c=self.cwst_2(self.relu(tconvc1),vmask,doL1=doL1,doL2=doL2)-self.cwst_2(self.relu(-tconvc1),vmask,doL1=doL1,doL2=doL2)
            
                tshape2=val2c.get_shape().as_list()
                
                ts2= vscale*val2c
                if iscale==0:
                    s2=ts2
                else:
                    s2=tf.concat([s2,ts2],2)
                
                if imaginary:
                    val2s=self.cwst_2(self.relu(tconvs1),vmask,doL1=doL1,doL2=doL2)-self.cwst_2(self.relu(-tconvs1),vmask,doL1=doL1,doL2=doL2)
                    
                    ts2= vscale*val2s
                    
                    if iscale==0:
                        c2=ts2
                    else:
                        c2=tf.concat([c2,ts2],2)

                lim1=2*tf.nn.avg_pool(lim1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
                lim2=2*tf.nn.avg_pool(lim2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
                vmask=tf.reshape(tf.nn.avg_pool(tf.reshape(vmask,[self.NMASK,n0,n1,1]), ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],padding='SAME'),[self.NMASK,n0//2,n1//2,1])
                n0=n0//2
                n1=n1//2
                vscale=vscale*slope
                
        if imaginary:
            s1=tf.concat([s1,c1],0)
            s2=tf.concat([s2,c2],0)

        if avg_ang:
            lshape=s1.get_shape().as_list()
            s1=tf.math.reduce_mean(tf.reshape(s1,[lshape[0],lshape[2]//self.NORIENT,self.NORIENT]),2)
            lshape=s2.get_shape().as_list()
            s2=tf.reshape(s2,[lshape[0],self.NORIENT,lshape[2]//self.NORIENT,self.NORIENT])
            s2=tf.reshape(tf.transpose(s2,[0,2,1,3]),[lshape[0],lshape[2]//self.NORIENT,self.NORIENT*self.NORIENT])
            s2=tf.reshape(tf.matmul(s2,self.mat_avg_ang),[lshape[0],lshape[2]])
            
        return(s1,s2)
    # ---------------------------------------------−---------
    
    def donaiveinterpolH(self,im,mask):
        lmask=1*mask
        lim=1*im
        lim[mask==0]=hp.UNSEEN
        idx=np.where(lim==hp.UNSEEN)[0]
        nin=int(np.sqrt(im.shape[0]//12))//2
        nout=2*nin
        while nin>0 and len(idx)>0:
            th,ph=hp.pix2ang(nout,idx)
            pidx=hp.ang2pix(nin,th,ph)
            llim=hp.ud_grade(lim,nin)
            lim[idx]=llim[pidx]
            nin=nin//2
            idx=np.where(lim==hp.UNSEEN)[0]
        return(lim)
    
    # ---------------------------------------------−---------
    
    def donaiveinterpol(self,im,mask,xpadding=False):
        lmask=1*mask
        tot=lmask.mean()
        while tot!=1.0:
            idx=np.where(lmask==0)
            nx,ny=im.shape
            res=im[(idx[0]+1)%nx,idx[1]]   *lmask[(idx[0]+1)%nx,idx[1]]
            res+=im[(idx[0]-1+nx)%nx,idx[1]]*lmask[(idx[0]-1+nx)%nx,idx[1]]
            res+=im[(idx[0]+1)%nx,(idx[1]+1)%ny]   *lmask[(idx[0]+1)%nx,(idx[1]+1)%ny]
            res+=im[(idx[0]-1+nx)%nx,(idx[1]+1)%ny]*lmask[(idx[0]-1+nx)%nx,(idx[1]+1)%ny]
            res+=im[(idx[0]+1)%nx,(idx[1]-1+ny)%ny]   *lmask[(idx[0]+1)%nx,(idx[1]-1+ny)%ny]
            res+=im[(idx[0]-1+nx)%nx,(idx[1]-1+ny)%ny]*lmask[(idx[0]-1+nx)%nx,(idx[1]-1+ny)%ny]
            res+=im[idx[0],(idx[1]+1)%ny]   *lmask[idx[0],(idx[1]+1)%ny]
            res+=im[idx[0],(idx[1]-1+ny)%ny]*lmask[idx[0],(idx[1]-1+ny)%ny]
            nres=lmask[(idx[0]+1)%nx,idx[1]]
            nres+=lmask[(idx[0]-1+nx)%nx,idx[1]]
            nres+=lmask[idx[0],(idx[1]+1)%ny]
            nres+=lmask[idx[0],(idx[1]-1+ny)%ny]
            nres+=lmask[(idx[0]+1)%nx,(idx[1]+1)%ny]
            nres+=lmask[(idx[0]-1+nx)%nx,(idx[1]+1)%ny]
            nres+=lmask[(idx[0]+1)%nx,(idx[1]-1+ny)%ny]
            nres+=lmask[(idx[0]-1+nx)%nx,(idx[1]-1+ny)%ny]
            im[idx[0][nres>0],idx[1][nres>0]]=res[nres>0]/nres[nres>0]
            lmask[idx[0][nres>0],idx[1][nres>0]]=1.0
            tot=lmask.mean()
        if xpadding:
            im[:,0:im.shape[1]//4]=im[:,im.shape[1]//2:im.shape[1]//2+im.shape[1]//4]
            im[:,-im.shape[1]//4:]=im[:,im.shape[1]//4:im.shape[1]//2]
        return(im)
    # ---------------------------------------------−---------
    def add_mask(self,mask):
        
        for i in range(1,mask.shape[0]):
            mask[i,:]=mask[i]*mask[0].sum()/mask[i].sum()
            
        if self.healpix:
            self.mask=tf.reshape(tf.constant(mask.astype(self.all_type)),[mask.shape[0],mask.shape[1],1,1])
            self.NMASK=mask.shape[0]
        else:
            self.mask=tf.reshape(tf.constant(mask.astype(self.all_type)),[mask.shape[0],mask.shape[1],mask.shape[2],1])
            self.NMASK=mask.shape[0]
        if self.rank==0:
            print('Use %d masks'%(self.NMASK))
            sys.stdout.flush()
            
    # ---------------------------------------------−---------
    def add_loss_diff(self,image1,image2,mask,weight=1.0):
        with tf.device(self.gpulist[self.gpupos%self.ngpu]):
            self.diff_map1[self.nloss]=image1
            self.diff_map2[self.nloss]=image2
            self.diff_mask[self.nloss]=mask
            self.diff_weight[self.nloss]=tf.constant(np.array([weight]).astype(self.all_type))
            self.loss_type[self.nloss]=self.MAPDIFF
            self.nloss=self.nloss+1
            
    # ---------------------------------------------−---------
    def add_loss_noise_stat(self,mask,noise,image,weight=1.0):
        with tf.device(self.gpulist[self.gpupos%self.ngpu]):
            vv=np.zeros([self.NMASK])
            for k in range(self.NMASK):
                vv[k]=np.mean((noise*mask[k].reshape(1,12*self.nout*self.nout))**2)
            nstat=tf.reduce_mean(tf.square(tf.reshape(self.mask,[self.NMASK,12*self.nout*self.nout])
                                           *tf.reshape(self.param[0],[1,12*self.nout*self.nout])),1)
            self.diff_map1[self.nloss]=tf.constant(vv)
            self.diff_map2[self.nloss]=nstat
            self.diff_weight[self.nloss]=weight
            self.loss_type[self.nloss]=self.NOISESTAT
            self.nloss=self.nloss+1
        
    # ---------------------------------------------−---------
    def add_loss(self,image1,image2,image3,image4,doL1=True,doL2=False,imaginary=False,avg_ang=False,weight=1.0):

        if doL1==False and doL2==False:
            print('You have to choose a statistic L1 or L2 or both here doL1=False and doL2=False')
            exit(0)
            
        with tf.device(self.gpulist[self.nloss%self.ngpu]):
            if self.nout!=-1:
                os1,os2=self.hpwst1(image1,image2,doL1=doL1,doL2=doL2,imaginary=imaginary,avg_ang=avg_ang)
                is1,is2=self.hpwst1(image3,image4,doL1=doL1,doL2=doL2,imaginary=imaginary,avg_ang=avg_ang)
            else:
                os1,os2=self.cwst1(image1,image2,imaginary=imaginary,avg_ang=avg_ang,doL1=doL1,doL2=doL2)
                is1,is2=self.cwst1(image3,image4,imaginary=imaginary,avg_ang=avg_ang,doL1=doL1,doL2=doL2)

            self.os1[self.nloss]=os1
            self.os2[self.nloss]=os2
            self.is1[self.nloss]=is1
            self.is2[self.nloss]=is2

            ss1=is1.get_shape().as_list()
            ss2=is2.get_shape().as_list()

            if avg_ang:
                self.tw1[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss1[0],ss1[1]),
                                                              name='TW1_%d'%(self.nloss))
                self.tw2[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss2[0],ss2[1]),
                                                              name='TW2_%d'%(self.nloss))
                self.tb1[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss1[0],ss1[1]),
                                                              name='TB1_%d'%(self.nloss))
                self.tb2[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss2[0],ss2[1]),
                                                              name='TB2_%d'%(self.nloss))
            else:
                self.tw1[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss1[0],ss1[1],ss1[2]),
                                                              name='TW1_%d'%(self.nloss))
                self.tw2[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss2[0],ss2[1],ss2[2]),
                                                              name='TW2_%d'%(self.nloss))
                self.tb1[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss1[0],ss1[1],ss1[2]),
                                                              name='TB1_%d'%(self.nloss))
                self.tb2[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss2[0],ss2[1],ss2[2]),
                                                              name='TB2_%d'%(self.nloss))
                
            self.ss1[self.nloss]=ss1
            self.ss2[self.nloss]=ss2
            self.loss_type[self.nloss]=self.SCATDIFF
            self.loss_weight[self.nloss]=weight
            self.nloss=self.nloss+1

    # ---------------------------------------------−---------
    def add_loss_cov(self, image1, image2, weight=1.0):

        with tf.device(self.gpulist[self.nloss % self.ngpu]):
            if self.nout != -1:
                iS1, iP00, iC01, iC11 = self.hpwst_2_cov(image1)
                oS1, oP00, oC01, oC11 = self.hpwst_2_cov(image2)
            else:
                print('Not developed for 2D plan.')
                exit(0)

            self.os1[self.nloss] = iS1
            self.os2[self.nloss] = iP00
            self.os3[self.nloss] = iC01
            self.os4[self.nloss] = iC11

            self.is1[self.nloss] = oS1
            self.is2[self.nloss] = oP00
            self.is3[self.nloss] = oC01
            self.is4[self.nloss] = oC11

            ss1 = iS1.shape
            ss2 = iP00.shape
            ss3 = iC01.shape
            ss4 = iC11.shape

            # Sigma
            self.tw1[self.nloss] = tf.compat.v1.placeholder(self.all_tf_type,
                                                            shape=ss1,
                                                            name='TW1_%d' % (self.nloss))
            self.tw2[self.nloss] = tf.compat.v1.placeholder(self.all_tf_type,
                                                            shape=ss2,
                                                            name='TW2_%d' % (self.nloss))
            self.tw3[self.nloss] = tf.compat.v1.placeholder(self.all_tf_type,
                                                            shape=ss3,
                                                            name='TW1_%d' % (self.nloss))
            self.tw4[self.nloss] = tf.compat.v1.placeholder(self.all_tf_type,
                                                            shape=ss4,
                                                            name='TW2_%d' % (self.nloss))
            # Bias
            self.tb1[self.nloss] = tf.compat.v1.placeholder(self.all_tf_type,
                                                            shape=ss1,
                                                            name='TW1_%d' % (self.nloss))
            self.tb2[self.nloss] = tf.compat.v1.placeholder(self.all_tf_type,
                                                            shape=ss2,
                                                            name='TW2_%d' % (self.nloss))
            self.tb3[self.nloss] = tf.compat.v1.placeholder(self.all_tf_type,
                                                            shape=ss3,
                                                            name='TW1_%d' % (self.nloss))
            self.tb4[self.nloss] = tf.compat.v1.placeholder(self.all_tf_type,
                                                            shape=ss4,
                                                            name='TW2_%d' % (self.nloss))

            self.ss1[self.nloss] = ss1
            self.ss2[self.nloss] = ss2
            self.ss3[self.nloss] = ss3
            self.ss4[self.nloss] = ss4
            self.loss_type[self.nloss] = self.SCATCOV
            self.loss_weight[self.nloss] = weight
            self.nloss = self.nloss + 1

    # ---------------------------------------------−---------
    def add_loss_healpix(self,image1,image2,image3,image4,avg_ang=False,imaginary=False,weight=1.0,doL1=True,doL2=False):
        
        self.add_loss(image1,image2,image3,image4,avg_ang=avg_ang,imaginary=imaginary,weight=weight,doL1=doL1,doL2=doL2)
    # ---------------------------------------------−---------  
    def add_loss_2d(self,image1,image2,image3,image4,avg_ang=False,imaginary=False,weight=1.0,doL1=True,doL2=False):
        
        self.add_loss(image1,image2,image3,image4,avg_ang=avg_ang,imaginary=imaginary,weight=weight,doL1=doL1,doL2=doL2)
    
    # ---------------------------------------------−---------  
    def add_loss_determ(self,image1,image2,doL1=True,doL2=False,avg_ang=False,imaginary=False,weight=1.0):
        
        if doL1==False and doL2==False:
            print('You have to choose a statistic L1 or L2 or both here doL1=False and doL2=False')
            exit(0)
            
        with tf.device(self.gpulist[self.nloss%self.ngpu]):
            if self.nout!=-1:
                os1,os2=self.hpwst1(image1,image2,doL1=doL1,doL2=doL2,imaginary=imaginary,avg_ang=avg_ang)
            else:
                os1,os2=self.cwst1(image1,image2,doL1=doL1,doL2=doL2,imaginary=imaginary,avg_ang=avg_ang)

            self.os1[self.nloss]=os1
            self.os2[self.nloss]=os2
            self.is1[self.nloss]=tf.constant(0,self.all_tf_type)
            self.is2[self.nloss]=tf.constant(0,self.all_tf_type)

            ss1=os1.get_shape().as_list()
            ss2=os2.get_shape().as_list()

            if avg_ang:
                self.tw1[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss1[0],ss1[1]),
                                                              name='TW1_%d'%(self.nloss))
                self.tw2[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss2[0],ss2[1]),
                                                              name='TW2_%d'%(self.nloss))
                self.tb1[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss1[0],ss1[1]),
                                                              name='TB1_%d'%(self.nloss))
                self.tb2[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss2[0],ss2[1]),
                                                              name='TB2_%d'%(self.nloss))
            else:
                self.tw1[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss1[0],ss1[1],ss1[2]),
                                                              name='TW1_%d'%(self.nloss))
                self.tw2[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss2[0],ss2[1],ss2[2]),
                                                              name='TW2_%d'%(self.nloss))
                self.tb1[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss1[0],ss1[1],ss1[2]),
                                                              name='TB1_%d'%(self.nloss))
                self.tb2[self.nloss]=tf.compat.v1.placeholder(self.all_tf_type,
                                                              shape=(ss2[0],ss2[1],ss2[2]),
                                                              name='TB2_%d'%(self.nloss))
                
            self.ss1[self.nloss]=ss1
            self.ss2[self.nloss]=ss2
            self.loss_type[self.nloss]=self.SCATDIFF
            self.loss_weight[self.nloss]=weight

            self.nloss=self.nloss+1
        
    # ---------------------------------------------−---------
    def calc_stat(self,n1,n2,imaginary=False,gpupos=0,avg_ang=False,doL1=True,doL2=False):
        
        if doL1==False and doL2==False:
            print('You have to choose a statistic L1 or L2 or both here doL1=False and doL2=False')
            exit(0)
            
        with tf.device(self.gpulist[gpupos%self.ngpu]):
            nsim=n1.shape[0]
            for i in range(nsim):
                feed_dict={}
                if self.nout!=-1:
                    feed_dict[self.noise1]=n1[i].reshape(1,12*self.nout*self.nout,1,1)
                    feed_dict[self.noise2]=n2[i].reshape(1,12*self.nout*self.nout,1,1)
                else:
                    nx=n1.shape[1]
                    ny=n1.shape[2]
                    feed_dict[self.noise1]=n1[i].reshape(1,nx,ny,1)
                    feed_dict[self.noise2]=n2[i].reshape(1,nx,ny,1)

                    
                icase=int((imaginary==True)
                          +2*(avg_ang==True)
                          +4*(doL1==True)
                          +8*(doL2==True))
                
                o1,o2= self.sess.run([self.on1[icase],self.on2[icase]],feed_dict=feed_dict)

                if i==0:
                    if avg_ang:
                        stat_o1=np.zeros([nsim,o1.shape[0],o1.shape[1]])
                        stat_o2=np.zeros([nsim,o2.shape[0],o2.shape[1]])
                    else:
                        stat_o1=np.zeros([nsim,o1.shape[0],o1.shape[1],o1.shape[2]])
                        stat_o2=np.zeros([nsim,o2.shape[0],o2.shape[1],o2.shape[2]])
                stat_o1[i]=o1
                stat_o2[i]=o2
                
            return(stat_o1,stat_o2)

    # ---------------------------------------------−---------
    def calc_stat_cov(self, n1, gpupos=0):

        with tf.device(self.gpulist[gpupos % self.ngpu]):
            nsim = n1.shape[0]
            for i in range(nsim):
                feed_dict = {}
                if self.nout != -1:
                    feed_dict[self.noise1] = n1[i].reshape(1, 12 * self.nout * self.nout, 1, 1)
                else:
                    print('Only work in Healpix.')

                S1, P00, C01, C11 = self.sess.run([self.S1, self.P00, self.C01, self.C11], feed_dict=feed_dict)

                if i == 0:
                    stat_S1 = np.zeros([nsim] + list(S1.shape))
                    stat_P00 = np.zeros([nsim] + list(P00.shape))
                    stat_C01 = np.zeros([nsim] + list(C01.shape))
                    stat_C11 = np.zeros([nsim] + list(C11.shape))
                    # stat_S1 = np.zeros([nsim, S1.shape[0], S1.shape[1]])
                    # stat_P00 = np.zeros([nsim, P00.shape[0], P00.shape[1]])
                    # stat_C01 = np.zeros([nsim, C01.shape[0], C01.shape[1]])
                    # stat_C11 = np.zeros([nsim, C11.shape[0], C11.shape[1]])
                stat_S1[i] = S1
                stat_P00[i] = P00
                stat_C01[i] = C01
                stat_C11[i] = C11

            return (stat_S1, stat_P00, stat_C01, stat_C11)

    # ---------------------------------------------−---------    
    def convimage(self,image):
        with tf.device(self.gpulist[self.gpupos%self.ngpu]):
            if self.healpix:
                return(self.convhealpix(image))
            
            return(tf.constant((image.astype(self.all_type)).reshape(1,image.shape[0],image.shape[1],1)))
    def convhealpix(self,image):
        with tf.device(self.gpulist[self.gpupos%self.ngpu]):
            return(tf.constant((image.astype(self.all_type)).reshape(1,image.shape[0],1,1)))
    # ---------------------------------------------−---------
    def reset(self):
        self.sess.run(self.doreset)

    # ---------------------------------------------−---------   
    def init_synthese(self,image,image1=None,image2=None,interpol=[],xpadding=False):
        self.nout=-1
        with tf.device(self.gpulist[self.gpupos%self.ngpu]):
            self.nparam=1
            self.learndata={}
            self.param={}
            self.pshape={}
            self.doreset={}
            self.logits={}
            
            if self.healpix==True:
                if len(interpol)>0:
                    limage=self.donaiveinterpolH(image,interpol)
                else:
                    limage=image
                    
                self.widx2={}
                self.wcos=tf.reshape(tf.transpose(self.wwc),[1,self.NORIENT,1,self.KERNELSZ**2])
                self.wsin=tf.reshape(tf.transpose(self.wws),[1,self.NORIENT,1,self.KERNELSZ**2])
                
                nout=int(np.sqrt(image.shape[0]//12))
                self.nout=nout
                nstep=int(np.log(nout)/np.log(2))-self.OSTEP
                if self.rank==0:
                    print('Initialize HEALPIX synthesis NSIDE=',nout)
                    sys.stdout.flush()
                self.ampnorm={}
                for i in range(nstep):
                    lout=nout//(2**i)
                    try:
                        tmp=np.load('%s/W%d_%d_IDX.npy'%(self.TEMPLATE_PATH,self.KERNELSZ**2,lout))
                    except:
                        if self.KERNELSZ**2==9:
                            if self.rank==0:
                                self.comp_idx_w9(max_nside=nout)
                        elif self.KERNELSZ**2==25:
                            if self.rank==0:
                                self.comp_idx_w25(max_nside=nout)
                        else:
                            if self.rank==0:
                                print('Only 3x3 and 5x5 kernel have been developped for Healpix and you ask for %dx%d'%(KERNELSZ,KERNELSZ))
                            exit(0)
                            
                        self.barrier()
                        tmp=np.load('%s/W%d_%d_IDX.npy'%(self.TEMPLATE_PATH,self.KERNELSZ**2,lout))

                    npt=tmp.shape[1]
                    self.widx2[lout]=tf.constant(tmp.flatten())

                    if (image1 is not None):
                        atmp=np.zeros([1,4,1])
                        if i==0:
                            tmp1=image1
                            tmp2=image2
                        else:
                            tmp1=np.mean((image1).reshape(12*nout*nout//4**(i),4**i),1)
                            tmp2=np.mean((image2).reshape(12*nout*nout//4**(i),4**i),1)
                
                        cs1=np.mean(tmp1[tmp]*(self.np_wwc.T).reshape(self.NORIENT,1,self.KERNELSZ**2),2)
                        ss1=np.mean(tmp1[tmp]*(self.np_wws.T).reshape(self.NORIENT,1,self.KERNELSZ**2),2)
                        cs2=np.mean(tmp2[tmp]*(self.np_wwc.T).reshape(self.NORIENT,1,self.KERNELSZ**2),2)
                        ss2=np.mean(tmp2[tmp]*(self.np_wws.T).reshape(self.NORIENT,1,self.KERNELSZ**2),2)

                        #import healpy as hp
                        
                        for k in range(self.NORIENT):
                            res=cs1[k]*cs2[k]+ss1[k]*ss2[k]
                            atmp[0,k,0]=(1/res.std())
                
                        if self.rank==0:
                            print('Scale ',i,atmp[0,:,0])
                            sys.stdout.flush()

                        self.ampnorm[i]=tf.constant(atmp,dtype=self.all_tf_type)
                    else:
                        self.ampnorm[i]=tf.constant(1.0,dtype=self.all_tf_type)
                    
                    
                self.learndata[0]=tf.constant(limage.astype(self.all_type).reshape(1,image.shape[0],1,1))
                self.pshape[0]=image.shape[0]
                self.param[0]=tf.Variable(0*image.astype(self.all_type).reshape(image.shape[0]))
                self.doreset[0]=self.param[0].assign(0*image.astype(self.all_type).reshape(image.shape[0]))
                self.logits[0] = self.learndata[0]-tf.reshape(self.param[0],[1,image.shape[0],1,1])
            else:
                limage=1*image
                if len(interpol)>0:
                    limage=self.donaiveinterpol(image,interpol,xpadding=xpadding)
                    
                self.learndata[0]=tf.constant(limage.astype(self.all_type).reshape(1,image.shape[0],image.shape[1],1))
                if xpadding:
                    self.pshape[0]=image.shape[0]*image.shape[1]//2
                    tmp=0*image[:,0:image.shape[1]//2].astype(self.all_type).reshape(image.shape[0]*image.shape[1]//2)
                    self.param[0]=tf.Variable(tmp)
                    self.doreset[0]=self.param[0].assign(tmp)
                    lpar=tf.reshape(self.param[0],[1,image.shape[0],image.shape[1]//2,1])
                    b = tf.constant([1,1,2,1], tf.int32)
                    lpar=tf.tile(lpar,b)
                    self.logits[0] = self.learndata[0]-lpar
                else:
                    self.pshape[0]=image.shape[0]*image.shape[1]
                    self.param[0]=tf.Variable(0*image.astype(self.all_type).reshape(image.shape[0]*image.shape[1]))
                    self.doreset[0]=self.param[0].assign(0*image.astype(self.all_type).reshape(image.shape[0]*image.shape[1]))
                    self.logits[0] = self.learndata[0]-tf.reshape(self.param[0],[1,image.shape[0],image.shape[1],1])
                

            self.inpar[0]=tf.placeholder(self.all_tf_type,shape=(self.pshape[0]))
            self.rewind[0]=self.param[0].assign(self.inpar[0])
            sim1=self.learndata[0].get_shape().as_list()

                
            if self.healpix==True:
                self.noise1=tf.compat.v1.placeholder(self.all_tf_type,
                                                     shape=(1,sim1[1],1,1),
                                                     name='NOISE1_%d'%(self.nloss))
                self.noise2=tf.compat.v1.placeholder(self.all_tf_type,
                                                     shape=(1,sim1[1],1,1),
                                                     name='NOISE2_%d'%(self.nloss))

                # Scattering covariance coefficient
                S1, P00, C01, C11 = self.hpwst_2_cov(self.noise1)
                self.S1 = S1
                self.P00 = P00
                self.C01 = C01
                self.C11 = C11
                for i in range(16):
                    if ((i//4)%2)+((i//8)%2)!=0:
                        on1,on2=self.hpwst1(self.noise1,
                                            self.noise2,
                                            imaginary=(i%2)==1,
                                            avg_ang=((i//2)%2)==1,
                                            doL1=((i//4)%2)==1,
                                            doL2=((i//8)%2)==1)
                        self.on1[i]=on1
                        self.on2[i]=on2
            else:
                self.noise1=tf.compat.v1.placeholder(self.all_tf_type,
                                                     shape=(1,sim1[1],sim1[2],1),
                                                     name='NOISE1_%d'%(self.nloss))
                self.noise2=tf.compat.v1.placeholder(self.all_tf_type,
                                                     shape=(1,sim1[1],sim1[2],1),
                                                     name='NOISE2_%d'%(self.nloss))

                for i in range(16):
                    if ((i//4)%2)+((i//8)%2)!=0:
                        on1,on2=self.cwst1(self.noise1,self.noise2,
                                           imaginary=(i%2)==1,
                                           avg_ang=((i//2)%2)==1,
                                           doL1=((i//4)%2)==1,
                                           doL2=((i//8)%2)==1)
                        self.on1[i]=on1
                        self.on2[i]=on2
                    
            return(self.logits[0])
        
    # ---------------------------------------------−---------   
    def add_image(self,image,xpadding=False,interpol=[]):
        with tf.device(self.gpulist[self.gpupos%self.ngpu]):
            
            if self.healpix==True:
                if len(interpol)>0:
                    limage=self.donaiveinterpolH(image,interpol)
                else:
                    limage=image
                self.learndata[self.nparam]=tf.constant(limage.astype(self.all_type).reshape(1,image.shape[0],1,1))
                self.pshape[self.nparam]=image.shape[0]
                self.param[self.nparam]=tf.Variable(0*image.astype(self.all_type).reshape(image.shape[0]))
                self.doreset[self.nparam]=self.param[self.nparam].assign(0*image.astype(self.all_type).reshape(image.shape[0]))
                self.logits[self.nparam] = self.learndata[self.nparam]-tf.reshape(self.param[self.nparam],[1,image.shape[0],1,1])
            else:
                limage=1*image
                if len(interpol)>0:
                    limage=self.donaiveinterpol(image,interpol,xpadding=xpadding)
                self.learndata[self.nparam]=tf.constant(limage.astype(self.all_type).reshape(1,image.shape[0],image.shape[1],1))
                if xpadding:
                    self.pshape[self.nparam]=image.shape[0]*image.shape[1]//2
                    tmp=0*image[:,0:image.shape[1]//2].astype(self.all_type).reshape(image.shape[0]*image.shape[1]//2)
                    self.param[self.nparam]=tf.Variable(tmp)
                    self.doreset[self.nparam]=self.param[self.nparam].assign(tmp)
                    lpar=tf.reshape(self.param[self.nparam],[1,image.shape[0],image.shape[1]//2,1])
                    b = tf.constant([1,1,2,1], tf.int32)
                    lpar=tf.tile(lpar,b)
                    self.logits[self.nparam] = self.learndata[self.nparam]-lpar
                else:
                    self.pshape[self.nparam]=image.shape[0]*image.shape[1]
                    self.param[self.nparam]=tf.Variable(0*image.astype(self.all_type).reshape(image.shape[0]*image.shape[1]))
                    self.doreset[self.nparam]=self.param[self.nparam].assign(0*image.astype(self.all_type).reshape(image.shape[0]*image.shape[1]))
                    self.logits[self.nparam] = self.learndata[self.nparam]-tf.reshape(self.param[self.nparam],[1,image.shape[0],image.shape[1],1])
                

            self.inpar[self.nparam]=tf.placeholder(self.all_tf_type,shape=(self.pshape[self.nparam]))
            self.rewind[self.nparam]=self.param[self.nparam].assign(self.inpar[self.nparam])
            self.nparam=self.nparam+1
            
            return(self.logits[self.nparam-1])
    # ---------------------------------------------−---------
    def init_optim(self):
        
        LEARNING_RATE = 0.03
        with tf.device(self.gpulist[self.gpupos%self.ngpu]):
            self.Tloss={} 
            self.Topti={} 
            self.nvarl={}
            self.nvar=0.0
            
            for i in range(self.nloss):
                self.nvarl[i]=1.0
                
                if self.loss_type[i]==self.SCATDIFF:
                    self.Tloss[i]=self.loss_weight[i]*(tf.reduce_sum(tf.square(self.tw1[i]*(self.os1[i] - self.is1[i] -self.tb1[i]))) + tf.reduce_sum(tf.square(self.tw2[i]*(self.os2[i]- self.is2[i] -self.tb2[i]))))
                    tshape=self.os1[i].get_shape().as_list()
                    for j in range(len(tshape)):
                        self.nvarl[i]=self.nvarl[i]*tshape[j]
                    self.nvar=self.nvar+self.nvarl[i]

                if self.loss_type[i]==self.SCATCOV:
                    self.Tloss[i]=self.loss_weight[i]*(tf.reduce_sum(tf.square(self.tw1[i]*(self.os1[i] - self.is1[i] -self.tb1[i]))) +
                                                       tf.reduce_sum(tf.square(self.tw2[i]*(self.os2[i] - self.is2[i] -self.tb2[i]))) +
                                                       tf.reduce_sum(tf.square(self.tw3[i]*(self.os3[i] - self.is3[i] -self.tb3[i]))) +
                                                       tf.reduce_sum(tf.square(self.tw4[i]*(self.os4[i] - self.is4[i] -self.tb4[i])))
                                                       )
                    tshape=self.os1[i].get_shape().as_list()
                    for j in range(len(tshape)):
                        self.nvarl[i]=self.nvarl[i]*tshape[j]
                    self.nvar=self.nvar+self.nvarl[i]
                
                if self.loss_type[i]==self.MAPDIFF:
                    self.Tloss[i]=self.diff_weight[i]*tf.reduce_sum(tf.square((self.diff_map1[i]-self.diff_map2[i])*self.diff_mask[i]))
                    self.nvarl[i]=1.0
                    
                if self.loss_type[i]==self.NOISESTAT:
                    self.Tloss[i]=self.diff_weight[i]*tf.reduce_sum(tf.square(self.diff_map1[i]-self.diff_map2[i]))
                    self.nvarl[i]=self.NMASK

                if i==0:
                    self.loss = self.Tloss[i]
                else:
                    self.loss = self.loss+self.Tloss[i]
                
            self.learning_rate = tf.compat.v1.placeholder_with_default(tf.constant(LEARNING_RATE),shape=())
            self.numbatch = tf.Variable(0, dtype=self.all_tf_type)
        
            # Use simple momentum for the optimization.
            opti=tf.train.AdamOptimizer(self.learning_rate)

            # if mpi is up compute gradient for each node
            if len(self.Tloss)>0:
                if self.size>1:
                    self.igrad={}
                    self.gradient={}
                    self.tgradient={}
                    self.apply_grad={}
                    for k in range(self.nparam):
                        self.igrad[k]    = tf.placeholder(self.all_tf_type,shape=(self.pshape[k]))
                        self.gradient[k] = opti.compute_gradients(self.loss,var_list=[self.param[k]])[0]
                        self.tgradient[k] = [(self.igrad[k],self.gradient[k][1])]
                        self.apply_grad[k] = opti.apply_gradients(self.tgradient[k],global_step=self.numbatch)

                    tlin=np.zeros([self.size+1],dtype=self.all_type)
                    tlout=np.zeros([self.size+1],dtype=self.all_type)
                    tlin[0]=self.nvar
                    tlin[1+self.rank]=self.nvarl[0]
                    self.comm.Allreduce((tlin,self.MPI_ALL_TYPE),(tlout,self.MPI_ALL_TYPE))
                    for i in range(self.size):
                        self.nvarl[i]=tlout[1+i]
                    self.nvar=tlout[0]
                else:
                    self.optimizer=opti.minimize(self.loss,global_step=self.numbatch)
                    for i in self.Tloss:
                        self.Topti[i]=opti.minimize(self.Tloss[i],global_step=self.numbatch)
            else:
                 print('NO LOSS DEFINED: this foscat session only computes coeficients, learn operation will crash')
                 self.loss=tf.constant(0)
                
            self.sess=tf.Session()

            tf.global_variables_initializer().run(session=self.sess)

            if self.rank==0:
                print('Initialized!')
                sys.stdout.flush()
        
        return(self.loss)
    # ---------------------------------------------−---------
    def learn(self,iw1,iw2,ib1,ib2,
              NUM_EPOCHS = 1000,
              DECAY_RATE=0.95,
              EVAL_FREQUENCY = 100,
              DEVAL_STAT_FREQUENCY = 1000,
              LEARNING_RATE = 0.03,
              ACURACY = 1E16,
              gradmask=[],
              ADDAPT_LEARN=1E30,
              SEQUENTIAL_ITT=0,
              IW3=None,
              IW4=None,
              IB3=None,
              IB4=None):

        if len(gradmask)==0:
            gradmask=[True for i in range(self.nparam)]

        minloss=1E300
        with tf.device(self.gpulist[self.gpupos%self.ngpu]):
            feed_dict={}
            for i in range(self.nloss):
                if self.loss_type[i]==self.SCATDIFF:
                    feed_dict[self.tw1[i]]=iw1[i]
                    feed_dict[self.tw2[i]]=iw2[i]
                    feed_dict[self.tb1[i]]=ib1[i]
                    feed_dict[self.tb2[i]]=ib2[i]
                if self.loss_type[i]==self.SCATCOV:
                    feed_dict[self.tw1[i]]=iw1[i]
                    feed_dict[self.tw2[i]]=iw2[i]
                    feed_dict[self.tw3[i]]=IW3[i]
                    feed_dict[self.tw4[i]]=IW4[i]
                    feed_dict[self.tb1[i]]=ib1[i]
                    feed_dict[self.tb2[i]]=ib2[i]
                    feed_dict[self.tb3[i]]=IB3[i]
                    feed_dict[self.tb4[i]]=IB4[i]
                    
            feed_dict[self.learning_rate]=LEARNING_RATE

            start_time = time.time()
            tl,l,lr=self.sess.run([self.Tloss,self.loss,self.learning_rate],
                                  feed_dict=feed_dict)
            if self.size>1:
                tlin=np.zeros([1],dtype=self.all_type)
                tlout=np.zeros([1],dtype=self.all_type)
                tlin[0]=l
                self.comm.Allreduce((tlin,self.MPI_ALL_TYPE),(tlout,self.MPI_ALL_TYPE))
                l=tlout[0]
            if l==0:
                l=1E300
            lstart=l
            step=0
            lpar={}
            #initialize omap
            for i in range(self.nparam):
                            
                lmap=self.sess.run([self.logits[i]])[0]
                lsize=lmap.shape
                if self.nout==-1:
                    lmap=lmap.reshape(lsize[1],lsize[2])
                    if i==0:
                        if self.nparam>1:
                            omap=lmap.reshape(lsize[1],lsize[2],1)
                        else:
                            omap=lmap
                    else:
                        omap=np.concatenate((omap,lmap.reshape(lsize[1],lsize[2],1)),2)
                else:
                    lmap=lmap.flatten()
                    if i==0:
                        if self.nparam>1:
                            omap=lmap.reshape(lmap.shape[0],1)
                        else:
                            omap=lmap
                    else:
                        omap=np.concatenate((omap,lmap.reshape(lmap.shape[0],1)),1)

            while step<NUM_EPOCHS and l*ACURACY>lstart:
                    
                
                if step%EVAL_FREQUENCY==0 or step==NUM_EPOCHS-1:
                
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    tl,l,lr=self.sess.run([self.Tloss,self.loss,self.learning_rate],
                                          feed_dict=feed_dict)
                    if self.size>1:
                        tlin=np.zeros([self.size+1],dtype=self.all_type)
                        tlout=np.zeros([self.size+1],dtype=self.all_type)
                        if len(tl)==1:
                            tlin[0]=l
                        else:
                            tlin[0]=l
                        if self.nloss==1:
                            tlin[1+self.rank]=tl[0]
                        else:
                            tlin[1+self.rank]=l
                            
                        self.comm.Allreduce((tlin,self.MPI_ALL_TYPE),(tlout,self.MPI_ALL_TYPE))
                        tl=tlout[1:1+self.size]
                        l=tlout[0]
                        
                    if l<minloss:
                        for i in range(self.nparam):
                            lmap=self.sess.run([self.param[i]])[0]
                            lpar[i]=lmap
                            
                            lmap=self.sess.run([self.logits[i]])[0]
                            lsize=lmap.shape
                            if self.nout==-1:
                                lmap=lmap.reshape(lsize[1],lsize[2])
                                if i==0:
                                    if self.nparam>1:
                                        omap=lmap.reshape(lsize[1],lsize[2],1)
                                    else:
                                        omap=lmap
                                else:
                                    omap=np.concatenate((omap,lmap.reshape(lsize[1],lsize[2],1)),2)
                            else:
                                lmap=lmap.flatten()
                                if i==0:
                                    if self.nparam>1:
                                        omap=lmap.reshape(lmap.shape[0],1)
                                    else:
                                        omap=lmap
                                else:
                                    omap=np.concatenate((omap,lmap.reshape(lmap.shape[0],1)),1)
                                    
                                
                        minloss=l
                    if l>ADDAPT_LEARN*minloss and len(lpar)>0:
                        lfeed={}
                        for i in range(self.nparam):
                            lfeed[self.inpar[i]]=lpar[i]
                            
                        self.sess.run(self.rewind, feed_dict=lfeed)
                        feed_dict[self.learning_rate]=feed_dict[self.learning_rate]/10.0
                            
                    if self.size>1:
                        losstab=''
                        for i in range(self.size):
                            losstab=losstab+'%9.4lg'%(np.sqrt(tl[i]))
                            if i<self.size-1:
                                losstab=losstab+','
                    else:
                        losstab=''
                        for i in range(self.nloss):
                            losstab=losstab+'%9.4lg'%(np.sqrt(tl[i]))
                            if i<self.nloss-1:
                                losstab=losstab+','
                                
                    if self.rank==0:
                        print('STEP %d mLoss=%9.3g Loss=%9.3lg(%s) Lr=%.3lg DT=%4.2fs'%(step,
                                                                                      np.sqrt(minloss),
                                                                                      np.sqrt(l),
                                                                                      losstab,
                                                                                      lr,
                                                                                      elapsed_time))
                        sys.stdout.flush()
                        if self.nlog==self.log.shape[0]:
                            new_log=np.zeros([self.log.shape[0]*2])
                            new_log[0:self.nlog]=self.log
                            self.log=new_log
                        self.log[self.nlog]=l
                        self.nlog=self.nlog+1
                        
                if self.size==1:
                    if step>=SEQUENTIAL_ITT:
                        self.sess.run(self.optimizer,feed_dict=feed_dict)
                    else:
                        for i in self.Tloss:
                            self.sess.run(self.Topti[i],feed_dict=feed_dict)
                else:
                    #====================================================================
                    # compute the gradients1
                    for k in range(self.nparam):
                        if gradmask[k]:
                            lgrad=self.sess.run([self.gradient[k]],feed_dict=feed_dict)[0]
                            if 'dense_shape' in dir(lgrad[0]):
                                vgrad = np.bincount(lgrad[0][1],weights=lgrad[0][0],minlength=lgrad[0][2][0]).astype(all_type)
                            else:
                                vgrad = np.array(lgrad)[0]
                        else:
                            vgrad=np.zeros([self.pshape[k]])
                            
                        feed_dict_grad={}
                        if step>=SEQUENTIAL_ITT:
                            tsgrad=np.zeros([self.pshape[k]])
                            self.comm.Allreduce((vgrad,self.MPI_ALL_TYPE),(tsgrad,self.MPI_ALL_TYPE))
                            feed_dict_grad[self.igrad[k]]=tsgrad
                        else:
                            self.comm.Bcast((vgrad,self.MPI_ALL_TYPE),step%self.size)
                            feed_dict_grad[self.igrad[k]]=vgrad
                        # apply the gradients1
                        self.sess.run(self.apply_grad[k], feed_dict=feed_dict_grad)
                
                feed_dict[self.learning_rate]=feed_dict[self.learning_rate]*DECAY_RATE
                step=step+1
        return(omap)

    # ---------------------------------------------−---------
    def get_log(self):
        return(self.log[0:self.nlog])
    
    # ---------------------------------------------−---------
    def get_map(self,idx=0):
                
        l=self.sess.run([self.logits[idx]])[0]
        lsize=l.shape
        if self.nout==-1:
            l=l.reshape(lsize[1],lsize[2])
        else:
            l=l.flatten()
        return(l)
    
    # ---------------------------------------------−---------
    def get_param(self,idx=0):
                
        l=self.sess.run([self.param[idx]])[0]
        return(l)
        
