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
                 slope=2.0,
                 all_type='float64',
                 padding='SAME',
                 gpupos=0,
                 healpix=False,
                 OSTEP=0,
                 isMPI=False,
                 TEMPLATE_PATH='data'):

        self.TEMPLATE_PATH=TEMPLATE_PATH
        
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
        self.MAPDIFF=1
        self.SCATDIFF=2

        self.log=np.zeros([10])
        self.nlog=0
        
        self.padding=padding
        self.healpix=healpix
        self.OSTEP=OSTEP
        self.nparam=0

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
        self.tb1={}
        self.tb2={}

        self.ss1={}
        self.ss2={}
        
        self.os1={}
        self.os2={}
        self.is1={}
        self.is2={}
        
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
    def barrier(self):
        if self.isMPI:
            self.comm.Barrier()
            
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
            print((c[:,i]**2+s[:,i]**2).sum(),(c[:,i]**2).sum(),c[:,i].sum(),s[:,i].sum())
            sys.stdout.flush()
        plt.show()
    # ---------------------------------------------−---------
    def relu(self,x):
        return tf.nn.relu(x)
    # ---------------------------------------------−---------
    def hpwst_2(self,image1,mask,doL1=True):
            
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
            vnorm=tf.reshape(2.0/(tf.math.reduce_sum(vmask,1)),[self.NMASK,1,1])
            im_shape = lim1.get_shape().as_list()
            alim1=tf.reshape(tf.gather(lim1,self.widx2[n0],axis=1),[BATCH_SIZE*norient,1,12*n0*n0,npt])
            cconv1 = tf.reduce_sum(self.wcos[n0]*alim1,3)
            sconv1 = tf.reduce_sum(self.wsin[n0]*alim1,3)

            tconvc1=cconv1*cconv1+sconv1*sconv1
            tmp1=self.L1(tconvc1)
            l_shape=tconvc1.get_shape().as_list()

            if doL1:
                vals=vnorm*tf.math.reduce_sum(tf.reshape(vmask,[self.NMASK,1,1,12*n0*n0])*tf.reshape(tmp1,[1,l_shape[0],l_shape[1],l_shape[2]]),3)
                ts1=vscale*tf.reshape(vals,[BATCH_SIZE*self.NMASK,norient,norient])
                s1.append(ts1)
            else:
                valc=vnorm*tf.math.reduce_sum(tf.reshape(vmask,[self.NMASK,1,1,12*n0*n0])*tf.reshape(tconvc1,[1,l_shape[0],l_shape[1],l_shape[2]]),3)
                ts1=vscale*tf.reshape(valc,[BATCH_SIZE*self.NMASK,norient,norient])
                s1.append(ts1)
                

            lim1=tf.reshape(tf.nn.avg_pool(tf.reshape(lim1,[BATCH_SIZE*norient,12*n0*n0,1,1]),
                                           ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1],padding='SAME'),[BATCH_SIZE*norient,12*n0*n0//4])
                
            tshape=vmask.get_shape().as_list()
            vmask=tf.math.reduce_mean(tf.reshape(vmask,[self.NMASK,tshape[1]//4,4]),2) 
            n0=n0//2
            vscale=vscale*self.slope
        
        return(tf.concat(s1,2))

    # ---------------------------------------------−---------
    def hpwst1(self,image1,image2,doL1=True,imaginary=False,avg_ang=False):
            
        BATCH_SIZE=1
        im_shape = image1.get_shape().as_list()
        nout=int(im_shape[1])
        norient=self.NORIENT
            
        nstep=int(np.log(np.sqrt(nout/12))/np.log(2))-self.OSTEP
        lim1=image1
        lim2=image2
        tshape=self.mask.get_shape().as_list()
        
        vmask=self.mask

        n0=int(np.sqrt(nout/12))


        vscale=1.0
        all_nstep=0
        wshape=self.widx2[n0].get_shape().as_list()
        npt=wshape[0]//(12*n0*n0)
        
        for iscale in range(nstep):

            vnorm=tf.reshape(2.0/(tf.math.reduce_sum(vmask,1)),[self.NMASK,1,1])
            im_shape = lim1.get_shape().as_list()
            
            alim1=tf.reshape(tf.gather(lim1,self.widx2[n0],axis=1),[BATCH_SIZE,1,12*n0*n0,npt])
            alim2=tf.reshape(tf.gather(lim2,self.widx2[n0],axis=1),[BATCH_SIZE,1,12*n0*n0,npt])
            
            cconv1 = tf.reduce_sum(self.wcos[n0]*alim1,3)
            cconv2 = tf.reduce_sum(self.wcos[n0]*alim2,3)
            sconv1 = tf.reduce_sum(self.wsin[n0]*alim1,3)
            sconv2 = tf.reduce_sum(self.wsin[n0]*alim2,3)
                
            tconvc1=(cconv1*cconv2+sconv1*sconv2)
            tmpc1=tf.reshape(self.L1(tconvc1),[BATCH_SIZE*norient,12*n0*n0])
            
            l_shape=tconvc1.get_shape().as_list()
                
            if doL1:
                vals=vnorm*tf.math.reduce_sum(tf.reshape(vmask,[self.NMASK,1,1,12*n0*n0])*tf.reshape(tmpc1    ,[1,l_shape[0],l_shape[1],l_shape[2]]),3)
                ts1=vscale*tf.reshape(vals,[BATCH_SIZE*self.NMASK,1,norient])
                if iscale==0:
                    s1=ts1
                else:
                    s1=tf.concat([s1,ts1],2)
            else:
                valc=vnorm*tf.math.reduce_sum(tf.reshape(vmask,[self.NMASK,1,1,12*n0*n0])*tf.reshape(tconvc1,[1,l_shape[0],l_shape[1],l_shape[2]]),3)
                ts1=vscale*tf.reshape(valc,[BATCH_SIZE*self.NMASK,1,norient])
                if iscale==0:
                    s1=ts1
                else:
                    s1=tf.concat([s1,ts1],2)

            if imaginary:
                tconvs1=(cconv1*sconv2-sconv1*cconv2)
                tmps1=tf.reshape(self.L1(tconvs1),[BATCH_SIZE*norient,12*n0*n0])
                    
                if doL1:
                    vals=vnorm*tf.math.reduce_sum(tf.reshape(vmask,[self.NMASK,1,1,12*n0*n0])*tf.reshape(tmps1    ,[1,l_shape[0],l_shape[1],l_shape[2]]),3)
                    ts1=vscale*tf.reshape(vals,[BATCH_SIZE*self.NMASK,1,norient])
                    if iscale==0:
                        c1=ts1
                    else:
                        c1=tf.concat([c1,ts1],2)
                else:
                    valc=vnorm*tf.math.reduce_sum(tf.reshape(vmask,[NMASK,1,1,12*n0*n0])*tf.reshape(tconvs1,[1,l_shape[0],l_shape[1],l_shape[2]]),3)
                    ts1=vscale*tf.reshape(valc,[BATCH_SIZE*self.NMASK,1,norient])
                    if iscale==0:
                        c1=ts1
                    else:
                        c1=tf.concat([c1,ts1],2)

            if iscale<nstep-1:
                val2c=self.hpwst_2(self.relu(tmpc1),vmask,doL1=doL1)-self.hpwst_2(self.relu(-tmpc1),vmask,doL1=doL1)
                ts2= vscale*val2c
                if iscale==0:
                    s2=ts2
                else:
                    s2=tf.concat([s2,ts2],2)
                if imaginary:
                    val2s=self.hpwst_2(self.relu(tmps1),vmask,doL1=doL1)-self.hpwst_2(self.relu(-tmps1),vmask,doL1=doL1)
                
                    ts2= vscale*val2s
                    if iscale==0:
                        c2=ts2
                    else:
                        c2=tf.concat([c2,ts2],2)

            lim1=tf.nn.avg_pool(lim1, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1],padding='SAME')
            lim2=tf.nn.avg_pool(lim2, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1],padding='SAME')
                
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
    def cwst_2(self,image1,mask):

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
            
            tconvc1=convc1*convc1+convs1*convs1
            tconvc1=self.L1(tconvc1)
            valc=tf.math.reduce_sum(tf.reshape(tf.reshape(vmask,[BATCH_SIZE,1,self.NMASK,n0,n1,1])*
                                               tf.reshape(tconvc1,[BATCH_SIZE,self.NORIENT,1,n0,n1,self.NORIENT]),
                                               [BATCH_SIZE*self.NMASK*self.NORIENT,n0*n1,self.NORIENT]),1)
            
            tshape=valc.get_shape().as_list()
            
            ts1=vscale*tf.reshape(valc,[BATCH_SIZE*self.NMASK,self.NORIENT,self.NORIENT])
            if iscale==0:
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
    def cwst1(self,image1,image2,imaginary=False,avg_ang=False):

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
            
            tconvc1=convc1*convc2+convs1*convs2
            tconvc1=self.L1(tconvc1)
            valc=tf.math.reduce_sum(tf.reshape(tf.reshape(vmask,[BATCH_SIZE,self.NMASK,n0,n1,1])*
                                               tf.reshape(tconvc1,[BATCH_SIZE,1,n0,n1,norient]),
                                               [BATCH_SIZE*self.NMASK,n0*n1,norient]),1)
            tshape=valc.get_shape().as_list()
            
            ts1=vscale*tf.reshape(valc,[BATCH_SIZE*self.NMASK,1,norient])
            if iscale==0:
                s1=ts1
            else:
                s1=tf.concat([s1,ts1],2)

            if imaginary:
                tconvs1=convc1*convs2-convc2*convs1
                tconvs1=self.L1(tconvs1)
                vals=tf.math.reduce_sum(tf.reshape(tf.reshape(vmask,[BATCH_SIZE,self.NMASK,n0,n1,1])*
                                                   tf.reshape(tconvs1,[BATCH_SIZE,1,n0,n1,norient]),
                                                   [BATCH_SIZE*self.NMASK,n0*n1,norient]),1)
                ts1=vscale*tf.reshape(vals,[BATCH_SIZE*self.NMASK,1,norient])
                if iscale==0:
                    c1=ts1
                else:
                    c1=tf.concat([c1,ts1],2)

            if iscale<nstep-1:
                val2c=self.cwst_2(self.relu(tconvc1),vmask)-self.cwst_2(self.relu(-tconvc1),vmask)
            
                tshape2=val2c.get_shape().as_list()
                
                ts2= vscale*val2c
                if iscale==0:
                    s2=ts2
                else:
                    s2=tf.concat([s2,ts2],2)
                
                if imaginary:
                    val2s=self.cwst_2(self.relu(tconvs1),vmask)-self.cwst_2(self.relu(-tconvs1),vmask)
                    
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
    def add_loss(self,image1,image2,image3,image4,doL1=True,imaginary=False,avg_ang=False):

        with tf.device(self.gpulist[self.nloss%self.ngpu]):
            if self.nout!=-1:
                os1,os2=self.hpwst1(image1,image2,doL1=doL1,imaginary=imaginary,avg_ang=avg_ang)
                is1,is2=self.hpwst1(image3,image4,doL1=doL1,imaginary=imaginary,avg_ang=avg_ang)
            else:
                os1,os2=self.cwst1(image1,image2,imaginary=imaginary,avg_ang=avg_ang)
                is1,is2=self.cwst1(image3,image4,imaginary=imaginary,avg_ang=avg_ang)

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

            self.nloss=self.nloss+1

    # ---------------------------------------------−---------
    def add_loss_healpix(self,image1,image2,image3,image4,avg_ang=False,imaginary=False):
        
        self.add_loss(image1,image2,image3,image4,avg_ang=avg_ang,imaginary=imaginary)
    # ---------------------------------------------−---------  
    def add_loss_2d(self,image1,image2,image3,image4,avg_ang=False,imaginary=False):
        
        self.add_loss(image1,image2,image3,image4,avg_ang=avg_ang,imaginary=imaginary)
    
    # ---------------------------------------------−---------  
    def add_loss_determ(self,image1,image2,doL1=True,avg_ang=False,imaginary=False):
        
        with tf.device(self.gpulist[self.nloss%self.ngpu]):
            if self.nout!=-1:
                os1,os2=self.hpwst1(image1,image2,doL1=self.doL1,imaginary=imaginary,avg_ang=avg_ang)
            else:
                os1,os2=self.cwst1(image1,image2,imaginary=imaginary,avg_ang=avg_ang)

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

            self.nloss=self.nloss+1
        
    # ---------------------------------------------−---------
    def calc_stat(self,n1,n2,imaginary=False,gpupos=0,avg_ang=False):
        
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

                if imaginary:
                    if avg_ang==False:
                        o1,o2= self.sess.run([self.oni1,self.oni2],feed_dict=feed_dict)
                    else:
                        o1,o2= self.sess.run([self.av_oni1,self.av_oni2],feed_dict=feed_dict)
                else:
                    if avg_ang==False:
                        o1,o2= self.sess.run([self.on1,self.on2],feed_dict=feed_dict)
                    else:
                        o1,o2= self.sess.run([self.av_on1,self.av_on2],feed_dict=feed_dict)

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
    def init_synthese(self,image,interpol=[],xpadding=False):
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
                self.wcos={}
                self.wsin={}
                nout=int(np.sqrt(image.shape[0]//12))
                self.nout=nout
                nstep=int(np.log(nout)/np.log(2))-self.OSTEP
                if self.rank==0:
                    print('Initialize HEALPIX synthesis NSIDE=',nout)
                    sys.stdout.flush()
                    
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
                    self.wcos[lout]=tf.reshape(tf.transpose(self.wwc),[1,self.NORIENT,1,npt])
                    self.wsin[lout]=tf.reshape(tf.transpose(self.wws),[1,self.NORIENT,1,npt])

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
                oni1,oni2=self.hpwst1(self.noise1,self.noise2,imaginary=True)
                on1,on2=self.hpwst1(self.noise1,self.noise2)
                av_oni1,av_oni2=self.hpwst1(self.noise1,self.noise2,imaginary=True,avg_ang=True)
                av_on1,av_on2=self.hpwst1(self.noise1,self.noise2,avg_ang=True)
            else:
                self.noise1=tf.compat.v1.placeholder(self.all_tf_type,
                                                     shape=(1,sim1[1],sim1[2],1),
                                                     name='NOISE1_%d'%(self.nloss))
                self.noise2=tf.compat.v1.placeholder(self.all_tf_type,
                                                     shape=(1,sim1[1],sim1[2],1),
                                                     name='NOISE2_%d'%(self.nloss))
                oni1,oni2=self.cwst1(self.noise1,self.noise2,imaginary=True)
                on1,on2=self.cwst1(self.noise1,self.noise2)
                av_oni1,av_oni2=self.cwst1(self.noise1,self.noise2,imaginary=True,avg_ang=True)
                av_on1,av_on2=self.cwst1(self.noise1,self.noise2,avg_ang=True)
            self.av_on1=av_on1
            self.av_on2=av_on2
            self.av_oni1=av_oni1
            self.av_oni2=av_oni2
            self.on1=on1
            self.on2=on2
            self.oni1=oni1
            self.oni2=oni2
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
            self.loss=tf.constant(0,self.all_tf_type)
            self.nvarl={}
            self.nvar=0.0
            
            for i in range(self.nloss):
                self.nvarl[i]=1.0
                
                if self.loss_type[i]==self.SCATDIFF:
                    self.Tloss[i]=tf.reduce_sum(tf.square(self.tw1[i]*(self.os1[i] - self.is1[i] -self.tb1[i]))) + tf.reduce_sum(tf.square(self.tw2[i]*(self.os2[i]- self.is2[i] -self.tb2[i])))
                    tshape=self.os1[i].get_shape().as_list()
                    for j in range(len(tshape)):
                        self.nvarl[i]=self.nvarl[i]*tshape[j]
                    self.nvar=self.nvar+self.nvarl[i]
                
                if self.loss_type[i]==self.MAPDIFF:
                    self.Tloss[i]=self.diff_weight[i]*tf.reduce_sum(tf.square((self.diff_map1[i]-self.diff_map2[i])*self.diff_mask[i]))
                    self.nvarl[i]=1.0
                
                self.loss=self.loss+self.Tloss[i]
                
            self.learning_rate = tf.compat.v1.placeholder_with_default(tf.constant(LEARNING_RATE),shape=())
            self.numbatch = tf.Variable(0, dtype=self.all_tf_type)
        
            # Use simple momentum for the optimization.
            opti=tf.train.AdamOptimizer(self.learning_rate)

            # if mpi is up compute gradient for each node
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
              ADDAPT_LEARN=1E30):

        if len(gradmask)==0:
            gradmask=[True for i in range(self.nparam)]

        minloss=1E30
        with tf.device(self.gpulist[self.gpupos%self.ngpu]):
            feed_dict={}
            for i in range(self.nloss):
                if self.loss_type[i]!=self.MAPDIFF:
                    feed_dict[self.tw1[i]]=iw1[i]
                    feed_dict[self.tw2[i]]=iw2[i]
                    feed_dict[self.tb1[i]]=ib1[i]
                    feed_dict[self.tb2[i]]=ib2[i]
                    
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
                l=1E30
            lstart=l
            step=0
                
            while step<NUM_EPOCHS and l*ACURACY>lstart:
                    
                
                if step%EVAL_FREQUENCY==0 or step==NUM_EPOCHS-1:
                
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    tl,l,lr=self.sess.run([self.Tloss,self.loss,self.learning_rate],
                                          feed_dict=feed_dict)
                    if self.size>1:
                        tlin=np.zeros([self.size+1],dtype=self.all_type)
                        tlout=np.zeros([self.size+1],dtype=self.all_type)
                        tlin[0]=l
                        if self.nloss==1:
                            tlin[1+self.rank]=tl[0]
                        self.comm.Allreduce((tlin,self.MPI_ALL_TYPE),(tlout,self.MPI_ALL_TYPE))
                        tl=tlout[1:1+self.size]
                        l=tlout[0]
                        
                    if l<minloss:
                        lpar={}
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
                    if l>ADDAPT_LEARN*minloss:
                        lfeed={}
                        for i in range(self.nparam):
                            lfeed[self.inpar[i]]=lpar[i]
                        self.sess.run(self.rewind, feed_dict=lfeed)
                        feed_dict[self.learning_rate]=feed_dict[self.learning_rate]/10.0
                            
                    if self.size>1:
                        losstab=''
                        for i in range(self.size):
                            losstab=losstab+'%9.3lg'%(tl[i]/self.nvarl[i])
                            if i<self.size-1:
                                losstab=losstab+','
                    else:
                        losstab=''
                        for i in range(self.nloss):
                            losstab=losstab+'%9.3lg'%(tl[i]/self.nvarl[i])
                            if i<self.nloss-1:
                                losstab=losstab+','
                                
                    if self.rank==0:
                        print('STEP %d mLoss=%9.3g Loss=%9.3lg(%s) Lr=%.3lg DT=%4.2fs'%(step,
                                                                                      minloss/self.nvar,
                                                                                      l/self.nvar,
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
                    self.sess.run(self.optimizer,feed_dict=feed_dict)
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
                            
                        tsgrad=np.zeros([self.pshape[k]])
                        self.comm.Allreduce((vgrad,self.MPI_ALL_TYPE),(tsgrad,self.MPI_ALL_TYPE))
                        # apply the gradients1
                        feed_dict_grad={}
                        feed_dict_grad[self.igrad[k]]=tsgrad
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
        
