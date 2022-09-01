import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os, sys
import time

class FoCUS:

    def __init__(self,NORIENT=4,LAMBDA=1.2,KERNELSZ=3,slope=2.0,
                 all_type='float64',padding='SAME',gpupos=0,healpix=False,
                 OSTEP=-1):
        self.nloss=0
        self.padding=padding
        self.healpix=healpix
        self.OSTEP=OSTEP
        
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
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        sys.stdout.flush()
        tf.debugging.set_log_device_placement(False)
        tf.config.set_soft_device_placement(True)
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                gpuname=logical_gpus[gpupos].name
                self.gpulist={}
                self.ngpu=len(logical_gpus)
                for i in range(self.ngpu):
                    self.gpulist[i]=logical_gpus[i].name
                    
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
                gpuname='CPU:0'
                self.gpulist={}
                self.gpulist[0]=gpuname
                self.ngpu=1
                
        self.gpupos=gpupos%self.ngpu
        print('============================================================')
        print('==                                                        ==')
        print('==                                                        ==')
        print('==     RUN ON GPU : %s                 =='%(self.gpulist[self.gpupos%self.ngpu]))
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
    def add_mask(self,mask):
        
        for i in range(1,mask.shape[0]):
            mask[i,:]=mask[i]*mask[0].sum()/mask[i].sum()
            
        if self.healpix:
            self.mask=tf.reshape(tf.constant(mask.astype(self.all_type)),[mask.shape[0],mask.shape[1],1,1])
            self.NMASK=mask.shape[0]
        else:
            self.mask=tf.reshape(tf.constant(mask.astype(self.all_type)),[mask.shape[0],mask.shape[1],mask.shape[2],1])
            self.NMASK=mask.shape[0]
        print('Use %d masks'%(self.NMASK))
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

            self.nloss=self.nloss+1

            return self.nloss-1
    # ---------------------------------------------−---------
    def add_loss_healpix(self,image1,image2,image3,image4,avg_ang=False,imaginary=False):
        
        return self.add_loss(image1,image2,image3,image4,avg_ang=avg_ang,imaginary=imaginary)
    # ---------------------------------------------−---------  
    def add_loss_2d(self,image1,image2,image3,image4,avg_ang=False,imaginary=False):
        
        return self.add_loss(image1,image2,image3,image4,avg_ang=avg_ang,imaginary=imaginary)
    # ---------------------------------------------−---------
    def calc_stat(self,n1,n2,imaginary=False,gpupos=0,avg_ang=False):
        
        with tf.device(self.gpulist[gpupos]):
            nsim=n1.shape[0]
            nx=n1.shape[1]
            for i in range(nsim):
                feed_dict={}
                if self.nout!=-1:
                    feed_dict[self.noise1]=n1[i].reshape(1,12*self.nout*self.nout,1,1)
                    feed_dict[self.noise2]=n2[i].reshape(1,12*self.nout*self.nout,1,1)
                else:
                    feed_dict[self.noise1]=n1[i].reshape(1,nx,nx,1)
                    feed_dict[self.noise2]=n2[i].reshape(1,nx,nx,1)

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
        with tf.device(self.gpulist[self.gpupos]):
            if self.healpix:
                return(self.convhealpix(image))
            
            return(tf.constant((image.astype(self.all_type)).reshape(1,image.shape[0],image.shape[1],1)))
    def convhealpix(self,image):
        with tf.device(self.gpulist[self.gpupos]):
            return(tf.constant((image.astype(self.all_type)).reshape(1,image.shape[0],1,1)))
    # ---------------------------------------------−---------
    def reset(self):
        self.sess.run(self.doreset)
    # ---------------------------------------------−---------   
    def init_synthese(self,image):
        self.nout=-1
        with tf.device(self.gpulist[self.gpupos]):
            if self.healpix==True:
                self.widx2={}
                self.wcos={}
                self.wsin={}
                nout=int(np.sqrt(image.shape[0]//12))
                self.nout=nout
                nstep=int(np.log(nout)/np.log(2))-self.OSTEP
                print('Initialize HEALPIX synthesis NSIDE=',nout)
                for i in range(nstep):
                    lout=nout//(2**i)
                    tmp=np.load('../data/W%d_%d_IDX_FINAL.npy'%(self.KERNELSZ**2,lout))
                    npt=tmp.shape[1]
                    self.widx2[lout]=tf.constant(tmp.flatten())
                    self.wcos[lout]=tf.reshape(tf.transpose(self.wwc),[1,self.NORIENT,1,npt])
                    self.wsin[lout]=tf.reshape(tf.transpose(self.wws),[1,self.NORIENT,1,npt])

                self.learndata=tf.constant(image.astype(self.all_type).reshape(1,image.shape[0],1,1))
                self.param=tf.Variable(0*image.astype(self.all_type).reshape(1,image.shape[0],1,1))
                self.doreset=self.param.assign(0*image.astype(self.all_type).reshape(1,image.shape[0],1,1))
            else:
                self.learndata=tf.constant(image.astype(self.all_type).reshape(1,image.shape[0],image.shape[1],1))
                self.param=tf.Variable(0*image.astype(self.all_type).reshape(1,image.shape[0],image.shape[1],1))
                self.doreset=self.param.assign(0*image.astype(self.all_type).reshape(1,image.shape[0],image.shape[1],1))

            self.logits = self.learndata-self.param

            sim1=self.learndata.get_shape().as_list()

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
            return(self.logits)
    # ---------------------------------------------−---------
    def init_optim(self):
        
        LEARNING_RATE = 0.03
        with tf.device(self.gpulist[self.gpupos]):
            self.Tloss={} 
            self.loss=tf.constant(0,self.all_tf_type)
            self.nvarl={}
            self.nvar=0.0
            for i in range(self.nloss):
                self.nvarl[i]=1.0
                self.Tloss[i]=tf.reduce_sum(tf.square(self.tw1[i]*(self.os1[i] - self.is1[i] -self.tb1[i]))) + tf.reduce_sum(tf.square(self.tw2[i]*(self.os2[i]- self.is2[i] -self.tb2[i])))
                
                self.loss=self.loss+self.Tloss[i]
                tshape=self.os1[i].get_shape().as_list()
                for j in range(len(tshape)):
                    self.nvarl[i]=self.nvarl[i]*tshape[j]
                self.nvar=self.nvar+self.nvarl[i]
            self.learning_rate = tf.compat.v1.placeholder_with_default(tf.constant(LEARNING_RATE),shape=())
            self.numbatch = tf.Variable(0, dtype=self.all_tf_type)
        
            # Use simple momentum for the optimization.
            opti=tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer=opti.minimize(self.loss,global_step=self.numbatch)
        
            self.sess=tf.Session()

            tf.global_variables_initializer().run(session=self.sess)

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
              ACURACY = 1E16):

        minloss=1E30
        with tf.device(self.gpulist[self.gpupos]):
            feed_dict={}
            for i in range(self.nloss):
                feed_dict[self.tw1[i]]=iw1[i]
                feed_dict[self.tw2[i]]=iw2[i]
                feed_dict[self.tb1[i]]=ib1[i]
                feed_dict[self.tb2[i]]=ib2[i]
            feed_dict[self.learning_rate]=LEARNING_RATE

            start_time = time.time()
            tl,l,lr=self.sess.run([self.Tloss,self.loss,self.learning_rate],
                                  feed_dict=feed_dict)
            lstart=l
            step=0
            while step<NUM_EPOCHS and l*ACURACY>lstart:
                    
                
                if step%EVAL_FREQUENCY==0:
                
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    tl,l,lr=self.sess.run([self.Tloss,self.loss,self.learning_rate],
                                          feed_dict=feed_dict)
                    if l<minloss:
                        omap=self.sess.run([self.logits])[0]
                        lsize=omap.shape
                        if self.nout==-1:
                            omap=omap.reshape(lsize[1],lsize[2])
                        else:
                            omap=omap.flatten()
                        minloss=l
                            
                    losstab=''
                    for i in range(self.nloss):
                        losstab=losstab+'%.3lg'%(tl[i]/self.nvarl[i])
                        if i<self.nloss-1:
                            losstab=losstab+','
                    print('STEP %d mLoss=%.3g Loss=%.3lg(%s) Lr=%.3lg DT=%4.2fs'%(step,
                                                                                  minloss/self.nvar,
                                                                           l/self.nvar,
                                                                           losstab,
                                                                           lr,
                                                                           elapsed_time))
                    sys.stdout.flush()
                self.sess.run(self.optimizer,feed_dict=feed_dict)
                feed_dict[self.learning_rate]=feed_dict[self.learning_rate]*DECAY_RATE
                step=step+1
        return(omap)

    # ---------------------------------------------−---------
    def get_map(self):
                
        l=self.sess.run([self.logits])[0]
        lsize=l.shape
        if self.nout==-1:
            l=l.reshape(lsize[1],lsize[2])
        else:
            l=l.flatten()
        return(l)
        
