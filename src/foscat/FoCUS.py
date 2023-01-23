import numpy as np
import healpy as hp
import os, sys

class FoCUS:
    def __init__(self,
                 NORIENT=4,
                 LAMBDA=1.2,
                 KERNELSZ=3,
                 slope=1.0,
                 all_type='float64',
                 nstep_max=16,
                 padding='SAME',
                 gpupos=0,
                 healpix=True,
                 OSTEP=0,
                 isMPI=False,
                 TEMPLATE_PATH='data',
                 BACKEND='tensorflow'):

        self.TENSORFLOW=1
        self.TORCH=2
        self.NUMPY=3
        
        if BACKEND=='tensorflow':
            import tensorflow as tf
            
            self.backend=tf
            self.BACKEND=self.TENSORFLOW
        if BACKEND=='torch':
            import torch
            self.BACKEND=self.TORCH
            self.backend=torch
            
        if BACKEND=='numpy':
            self.BACKEND=self.NUMPY
            self.backend=np
            
        self.float64=self.backend.float64
        self.float32=self.backend.float32
        self.int64=self.backend.int64
        self.int32=self.backend.int32
        
        print('================================================')
        print('          START FOSCAT CONFIGURATION')
        print('================================================')

        self.TEMPLATE_PATH=TEMPLATE_PATH
        if os.path.exists(self.TEMPLATE_PATH)==False:
            print('The directory %s to store temporary information for FoCUS does not exist: Try to create it'%(self.TEMPLATE_PATH))
            try:
                os.system('mkdir -p %s'%(self.TEMPLATE_PATH))
                print('The directory %s is created')
            except:
                print('Impossible to create the directory %s'%(self.TEMPLATE_PATH))
                exit(0)
                
        self.number_of_loss=0

        self.history=np.zeros([10])
        self.nlog=0
        
        self.padding=padding
        self.healpix=healpix
        self.OSTEP=OSTEP
        
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
        
        self.all_type=all_type
        
            
        if all_type=='float32':
            self.all_bk_type=self.backend.float32
        else:
            if all_type=='float64':
                self.all_type='float64'
                self.all_bk_type=self.backend.float64
            else:
                print('ERROR INIT FOCUS ',all_type,' should be float32 or float64')
                exit(0)
                
        #===========================================================================
        # INIT 
        if self.rank==0:
            if BACKEND=='tensorflow':
                print("Num GPUs Available: ", len(self.backend.config.experimental.list_physical_devices('GPU')))
            sys.stdout.flush()
        
        if BACKEND=='tensorflow':
            self.backend.debugging.set_log_device_placement(False)
            self.backend.config.set_soft_device_placement(True)
        
            gpus = self.backend.config.experimental.list_physical_devices('GPU')
            
        if BACKEND=='torch':
            gpus=torch.cuda.is_available()
            
        gpuname='CPU:0'
        self.gpulist={}
        self.gpulist[0]=gpuname
        self.ngpu=1
        
        if gpus:
            try:
                if BACKEND=='tensorflow':
                # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        self.backend.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = self.backend.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                    sys.stdout.flush()
                    gpuname=logical_gpus[gpupos].name
                    self.gpulist={}
                    self.ngpu=len(logical_gpus)
                    for i in range(self.ngpu):
                        self.gpulist[i]=logical_gpus[i].name
                if BACKEND=='torch':
                    self.ngpu=torch.cuda.device_count()
                    self.gpulist={}
                    for k in range(self.ngpu):
                        self.gpulist[k]=torch.cuda.get_device_name(0)

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
            xx=(3/float(KERNELSZ))*LAMBDA*(x*np.cos(a)+y*np.sin(a))
            yy=(3/float(KERNELSZ))*LAMBDA*(x*np.sin(a)-y*np.cos(a))

            if KERNELSZ==5:
                w_smooth=np.exp(-4*((3.0/float(KERNELSZ)*xx)**2+(3.0/float(KERNELSZ)*yy)**2))
            else:
                w_smooth=np.exp(-0.5*((3.0/float(KERNELSZ)*xx)**2+(3.0/float(KERNELSZ)*yy)**2))
        
            tmp=np.cos(yy*np.pi)*w_smooth
            wwc[:,i]=tmp.flatten()-tmp.mean()
            tmp=np.sin(yy*np.pi)*w_smooth
            wws[:,i]=tmp.flatten()-tmp.mean()
            sigma=np.sqrt((wwc[:,i]**2+wws[:,i]**2).mean())
            wwc[:,i]/=sigma
            wws[:,i]/=sigma

            w_smooth=w_smooth.flatten()
        
        self.w_smooth=(w_smooth/w_smooth.sum()).astype(self.all_type)
        self.ww_Real=wwc.astype(self.all_type)
        self.ww_Imag=wws.astype(self.all_type)
          
        self.Idx_Neighbours={}
        self.pix_interp_val={}
        self.weight_interp_val={}
        self.ring2nest={}
            
        self.ampnorm={}
        
        for i in range(nstep_max):
            lout=(2**i)
            self.pix_interp_val[lout]={}
            self.weight_interp_val[lout]={}
            for j in range(nstep_max):
                lout2=(2**j)
                self.pix_interp_val[lout][lout2]=None
                self.weight_interp_val[lout][lout2]=None
            self.ring2nest[lout]=None
            self.Idx_Neighbours[lout]=None

        self.loss={}
        
    # ---------------------------------------------−---------
    # --             BACKEND DEFINITION                    --
    # ---------------------------------------------−---------
    def bk_ones(self,shape,dtype=None):
        if dtype is None:
            dtype=self.all_type
        return(self.backend.ones(shape,dtype=dtype))

    def bk_L1(self,x):
        return self.backend.sign(x)*self.backend.sqrt(self.backend.sign(x)*x)
        
    def bk_reduce_sum(self,data,axis=None):
        if axis is None:
            if self.BACKEND==self.TENSORFLOW:
                return(self.backend.reduce_sum(data))
            if self.BACKEND==self.TORCH:
                return(self.backend.sum(data))
            if self.BACKEND==self.NUMPY:
                return(np.sum(data))
        else:
            if self.BACKEND==self.TENSORFLOW:
                return(self.backend.reduce_sum(data,axis=axis))
            if self.BACKEND==self.TORCH:
                return(self.backend.sum(data,axis))
            if self.BACKEND==self.NUMPY:
                return(np.sum(data,axis))
        
    def bk_reduce_mean(self,data,axis=None):
        if axis is None:
            if self.BACKEND==self.TENSORFLOW:
                return(self.backend.reduce_mean(data))
            if self.BACKEND==self.TORCH:
                return(self.backend.mean(data))
            if self.BACKEND==self.NUMPY:
                return(np.mean(data))
        else:
            if self.BACKEND==self.TENSORFLOW:
                return(self.backend.reduce_mean(data,axis=axis))
            if self.BACKEND==self.TORCH:
                return(self.backend.mean(data,axis))
            if self.BACKEND==self.NUMPY:
                return(np.mean(data,axis))
        
    def bk_sqrt(self,data):
        return(self.backend.sqrt(data))
    
    def bk_abs(self,data):
        return(self.backend.abs(data))
        
    def bk_square(self,data):
        if self.BACKEND==self.TENSORFLOW:
            return(self.backend.square(data))
        if self.BACKEND==self.TORCH:
            return(self.backend.square(data))
        if self.BACKEND==self.NUMPY:
            return(data*data)
        
    def bk_log(self,data):
        if self.BACKEND==self.TENSORFLOW:
            return(self.backend.math.log(data))
        if self.BACKEND==self.TORCH:
            return(self.backend.log(data))
        if self.BACKEND==self.NUMPY:
            return(np.log(data))
        
    def bk_complex(self,real,imag):
        if self.BACKEND==self.TENSORFLOW:
            return(self.backend.dtypes.complex(real,imag))
        if self.BACKEND==self.TORCH:
            return(self.backend.complex(real,imag))
        if self.BACKEND==self.NUMPY:
            return(np.complex(real,imag))

    def bk_gather(self,data,shape,axis=None):
        if self.BACKEND==self.TENSORFLOW:
            if axis is None:
                return(self.backend.gather(data,shape))
            else:
                return(self.backend.gather(data,shape,axis=axis))
        if self.BACKEND==self.TORCH:
            my_tensor = self.backend.LongTensor(shape)
            my_data = self.backend.Tensor(data)
            
            return(self.backend.gather(my_data,axis,my_tensor))
        
        if self.BACKEND==self.NUMPY:
            if axis is None:
                return(np.take(data,shape))
            else:
                return(np.take(data,shape,axis=axis))

    def bk_reshape(self,data,shape):
        return(self.backend.reshape(data,shape))

    def bk_expand_dims(self,data,axis=0):
        if self.BACKEND==self.TENSORFLOW:
            return(self.backend.expand_dims(data,axis=axis))
        if self.BACKEND==self.TORCH:
            return(self.backend.unsqueeze(data,axis))
        if self.BACKEND==self.NUMPY:
            return(np.expand_dims(data,axis))

    def bk_transpose(self,data,thelist):
        if self.BACKEND==self.TENSORFLOW:
            return(self.backend.transpose(data,thelist))
        if self.BACKEND==self.TORCH:
            return(self.backend.transpose(data,thelist))
        if self.BACKEND==self.NUMPY:
            return(np.transpose(data,thelist))

    def bk_concat(self,data,axis=None):
        if axis is None:
            return(self.backend.concat(data))
        else:
            return(self.backend.concat(data,axis=axis))
        
    def bk_relu(self,x):
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.nn.relu(x)
        if self.BACKEND==self.TORCH:
            return self.backend.relu(x)
        if self.BACKEND==self.NUMPY:
            return (x>0)*x
        
    def bk_cast(self,x):
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.cast(x,self.all_bk_type)
        if self.BACKEND==self.TORCH:
            return self.backend.cast(x,self.all_bk_type)
        if self.BACKEND==self.NUMPY:
            return x.astype(self.all_type)
        
    # ---------------------------------------------−---------
    # --       COMPUTE 3X3 INDEX FOR HEALPIX WORK          --
    # ---------------------------------------------−---------
    def corr_idx_wXX(self,x,y):
        idx=np.where(x==-1)[0]
        res=x
        res[idx]=y[idx]
        return(res)
    
    def comp_idx_w9(self,nout):
        
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
            
    # ---------------------------------------------−---------
    # --       COMPUTE 5X5 INDEX FOR HEALPIX WORK          --
    # ---------------------------------------------−---------
    def comp_idx_w25(self,nout):
        
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
    def toring(self,image,axis=0):
        lout=int(np.sqrt(image.shape[axis]//12))
        
        if self.ring2nest[lout] is None:
            self.ring2nest[lout]=hp.ring2nest(lout,np.arange(12*lout**2))
            
        return(self.bk_gather(image,self.ring2nest[lout],axis=axis))

    #--------------------------------------------------------
    def ud_grade_2(self,im,axis=0):

        shape=im.shape
        lout=int(np.sqrt(shape[axis]//12))
        if im.__class__==np.zeros([0]).__class__:
            oshape=np.zeros([len(shape)+1],dtype='int')
            if axis>0:
                oshape[0:axis]=shape[0:axis]
            oshape[axis]=12*lout*lout//4
            oshape[axis+1]=4
            if len(shape)>axis:
                oshape[axis+2:]=shape[axis+1:]
        else:
            if axis>0:
                oshape=shape[0:axis]+[12*lout*lout//4,4]
            else:
                oshape=[12*lout*lout//4,4]
            if len(shape)>axis:
                oshape=oshape+shape[axis+1:]
        
        return(self.bk_reduce_mean(self.bk_reshape(im,oshape),axis=axis+1))
    
    #--------------------------------------------------------
    def up_grade(self,im,nout,axis=0):
        lout=int(np.sqrt(im.shape[axis]//12))
        
        if self.pix_interp_val[lout][nout] is None:
            th,ph=hp.pix2ang(nout,np.arange(12*nout**2,dtype='int'),nest=True)
            p, w = hp.get_interp_weights(lout,th,ph,nest=True)
            del th
            del ph
            if self.BACKEND==self.TORCH:
                self.pix_interp_val[lout][nout]=p.astype('int64')
            else:
                self.pix_interp_val[lout][nout]=p
                
            self.weight_interp_val[lout][nout]=w.astype(self.all_type)
            
        if lout==nout:
            imout=im
        else:
            if axis==0:
                imout=self.bk_reduce_sum(self.bk_gather(im,self.pix_interp_val[lout][nout],axis=axis)\
                                    *self.weight_interp_val[lout][nout],axis=0)

            else:
                amap=self.bk_gather(im,self.pix_interp_val[lout][nout],axis=axis)
                aw=self.weight_interp_val[lout][nout]
                for k in range(axis):
                    aw=self.bk_expand_dims(aw, axis=0)
                for k in range(axis+1,len(im.shape)):
                    aw=self.bk_expand_dims(aw, axis=-1)
                    
                imout=self.bk_reduce_sum(aw*amap,axis=axis)
        return(imout)

    # ---------------------------------------------−---------
    def init_index(self,nside):
        try:
            tmp=np.load('%s/W%d_%d_IDX.npy'%(self.TEMPLATE_PATH,self.KERNELSZ**2,nside))
        except:
            if self.KERNELSZ**2==9:
                if self.rank==0:
                    self.comp_idx_w9(nside)
            elif self.KERNELSZ**2==25:
                if self.rank==0:
                    self.comp_idx_w25(nside)
            else:
                if self.rank==0:
                    print('Only 3x3 and 5x5 kernel have been developped for Healpix and you ask for %dx%d'%(KERNELSZ,KERNELSZ))
                    exit(0)
            self.barrier()
            tmp=np.load('%s/W%d_%d_IDX.npy'%(self.TEMPLATE_PATH,self.KERNELSZ**2,nside))
                    
        if self.BACKEND==self.TORCH:
            self.Idx_Neighbours[nside]=tmp.as_type('int64')
        else:
            self.Idx_Neighbours[nside]=tmp
        
    # ---------------------------------------------−---------
    # Compute x [....,a,....] to [....,a*a,....]
    #NOT YET TESTED OR IMPLEMENTED
    def auto_cross_2(x,axis=0):
        shape=np.array(x.shape)
        if axis==0:
            y1=self.reshape(x,[shape[0],1,np.cumprod(shape[1:])])
            y2=self.reshape(x,[1,shape[0],np.cumprod(shape[1:])])
            oshape=np.concat([shape[0],shape[0],shape[1:]])
            return(self.reshape(y1*y2,oshape))
    
    # ---------------------------------------------−---------
    # Compute x [....,a,....,b,....] to [....,b*b,....,a*a,....]
    #NOT YET TESTED OR IMPLEMENTED
    def auto_cross_2(x,axis1=0,axis2=1):
        shape=np.array(x.shape)
        if axis==0:
            y1=self.reshape(x,[shape[0],1,np.cumprod(shape[1:])])
            y2=self.reshape(x,[1,shape[0],np.cumprod(shape[1:])])
            oshape=np.concat([shape[0],shape[0],shape[1:]])
            return(self.reshape(y1*y2,oshape))
        
    
    # ---------------------------------------------−---------
    # convert swap axes tensor x [....,a,....,b,....] to [....,b,....,a,....]
    def swapaxes(self,x,axis1,axis2):
        shape=x.shape.as_list()
        if axis1<0:
            laxis1=len(shape)+axis1
        if axis2<0:
            laxis2=len(shape)+axis2
        
        naxes=len(shape)
        thelist=[i for i in range(naxes)]
        thelist[laxis1]=laxis2
        thelist[laxis2]=laxis1
        return self.transpose(x,thelist)
    
    # ---------------------------------------------−---------
    # convert tensor x [....,a,b,....] to [....,a*b,....]
    def reduce_dim(self,x,axis=0):
        shape=x.shape.as_list()
        if axis<0:
            laxis=len(shape)+axis
        else:
            laxis=axis
            
        if laxis>0 :
            oshape=shape[0:laxis]
            oshape.append(shape[laxis]*shape[laxis+1])
        else:
            oshape=[shape[laxis]*shape[laxis+1]]
            
        if laxis<len(shape)-1:
            oshape.extend(shape[laxis+2:])
            
        return(self.reshape(x,oshape))
        
        
    # ---------------------------------------------−---------
    def convol(self,image,axis=0):

        nside=int(np.sqrt(image.shape[axis]//12))

        if self.Idx_Neighbours[nside] is None:
            self.init_index(nside)
            
        imX9=self.bk_expand_dims(self.bk_gather(image,self.Idx_Neighbours[nside],axis=axis),-1)

        l_ww_real=self.ww_Real
        l_ww_imag=self.ww_Imag
        for i in range(axis+1):
            l_ww_real=self.bk_expand_dims(l_ww_real,0)
            l_ww_imag=self.bk_expand_dims(l_ww_imag,0)
        
        for i in range(axis+2,len(imX9.shape)-1):
            l_ww_real=self.bk_expand_dims(l_ww_real,axis+2)
            l_ww_imag=self.bk_expand_dims(l_ww_imag,axis+2)

        rr=self.bk_reduce_sum(l_ww_real*imX9,axis+1)
        ii=self.bk_reduce_sum(l_ww_imag*imX9,axis+1)

        return(rr,ii)
        
            
    # ---------------------------------------------−---------
    def smooth(self,image,axis=0):

        nside=int(np.sqrt(image.shape[axis]//12))

        if self.Idx_Neighbours[nside] is None:
            self.init_index(nside)
            
        imX9=self.bk_gather(image,self.Idx_Neighbours[nside],axis=axis)

        l_w_smooth=self.w_smooth
        for i in range(axis+1):
            l_w_smooth=self.bk_expand_dims(l_w_smooth,0)
        
        for i in range(axis+2,len(imX9.shape)):
            l_w_smooth=self.bk_expand_dims(l_w_smooth,axis+2)
                            
        res=self.bk_reduce_sum(l_w_smooth*imX9,axis+1)
        return(res)
    
    # ---------------------------------------------−---------
    def get_kernel_size(self):
        return(self.KERNELSZ)
    
    # ---------------------------------------------−---------
    def get_nb_orient(self):
        return(self.NORIENT)
    
    # ---------------------------------------------−---------
    def get_ww(self):
        return(self.ww_Real,self.ww_Image)
    
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
        
    
    
    
