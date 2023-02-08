import numpy as np
import healpy as hp
import os, sys

class Rformat:
    def __init__(self,
                 im,
                 off,
                 axis):
        self.data=im
        self.shape=im.shape
        self.axis=axis
        self.off=off
        self.nside=im.shape[axis+1]-2*off

    def get(self):
        return self.data

    def __add__(self,other):
        assert isinstance(other, float) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, Rformat)
        
        if isinstance(other, Rformat):
            return Rformat(self.get()+other.get(),self.off,self.axis)
        else:
            return Rformat(self.get()+other,self.off,self.axis)

    def __sub__(self,other):
        assert isinstance(other, float) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, Rformat)
        
        if isinstance(other, Rformat):
            return Rformat(self.get()-other.get(),self.off,self.axis)
        else:
            return Rformat(self.get()-other,self.off,self.axis)
            
    def __neg__(self):
        
        return Rformat(-self.get(),self.off,self.axis)

    def __mul__(self,other):
        assert isinstance(other, float) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, Rformat)
        
        if isinstance(other, Rformat):
            return Rformat(self.get()*other.get(),self.off,self.axis)
        else:
            return Rformat(self.get()*other,self.off,self.axis)

            
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
                 BACKEND='tensorflow',
                 use_R_format=False,
                 mpi_size=1,
                 mpi_rank=0):

        self.TENSORFLOW=1
        self.TORCH=2
        self.NUMPY=3
        self.isMPI=isMPI
        
        if BACKEND=='tensorflow':
            import tensorflow as tf
            
            self.backend=tf
            self.BACKEND=self.TENSORFLOW
            #tf.config.threading.set_inter_op_parallelism_threads(1)
            #tf.config.threading.set_intra_op_parallelism_threads(1)

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
        sys.stdout.flush()

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
        self.use_R_format=use_R_format
        
        """
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
        """
        
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
        if mpi_rank==0:
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

        self.rank=mpi_rank
        
        self.gpupos=(gpupos+mpi_rank)%self.ngpu
        print('============================================================')
        print('==                                                        ==')
        print('==                                                        ==')
        print('==     RUN ON GPU Rank %d : %s                          =='%(mpi_rank,self.gpulist[self.gpupos%self.ngpu]))
        print('==                                                        ==')
        print('==                                                        ==')
        print('============================================================')
        sys.stdout.flush()
        
        self.NORIENT=NORIENT
        self.LAMBDA=LAMBDA
        self.KERNELSZ=KERNELSZ
        self.slope=slope
        
        self.R_off=(self.KERNELSZ-1)//2
        if (self.R_off//2)*2<self.R_off:
            self.R_off+=1

        wwc=np.zeros([KERNELSZ**2,NORIENT]).astype(all_type)
        wws=np.zeros([KERNELSZ**2,NORIENT]).astype(all_type)

        x=np.repeat(np.arange(KERNELSZ)-KERNELSZ//2,KERNELSZ).reshape(KERNELSZ,KERNELSZ)
        y=x.T

        for i in range(NORIENT):
            a=i/float(NORIENT)*np.pi
            xx=(3/float(KERNELSZ))*LAMBDA*(x*np.cos(a)+y*np.sin(a))
            yy=(3/float(KERNELSZ))*LAMBDA*(x*np.sin(a)-y*np.cos(a))

            if KERNELSZ==5:
                w_smooth=np.exp(-2*((3.0/float(KERNELSZ)*xx)**2+(3.0/float(KERNELSZ)*yy)**2))
                #w_smooth=np.exp(-0.5*(xx**2+yy**2))
            else:
                w_smooth=np.exp(-0.5*(xx**2+yy**2))
        
            tmp=np.cos(yy*np.pi)*w_smooth
            wwc[:,i]=tmp.flatten()-tmp.mean()
            tmp=np.sin(yy*np.pi)*w_smooth
            wws[:,i]=tmp.flatten()-tmp.mean()
            sigma=np.sqrt((wwc[:,i]**2).mean())
            wwc[:,i]/=sigma
            wws[:,i]/=sigma

            w_smooth=w_smooth.flatten()
        
        self.w_smooth=slope*(w_smooth/w_smooth.sum()).astype(self.all_type)
        self.ww_Real=wwc.astype(self.all_type)
        self.ww_Imag=wws.astype(self.all_type)
        self.ww_RealT=wwc.astype(self.all_type)
        self.ww_ImagT=wws.astype(self.all_type)
        tab=[0,1,2,3]
        def trans_kernel(a):
            b=1*a.reshape(KERNELSZ,KERNELSZ)
            for i in range(KERNELSZ):
                for j in range(KERNELSZ):
                    b[i,j]=a.reshape(KERNELSZ,KERNELSZ)[KERNELSZ-1-i,KERNELSZ-1-j]
            return b.reshape(KERNELSZ*KERNELSZ)

        self.ww_SmoothT = self.w_smooth.reshape(KERNELSZ,KERNELSZ,1)
        
        for i in range(NORIENT):
            self.ww_RealT[:,i]=trans_kernel(self.ww_Real[:,tab[i]])
            self.ww_ImagT[:,i]=trans_kernel(self.ww_Imag[:,tab[i]])
          
        self.Idx_Neighbours={}
        self.pix_interp_val={}
        self.weight_interp_val={}
        self.ring2nest={}
        self.nest2R={}
        self.nest2R1={}
        self.nest2R2={}
        self.nest2R3={}
        self.nest2R4={}
        self.inv_nest2R={}
        self.remove_border={}
            
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
            self.nest2R[lout]=None
            self.nest2R1[lout]=None
            self.nest2R2[lout]=None
            self.nest2R3[lout]=None
            self.nest2R4[lout]=None
            self.inv_nest2R[lout]=None
            self.remove_border[lout]=None

        self.loss={}

    def get_use_R(self):
        return self.use_R_format
    
    # ---------------------------------------------−---------
    # --             BACKEND DEFINITION                    --
    # ---------------------------------------------−---------
    def bk_device(self,device_name):
        return self.backend.device(device_name)
        
    def bk_ones(self,shape,dtype=None):
        if dtype is None:
            dtype=self.all_type
        return(self.backend.ones(shape,dtype=dtype))

    def bk_L1(self,x):
        if isinstance(x,Rformat):
            return Rformat(self.backend.sign(x.get())* \
                           self.backend.sqrt(self.backend.sign(x.get())*x.get()),
                           x.off,x.axis)
        else:
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
        
        if isinstance(data,Rformat):
            return Rformat(self.bk_sqrt(data.get()),data.off,0)
        
        return(self.backend.sqrt(self.backend.abs(data)))
    
    def bk_abs(self,data):
        
        if isinstance(data,Rformat):
            return Rformat(self.bk_abs(data.get()),data.off,0)
        return(self.backend.abs(data))
        
    def bk_square(self,data):
        
        if isinstance(data,Rformat):
            return Rformat(self.bk_square(data.get()),data.off,0)
        
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
        if isinstance(data,Rformat):
            return Rformat(self.bk_reshape(data.get(),shape),data.off,0)
        
        return(self.backend.reshape(data,shape))
    
    def bk_repeat(self,data,nn,axis=0):
        return(self.backend.repeat(data,nn,axis=axis))

    def bk_expand_dims(self,data,axis=0):
        if isinstance(data,Rformat):
            if axis<data.axis:
                l_axis=data.axis+1
            else:
                l_axis=data.axis
            return Rformat(self.bk_expand_dims(data.get(),axis=axis),data.off,l_axis)
            
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
        if isinstance(data[0],Rformat):
            l_data=[idata.get() for idata in data]
            return Rformat(self.bk_concat(l_data,axis=axis),data[0].off,data[0].axis)
                
        if axis is None:
            return(self.backend.concat(data))
        else:
            return(self.backend.concat(data,axis=axis))
        
    def bk_relu(self,x):
        if isinstance(x,Rformat):
            return Rformat(self.bk_relu(x.get()),x.off,x.axis)
        
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.nn.relu(x)
        if self.BACKEND==self.TORCH:
            return self.backend.relu(x)
        if self.BACKEND==self.NUMPY:
            return (x>0)*x
        
    def bk_cast(self,x):
        if isinstance(x,Rformat):
            return Rformat(self.bk_cast(x.get()),x.off,x.axis)
        
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.cast(x,self.all_bk_type)
        if self.BACKEND==self.TORCH:
            return self.backend.cast(x,self.all_bk_type)
        if self.BACKEND==self.NUMPY:
            return x.astype(self.all_type)
    
    # ---------------------------------------------−---------
    # --       COMPUTE 3X3 INDEX FOR HEALPIX WORK          --
    # ---------------------------------------------−---------
    # convert all numpy array in the used bakcend format (e.g. Rformat if it is used)
    def conv_to_FoCUS(self,x,axis=0):
        if self.use_R_format and isinstance(x,np.ndarray):
            return(self.to_R(x,axis))
        return x
        
    def calc_R_index(self,nside):
        
        outname='BRD%d'%(self.R_off)
        try:
            fidx =np.load('%s/%s_%d_FIDX.npy'%(self.TEMPLATE_PATH,outname,nside))
            fidx1=np.load('%s/%s_%d_FIDX1.npy'%(self.TEMPLATE_PATH,outname,nside))
            fidx2=np.load('%s/%s_%d_FIDX2.npy'%(self.TEMPLATE_PATH,outname,nside))
            fidx3=np.load('%s/%s_%d_FIDX3.npy'%(self.TEMPLATE_PATH,outname,nside))
            fidx4=np.load('%s/%s_%d_FIDX4.npy'%(self.TEMPLATE_PATH,outname,nside))
        except:
            nstep=int(np.log(nside)/np.log(2))
            yy=(np.arange(12*nside*nside)//nside)%nside
            xx=nside-1-np.arange(12*nside*nside)%nside
            idx=(nside*nside)*(np.arange(12*nside*nside)//(nside*nside))

            for i in range(nstep):
                idx=idx+(((xx)//(2**i))%2)*(4**i)+2*((yy//(2**i))%2)*(4**i)

            off=self.R_off

            fidx=idx

            tab=np.array([[1,3,4,5],[2,0,5,6],[3,1,6,7],[0,2,7,4],
                          [0,3,11,8],[1,0,8,9],[2,1,9,10],[3,2,10,11],
                          [5,4,11,9],[6,5,8,10],[7,6,9,11],[4,7,10,8]])

            if self.Idx_Neighbours[nside] is None:
                self.init_index(nside)

            fidx1=np.zeros([12,off,nside],dtype='int')
            fidx2=np.zeros([12,off,nside],dtype='int')
            fidx3=np.zeros([12,nside+2*off,off],dtype='int')
            fidx4=np.zeros([12,nside+2*off,off],dtype='int')

            lidx=np.arange(nside,dtype='int')
            lidx2=np.arange(off*nside,dtype='int')

            for i in range(0,4):
                fidx1[i,:,:]=(tab[i,3]*(nside*nside)+(nside-off+lidx2//nside)*nside+lidx2%nside).reshape(off,nside)
                fidx2[i,:,:]=(tab[i,1]*(nside*nside)+(nside-1-lidx2%nside)*nside+lidx2//nside).reshape(off,nside)
                fidx3[i,off:-off,:]=(tab[i,0]*(nside*nside)+(nside-off+lidx2%off)*nside+nside-1-lidx2//off).reshape(nside,off)
                fidx4[i,off:-off,:]=(tab[i,2]*(nside*nside)+(lidx2//off)*nside+lidx2%off).reshape(nside,off)

            for i in range(4,8):
                fidx1[i,:,:]=(tab[i,3]*(nside*nside)+(nside-off+lidx2//nside)*nside+lidx2%nside).reshape(off,nside)
                fidx2[i,:,:]=(tab[i,1]*(nside*nside)+(lidx2//nside)*nside+lidx2%nside).reshape(off,nside)
                fidx3[i,off:-off,:]=(tab[i,0]*(nside*nside)+(lidx2//2)*nside+nside-off+lidx2%2).reshape(nside,off)
                fidx4[i,off:-off,:]=(tab[i,2]*(nside*nside)+(lidx2//2)*nside+lidx2%2).reshape(nside,off)

            for i in range(8,12):
                fidx1[i,:,:]=(tab[i,3]*(nside*nside)+(nside-1-lidx2%nside)*nside+nside-off+lidx2//nside).reshape(off,nside)
                fidx2[i,:,:]=(tab[i,1]*(nside*nside)+(lidx2//nside)*nside+lidx2%nside).reshape(off,nside)
                fidx3[i,off:-off,:]=(tab[i,0]*(nside*nside)+(lidx2//2)*nside+nside-off+lidx2%2).reshape(nside,off)
                fidx4[i,off:-off,:]=(tab[i,2]*(nside*nside)+(lidx2%2)*nside+nside-1-lidx2//2).reshape(nside,off)

            for k in range(12):
                lidx=fidx.reshape(12,nside,nside)[k,0,0]
                fidx3[k,off-1,off-1]=np.where(fidx==self.Idx_Neighbours[nside][lidx,8])[0]
                lidx=fidx.reshape(12,nside,nside)[k,-1,0]
                fidx3[k,-off,off-1]=np.where(fidx==self.Idx_Neighbours[nside][lidx,2])[0]
                lidx=fidx.reshape(12,nside,nside)[k,0,-1]
                fidx4[k,off-1,0]=np.where(fidx==self.Idx_Neighbours[nside][lidx,6])[0]
                lidx=fidx.reshape(12,nside,nside)[k,-1,-1]
                fidx4[k,-off,0]=np.where(fidx==self.Idx_Neighbours[nside][lidx,0])[0] # OK

            np.save('%s/%s_%d_FIDX.npy'%(self.TEMPLATE_PATH,outname,nside),fidx)
            print('%s/%s_%d_FIDX.npy COMPUTED'%(self.TEMPLATE_PATH,outname,nside))
            np.save('%s/%s_%d_FIDX1.npy'%(self.TEMPLATE_PATH,outname,nside),fidx1)
            print('%s/%s_%d_FIDX1.npy COMPUTED'%(self.TEMPLATE_PATH,outname,nside))
            np.save('%s/%s_%d_FIDX2.npy'%(self.TEMPLATE_PATH,outname,nside),fidx2)
            print('%s/%s_%d_FIDX2.npy COMPUTED'%(self.TEMPLATE_PATH,outname,nside))
            np.save('%s/%s_%d_FIDX3.npy'%(self.TEMPLATE_PATH,outname,nside),fidx3)
            print('%s/%s_%d_FIDX3.npy COMPUTED'%(self.TEMPLATE_PATH,outname,nside))
            np.save('%s/%s_%d_FIDX4.npy'%(self.TEMPLATE_PATH,outname,nside),fidx4)
            print('%s/%s_%d_FIDX4.npy COMPUTED'%(self.TEMPLATE_PATH,outname,nside))
            sys.stdout.flush()
        
        self.nest2R[nside]=fidx
        self.nest2R1[nside]=fidx1
        self.nest2R2[nside]=fidx2
        self.nest2R3[nside]=fidx3
        self.nest2R4[nside]=fidx4
    
    def calc_R_inv_index(self,nside):
        nstep=int(np.log(nside)/np.log(2))
        idx=np.arange(nside*nside)
        xx=np.zeros([nside*nside],dtype='int')
        yy=np.zeros([nside*nside],dtype='int')
        
        for i in range(nstep):
            l_idx=(idx//(4**i))%4
            xx=xx+(2**i)*((l_idx)%2)
            yy=yy+(2**i)*((l_idx)//2)
            
        return np.repeat(np.arange(12,dtype='int'),nside*nside)*(nside+2*self.R_off)*(nside+2*self.R_off)+ \
            np.tile(self.R_off+yy,12)*(nside+2*self.R_off)+self.R_off+np.tile(nside-1-xx,12)
            
    def update_R_border(self,im,axis=0):
        if not isinstance(im,Rformat):
            return im
            
        nside=im.shape[axis+1]-2*self.R_off
        
        if self.nest2R[nside] is None:
            self.calc_R_index(nside)
            
        if axis==0:
            im_center=im.get()[:,self.R_off:-self.R_off,self.R_off:-self.R_off]
        if axis==1:
            im_center=im.get()[:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
        if axis==2:
            im_center=im.get()[:,:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
        if axis==3:
            im_center=im.get()[:,:,:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]

        shape=list(im.shape)
        
        oshape=shape[0:axis]+[12*nside*nside]
        if len(shape)>axis+3:
            oshape=oshape+shape[axis+3:]

        l_im=self.bk_reshape(im_center,oshape)
        v1=self.bk_gather(l_im,self.nest2R1[nside],axis=axis)
        v2=self.bk_gather(l_im,self.nest2R2[nside],axis=axis)
        v3=self.bk_gather(l_im,self.nest2R3[nside],axis=axis)
        v4=self.bk_gather(l_im,self.nest2R4[nside],axis=axis)
            
        imout=self.bk_concat([v1,im_center,v2],axis=axis+1)
        imout=self.bk_concat([v3,imout,v4],axis=axis+2)
        return Rformat(imout,self.R_off,axis)
        
    def to_R_center(self,im,axis=0):
        if isinstance(im,Rformat):
            return im
        
        nside=int(np.sqrt(im.shape[axis]//12))
        
        if self.nest2R[nside] is None:
            self.calc_R_index(nside)
            
        im_center=self.bk_gather(im,self.nest2R[nside],axis=axis)
        return self.bk_reshape(im_center,[12*nside*nside])
        
    def to_R(self,im,axis=0,only_border=False):
        if isinstance(im,Rformat):
            return im
        
        nside=int(np.sqrt(im.shape[axis]//12))
        
        if self.nest2R[nside] is None:
            self.calc_R_index(nside)
                
        if only_border:
            v1=self.bk_gather(im,self.nest2R1[nside],axis=axis)
            v2=self.bk_gather(im,self.nest2R2[nside],axis=axis)
            v3=self.bk_gather(im,self.nest2R3[nside],axis=axis)
            v4=self.bk_gather(im,self.nest2R4[nside],axis=axis)
            
            if axis==0:
                im_center=self.bk_reshape(im,[12,nside,nside])
            if axis==1:
                im_center=self.bk_reshape(im,[im.shape[0],12,nside,nside])
            if axis==2:
                im_center=self.bk_reshape(im,[im.shape[0],im.shape[1],12,nside,nside])
            if axis==3:
                im_center=self.bk_reshape(im,[im.shape[0],im.shape[1],im.shape[2],12,nside,nside])
                
            imout=self.bk_concat([v1,im_center,v2],axis=axis+1)
            imout=self.bk_concat([v3,imout,v4],axis=axis+2)

            return Rformat(imout,self.R_off,axis)
        
        else:
            im_center=self.bk_gather(im,self.nest2R[nside],axis=axis)
            v1=self.bk_gather(im_center,self.nest2R1[nside],axis=axis)
            v2=self.bk_gather(im_center,self.nest2R2[nside],axis=axis)
            v3=self.bk_gather(im_center,self.nest2R3[nside],axis=axis)
            v4=self.bk_gather(im_center,self.nest2R4[nside],axis=axis)

            shape=list(im.shape)
            oshape=shape[0:axis]+[12,nside,nside]

            if axis+1<len(shape):
                oshape=oshape+shape[axis+1:]
            
            im_center=self.bk_reshape(im_center,oshape)
            
            imout=self.bk_concat([v1,im_center,v2],axis=axis+1)
            imout=self.bk_concat([v3,imout,v4],axis=axis+2)
            return Rformat(imout,self.R_off,axis)
    
    def from_R(self,im,axis=0):
        if not isinstance(im,Rformat):
            print('fromR function only works with Rformat class')
            
        image=im.get()
        nside=image.shape[axis+1]-self.R_off*2
        
        if self.inv_nest2R[nside] is None:
            self.inv_nest2R[nside]=self.calc_R_inv_index(nside)

        res=self.reduce_dim(self.reduce_dim(image,axis=axis),axis=axis)
        
        return self.bk_gather(res,self.inv_nest2R[nside],axis=axis)
        
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
        sys.stdout.flush()
            
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
        sys.stdout.flush()
        
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
        
        if self.use_R_format:
            if isinstance(im, Rformat):
                l_image=im.get()
            else:
                l_image=self.to_R(im,axis=axis).get()

            lout=int(l_image.shape[axis+1]-2*self.R_off)
            
            if self.nest2R[lout//2] is None:
                self.calc_R_index(lout//2)
                
            shape=list(im.shape)
            
            if axis==0:
                l_image=l_image[:,self.R_off:-self.R_off,self.R_off:-self.R_off]
                oshape=[12,lout//2,2,lout//2,2]
                oshape2=[12*(lout//2)*(lout//2)]
                if len(shape)>3:
                    oshape=oshape+shape[3:]
                    oshape2=oshape2+shape[3:]
            if axis==1:
                l_image=l_image[:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
                oshape=[shape[0],12,lout//2,2,lout//2,2]
                oshape2=[shape[0],12*(lout//2)*(lout//2)]
                if len(shape)>4:
                    oshape=oshape+shape[4:]
                    oshape2=oshape2+shape[4:]
            if axis==2:
                l_image=l_image[:,:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
                oshape=[shape[0],shape[1],12,lout//2,2,lout//2,2]
                oshape2=[shape[0],shape[1],12*(lout//2)*(lout//2)]
                if len(shape)>5:
                    oshape=oshape+shape[5:]
                    oshape2=oshape2+shape[5:]
            if axis==3:
                l_image=l_image[:,:,:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
                oshape=[shape[0],shape[1],shape[2],12,lout//2,2,lout//2,2]
                oshape2=[shape[0],shape[1],shape[2],12*(lout//2)*(lout//2)]
                if len(shape)>6:
                    oshape=oshape+shape[6:]
                    oshape2=oshape2+shape[6:]
                    
            if axis>3:
                print('ud_grade_2 function not yet implemented for axis>3')

            l_image=self.bk_reduce_sum(self.bk_reduce_sum(self.bk_reshape(l_image,oshape) \
                                                          ,axis=axis+2),axis=axis+3)/4
            imout=self.bk_reshape(l_image,oshape2)
            
            v1=self.bk_gather(imout,self.nest2R1[lout//2],axis=axis)
            v2=self.bk_gather(imout,self.nest2R2[lout//2],axis=axis)
            v3=self.bk_gather(imout,self.nest2R3[lout//2],axis=axis)
            v4=self.bk_gather(imout,self.nest2R4[lout//2],axis=axis)
            
            imout=self.bk_concat([v1,l_image,v2],axis=axis+1)
            imout=self.bk_concat([v3,imout,v4],axis=axis+2)

            imout=Rformat(imout,self.R_off,axis)
            
            if not isinstance(im, Rformat):
                imout=self.from_R(imout,axis=axis)

            return imout
            
        else:
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
    def up_grade_2_R_format(self,l_image,axis=0):
        
        #l_image is [....,12,nside+2*R_off,nside+2*R_off,...]
        res=self.backend.repeat(self.backend.repeat(l_image,2,axis=axis+1),2,axis=axis+2)

        y00=res
        y10=self.backend.roll(res,1,axis=axis+1)
        y01=self.backend.roll(res,1,axis=axis+2)
        y11=self.backend.roll(y10,1,axis=axis+2)

        #imout is [....,12,2*nside+4*R_off,2*nside+4*R_off,...]
        imout=(0.25*y00+0.25*y10+0.25*y01+0.25*y11)
        
        #reshape imout [NPRE,to cut axes
        if axis==0:
            # cas c'est une simple image
            imout=imout[:,self.R_off:-self.R_off,self.R_off:-self.R_off]
        if axis==1:
            imout=imout[:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
        if axis==2:
            imout=imout[:,:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
        if axis==3:
            imout=imout[:,:,:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
                
        return(imout)
    
    #--------------------------------------------------------
    def up_grade(self,im,nout,axis=0):
        
        if self.use_R_format:
            if isinstance(im, Rformat):
                l_image=im.get()
            else:
                l_image=self.to_R(im,axis=axis).get()

            lout=int(l_image.shape[axis+1]-2*self.R_off)
            
            nscale=int(np.log(nout//lout)/np.log(2))

            if lout==nout:
                imout=l_image
            else:
                imout=self.up_grade_2_R_format(l_image,axis=axis)
                for i in range(1,nscale):
                    imout=self.up_grade_2_R_format(imout,axis=axis)
                        
            imout=Rformat(imout,self.R_off,axis)
            
            if not isinstance(im, Rformat):
                imout=self.from_R(imout,axis=axis)
            
        else:

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
        else:
            laxis1=axis1
        if axis2<0:
            laxis2=len(shape)+axis2
        else:
            laxis2=axis2
        
        naxes=len(shape)
        thelist=[i for i in range(naxes)]
        thelist[laxis1]=laxis2
        thelist[laxis2]=laxis1
        return self.bk_transpose(x,thelist)
    
    # ---------------------------------------------−---------
    # Mean using mask x [....,Npix,....], mask[Nmask,Npix]  to [....,Nmask,....]
    # if use_R_format
    # Mean using mask x [....,12,Nside+2*off,Nside+2*off,....], mask[Nmask,12,Nside+2*off,Nside+2*off]  to [....,Nmask,....]
    def bk_masked_mean(self,x,mask,axis=0):
        
        shape=x.shape.as_list()
        
        l_x=self.bk_expand_dims(x,axis)
            
        if self.use_R_format:
            nside=mask.nside
            if self.remove_border[nside] is None:
                self.remove_border[nside]=np.ones([1,12,nside+2*self.R_off,nside+2*self.R_off])
                self.remove_border[nside][0,:,0:self.R_off,:]=0.0
                self.remove_border[nside][0,:,-self.R_off:,:]=0.0
                self.remove_border[nside][0,:,:,0:self.R_off]=0.0
                self.remove_border[nside][0,:,:,-self.R_off:]=0.0
                
            l_mask=mask.get()*self.remove_border[nside]
        else:
            l_mask=mask
            
        for i in range(axis):
            l_mask=self.bk_expand_dims(l_mask,0)
            
        if self.use_R_format:
            for i in range(axis+3,len(x.shape)):
                l_mask=self.bk_expand_dims(l_mask,-1)
            
            return self.bk_reduce_sum(self.bk_reduce_sum(self.bk_reduce_sum(l_mask*l_x.get(),axis=axis+1),axis=axis+1),axis=axis+1)/(12*nside*nside)
        else:
            for i in range(axis+1,len(x.shape)):
                l_mask=self.bk_expand_dims(l_mask,-1)

            return self.bk_reduce_mean(l_mask*l_x,axis=axis+1)
        
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
            
        return(self.bk_reshape(x,oshape))
        
        
    # ---------------------------------------------−---------
    def conv2d(self,image,ww,axis=0):

        if len(ww.shape)==2:
            norient=ww.shape[1]
        else:
            norient=ww.shape[2]

        shape=image.shape

        if axis>0:
            o_shape=shape[0]
            for k in range(1,axis+1):
                o_shape=o_shape*shape[k]
                
        if len(shape)>axis+3:
            ishape=shape[axis+3]
            for k in range(axis+4,len(shape)):
                ishape=ishape*shape[k]
                
            oshape=[o_shape,shape[axis+1],shape[axis+2],ishape]

            #l_image=self.swapaxes(self.bk_reshape(image,oshape),-1,-3)
            l_image=self.bk_reshape(image,oshape)

            l_ww=np.zeros([self.KERNELSZ,self.KERNELSZ,ishape,ishape*norient])
            for k in range(ishape):
                l_ww[:,:,k,k*norient:(k+1)*norient]=ww.reshape(self.KERNELSZ,self.KERNELSZ,norient)
            
            res=self.backend.nn.conv2d(l_image,l_ww,strides=[1, 1, 1, 1],padding='SAME')

            res=self.bk_reshape(res,[o_shape,shape[axis+1],shape[axis+2],ishape,norient])
        else:
            oshape=[o_shape,shape[axis+1],shape[axis+2],1]
            l_ww=self.bk_reshape(ww,[self.KERNELSZ,self.KERNELSZ,1,norient])
        
            res=self.backend.nn.conv2d(self.bk_reshape(image,oshape),
                                       l_ww,
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')

        return Rformat(self.bk_reshape(res,shape+[norient]),self.R_off,axis)
    
    # ---------------------------------------------−---------
    def convol(self,image,axis=0):

        if self.use_R_format:
            
            if isinstance(image, Rformat):
                l_image=image
            else:
                l_image=self.to_R(image,axis=axis)
            
            rr=self.conv2d(l_image.get(),self.ww_RealT,axis=axis)
            ii=self.conv2d(l_image.get(),self.ww_ImagT,axis=axis)
                
            
            if not isinstance(image, Rformat):
                rr=self.from_R(rr,axis=axis)
                ii=self.from_R(ii,axis=axis)
                
        else:
            nside=int(np.sqrt(image.shape[axis]//12))

            if self.Idx_Neighbours[nside] is None:
                self.init_index(nside)

            imX9=self.bk_expand_dims(self.bk_gather(self.bk_cast(image),
                                                    self.Idx_Neighbours[nside],axis=axis),-1)

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
    
        if self.use_R_format:
            if isinstance(image, Rformat):
                l_image=image.get()
            else:
                l_image=self.to_R(image,axis=axis).get()
                
            res=self.conv2d(l_image,self.ww_SmoothT,axis=axis)

            res=self.bk_reshape(res,l_image.shape)
            
            if not isinstance(image, Rformat):
                res=self.from_R(res,axis=axis)
                
        else:
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
        return(self.ww_Real,self.ww_Imag)
    
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
        
    
    
    
