import numpy as np
import healpy as hp
import os, sys

class Rformat:
    def __init__(self,
                 im,
                 off,
                 axis,
                 chans=12):
        self.data=im
        self.shape=im.shape
        self.axis=axis
        self.off=off
        self.chans=chans
        self.nside=im.shape[axis+1]-2*off

    def get(self):
        return self.data
    
    def get_data(self):
        if self.axis==0:
            return self.data[:,self.off:-self.off,self.off:-self.off]
        if self.axis==1:
            return self.data[:,:,self.off:-self.off,self.off:-self.off]
        if self.axis==2:
            return self.data[:,:,:,self.off:-self.off,self.off:-self.off]
        if self.axis==3:
            return self.data[:,:,:,:,self.off:-self.off,self.off:-self.off]

        print('get_data is implemented till axis==3 and axis=',self.axis)
        exit(0)

    def __add__(self,other):
        assert isinstance(other, float) or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, Rformat)
        
        if isinstance(other, Rformat):
            return Rformat(self.get()+other.get(),self.off,self.axis,chans=self.chans)
        else:
            return Rformat(self.get()+other,self.off,self.axis,chans=self.chans)

    def __sub__(self,other):
        assert isinstance(other, float) or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, Rformat)
        
        if isinstance(other, Rformat):
            return Rformat(self.get()-other.get(),self.off,self.axis,chans=self.chans)
        else:
            return Rformat(self.get()-other,self.off,self.axis,chans=self.chans)
            
    def __neg__(self):
        
        return Rformat(-self.get(),self.off,self.axis,chans=self.chans)

    def __mul__(self,other):
        assert isinstance(other, float) or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, Rformat)
        
        if isinstance(other, Rformat):
            return Rformat(self.get()*other.get(),self.off,self.axis,chans=self.chans)
        else:
            return Rformat(self.get()*other,self.off,self.axis,chans=self.chans)

            
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
                 use_R_format=True,
                 chans=12,
                 DODIV=False,
                 mpi_size=1,
                 mpi_rank=0):

        # P00 coeff for normalization for scat_cov
        self.P1_dic = None
        self.P2_dic = None

        self.TENSORFLOW=1
        self.TORCH=2
        self.NUMPY=3
        self.isMPI=isMPI

        self.mpi_size=mpi_size
        self.mpi_rank=mpi_rank
        
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
        if OSTEP<-1:
            print('Warning : OSTEP can not be smaller than -1')
            exit(0)
            
        self.OSTEP=OSTEP
        self.use_R_format=use_R_format
        self.chans=chans
        
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
        
        if isMPI:
            from mpi4py import MPI
            
            if all_type=='float32':
                self.MPI_ALL_TYPE=MPI.FLOAT
            else:
                self.MPI_ALL_TYPE=MPI.DOUBLE
        else:
            self.MPI_ALL_TYPE=None
            
        self.all_type=all_type
        
            
        if all_type=='float32':
            self.all_bk_type=self.backend.float32
            self.all_cbk_type=self.backend.complex64
        else:
            if all_type=='float64':
                self.all_type='float64'
                self.all_bk_type=self.backend.float64
                self.all_cbk_type=self.backend.complex128
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
                    self.ngpu=len(logical_gpus)
                    gpuname=logical_gpus[gpupos%self.ngpu].name
                    self.gpulist={}
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

        l_NORIENT=NORIENT
        if DODIV:
            l_NORIENT=NORIENT+2
            
        self.NORIENT=l_NORIENT
        self.LAMBDA=LAMBDA
        self.slope=slope
        
        self.R_off=(KERNELSZ-1)//2
        if (self.R_off//2)*2<self.R_off:
            self.R_off+=1
            
        self.ww_Real  = {} 
        self.ww_Imag  = {} 
        self.ww_RealT = {} 
        self.ww_ImagT = {}
        
        if KERNELSZ>5:
            for i in range(nstep_max):
                lout=(2**i)
                all_nmax=128
                self.ww_Real[lout]=np.load('ALLWAVE_%d_%d_Wr.npy'%(lout,all_nmax)).astype(self.all_type)
                self.ww_Imag[lout]=np.load('ALLWAVE_%d_%d_Wi.npy'%(lout,all_nmax)).astype(self.all_type)
                
            KERNELSZ=5
            x=np.repeat(np.arange(KERNELSZ)-KERNELSZ//2,KERNELSZ).reshape(KERNELSZ,KERNELSZ)
            y=x.T
            xx=(3/float(KERNELSZ))*LAMBDA*x
            yy=(3/float(KERNELSZ))*LAMBDA*y
            w_smooth=np.exp(-(xx**2+yy**2)).flatten()
            self.do_wigner=True
        else:
            self.do_wigner=False
            wwc=np.zeros([KERNELSZ**2,l_NORIENT]).astype(all_type)
            wws=np.zeros([KERNELSZ**2,l_NORIENT]).astype(all_type)

            x=np.repeat(np.arange(KERNELSZ)-KERNELSZ//2,KERNELSZ).reshape(KERNELSZ,KERNELSZ)
            y=x.T

            if NORIENT==1:
                xx=(3/float(KERNELSZ))*LAMBDA*x
                yy=(3/float(KERNELSZ))*LAMBDA*y

                if KERNELSZ==5:
                    #w_smooth=np.exp(-2*((3.0/float(KERNELSZ)*xx)**2+(3.0/float(KERNELSZ)*yy)**2))
                    w_smooth=np.exp(-(xx**2+yy**2))
                    tmp=np.exp(-2*(xx**2+yy**2))-0.25*np.exp(-0.5*(xx**2+yy**2))
                else:
                    w_smooth=np.exp(-0.5*(xx**2+yy**2))
                    tmp=np.exp(-2*(xx**2+yy**2))-0.25*np.exp(-0.5*(xx**2+yy**2))

                wwc[:,0]=tmp.flatten()-tmp.mean()
                tmp=0*w_smooth
                wws[:,0]=tmp.flatten()
                sigma=np.sqrt((wwc[:,0]**2).mean())
                wwc[:,0]/=sigma
                wws[:,0]/=sigma

                w_smooth=w_smooth.flatten()
            else:
                for i in range(NORIENT):
                    a=i/float(NORIENT)*np.pi
                    xx=(3/float(KERNELSZ))*LAMBDA*(x*np.cos(a)+y*np.sin(a))
                    yy=(3/float(KERNELSZ))*LAMBDA*(x*np.sin(a)-y*np.cos(a))

                    if KERNELSZ==5:
                        #w_smooth=np.exp(-2*((3.0/float(KERNELSZ)*xx)**2+(3.0/float(KERNELSZ)*yy)**2))
                        w_smooth=np.exp(-(xx**2+yy**2))
                    else:
                        w_smooth=np.exp(-0.5*(xx**2+yy**2))
                    tmp1=np.cos(yy*np.pi)*w_smooth
                    tmp2=np.sin(yy*np.pi)*w_smooth

                    wwc[:,i]=tmp1.flatten()-tmp1.mean()
                    wws[:,i]=tmp2.flatten()-tmp2.mean()
                    sigma=np.sqrt((wwc[:,i]**2).mean())
                    wwc[:,i]/=sigma
                    wws[:,i]/=sigma
                    
                    if DODIV and i==0:
                        r=(xx**2+yy**2)
                        theta=np.arctan2(yy,xx)
                        theta[KERNELSZ//2,KERNELSZ//2]=0.0
                        tmp1=r*np.cos(2*theta)*w_smooth
                        tmp2=r*np.sin(2*theta)*w_smooth
                        
                        wwc[:,NORIENT]=tmp1.flatten()-tmp1.mean()
                        wws[:,NORIENT]=tmp2.flatten()-tmp2.mean()
                        sigma=np.sqrt((wwc[:,NORIENT]**2).mean())
                        
                        wwc[:,NORIENT]/=sigma
                        wws[:,NORIENT]/=sigma
                        tmp1=r*np.cos(2*theta+np.pi)
                        tmp2=r*np.sin(2*theta+np.pi)
                        
                        wwc[:,NORIENT+1]=tmp1.flatten()-tmp1.mean()
                        wws[:,NORIENT+1]=tmp2.flatten()-tmp2.mean()
                        sigma=np.sqrt((wwc[:,NORIENT+1]**2).mean())
                        wwc[:,NORIENT+1]/=sigma
                        wws[:,NORIENT+1]/=sigma
                        

                    w_smooth=w_smooth.flatten()
        
        self.KERNELSZ=KERNELSZ
        self.w_smooth=slope*(w_smooth/w_smooth.sum()).astype(self.all_type)
        for i in range(nstep_max):
            lout=(2**i)
            if not self.do_wigner:
                self.ww_Real[lout]=wwc.astype(self.all_type)
                self.ww_Imag[lout]=wws.astype(self.all_type)
                self.ww_RealT[lout]=wwc.astype(self.all_type)
                self.ww_ImagT[lout]=wws.astype(self.all_type)
            else:
                self.ww_RealT[lout]=np.zeros([KERNELSZ**2,l_NORIENT]).astype(all_type)
                self.ww_ImagT[lout]=np.zeros([KERNELSZ**2,l_NORIENT]).astype(all_type)
        
        def trans_kernel(a):
            b=1*a.reshape(KERNELSZ,KERNELSZ)
            for i in range(KERNELSZ):
                for j in range(KERNELSZ):
                    b[i,j]=a.reshape(KERNELSZ,KERNELSZ)[KERNELSZ-1-i,KERNELSZ-1-j]
            return b.reshape(KERNELSZ*KERNELSZ)

        self.ww_SmoothT = self.w_smooth.reshape(KERNELSZ,KERNELSZ,1)
        
        if not self.do_wigner:
            for i in range(nstep_max):
                for i in range(l_NORIENT):
                    self.ww_RealT[lout][:,i]=trans_kernel(self.ww_Real[lout][:,i])
                    self.ww_ImagT[lout][:,i]=trans_kernel(self.ww_Imag[lout][:,i])
          
        self.Idx_Neighbours={}
        self.Idx_convol={}
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
            self.Idx_convol[lout]=None
            self.nest2R[lout]=None
            self.nest2R1[lout]=None
            self.nest2R2[lout]=None
            self.nest2R3[lout]=None
            self.nest2R4[lout]=None
            self.inv_nest2R[lout]=None
            self.remove_border[lout]=None

        self.loss={}

    def get_type(self):
        return self.all_type

    def get_mpi_type(self):
        return self.MPI_ALL_TYPE
    
    def get_use_R(self):
        return self.use_R_format
    
    def is_R(self,data):
        return isinstance(data,Rformat)
    
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
            return Rformat(self.bk_L1(x.get()),x.off,x.axis,chans=self.chans)
        else:
            if x.dtype==self.all_cbk_type:
                xr=self.bk_real(x)
                xi=self.bk_imag(x)
                
                r=self.backend.sign(xr)*self.backend.sqrt(self.backend.sign(xr)*xr)
                i=self.backend.sign(xi)*self.backend.sqrt(self.backend.sign(xi)*xi)
                return self.bk_complex(r,i)
            else:
                return self.backend.sign(x)*self.backend.sqrt(self.backend.sign(x)*x)
        
    def bk_reduce_sum(self,data,axis=None):
        
        if isinstance(data,Rformat):
            return self.bk_reduce_sum(data.get())
        
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
        
        if isinstance(data,Rformat):
            return self.bk_reduce_mean(data.get())
        
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
            return Rformat(self.bk_sqrt(data.get()),data.off,0,chans=self.chans)
        
        return(self.backend.sqrt(self.backend.abs(data)))
    
    def bk_abs(self,data):
        
        if isinstance(data,Rformat):
            return Rformat(self.bk_abs(data.get()),data.off,0,chans=self.chans)
        return(self.backend.abs(data))
        
    def bk_square(self,data):
        
        if isinstance(data,Rformat):
            return Rformat(self.bk_square(data.get()),data.off,0,chans=self.chans)
        
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
            return Rformat(self.bk_reshape(data.get(),shape),data.off,0,chans=self.chans)
        
        return(self.backend.reshape(data,shape))
    
    def bk_repeat(self,data,nn,axis=0):
        return(self.backend.repeat(data,nn,axis=axis))

    def bk_expand_dims(self,data,axis=0):
        if isinstance(data,Rformat):
            if axis<data.axis:
                l_axis=data.axis+1
            else:
                l_axis=data.axis
            return Rformat(self.bk_expand_dims(data.get(),axis=axis),data.off,l_axis,chans=self.chans)
            
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
            return Rformat(self.bk_concat(l_data,axis=axis),data[0].off,data[0].axis,chans=self.chans)
                
        if axis is None:
            return(self.backend.concat(data))
        else:
            return(self.backend.concat(data,axis=axis))

    
    def bk_conjugate(self,data):
        if isinstance(data,Rformat):
            return Rformat(self.bk_conjugate(data.get()),data.off,data.axis,chans=self.chans)
                
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.math.conj(data)
        if self.BACKEND==self.TORCH:
            return self.backend.conjugate(data)
        if self.BACKEND==self.NUMPY:
            return data.conjugate()
        
    def bk_real(self,data):
        if isinstance(data,Rformat):
            return Rformat(self.bk_real(data.get()),data.off,data.axis,chans=self.chans)
                
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.math.real(data)
        if self.BACKEND==self.TORCH:
            return self.backend.real(data)
        if self.BACKEND==self.NUMPY:
            return self.backend.real(data)

    def bk_imag(self,data):
        if isinstance(data,Rformat):
            return Rformat(self.bk_imag(data.get()),data.off,data.axis,chans=self.chans)
                
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.math.imag(data)
        if self.BACKEND==self.TORCH:
            return self.backend.imag(data)
        if self.BACKEND==self.NUMPY:
            return self.backend.imag(data)
        
    def bk_relu(self,x):
        if isinstance(x,Rformat):
            return Rformat(self.bk_relu(x.get()),x.off,x.axis,chans=self.chans)
        
        if self.BACKEND==self.TENSORFLOW:
            if x.dtype==self.all_cbk_type:
                xr=self.backend.nn.relu(self.bk_real(x))
                xi=self.backend.nn.relu(self.bk_imag(x))
                return self.backend.complex(xr,xi)
            else:
                return self.backend.nn.relu(x)
        if self.BACKEND==self.TORCH:
            return self.backend.relu(x)
        if self.BACKEND==self.NUMPY:
            return (x>0)*x
        
    def bk_cast(self,x):
        if isinstance(x,Rformat):
            return Rformat(self.bk_cast(x.get()),x.off,x.axis,chans=self.chans)

        if x.dtype=='complex128' or x.dtype=='complex64':
            out_type=self.all_cbk_type
        else:
            out_type=self.all_bk_type
            
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.cast(x,out_type)
        if self.BACKEND==self.TORCH:
            return self.backend.cast(x,out_type)
        if self.BACKEND==self.NUMPY:
            return x.astype(out_type)
    
    # ---------------------------------------------−---------
    # --       COMPUTE 3X3 INDEX FOR HEALPIX WORK          --
    # ---------------------------------------------−---------
    # convert all numpy array in the used bakcend format (e.g. Rformat if it is used)
    def conv_to_FoCUS(self,x,axis=0):
        if self.use_R_format and isinstance(x,np.ndarray):
            return(self.to_R(x,axis,chans=self.chans))
        return x
        
    def calc_R_index(self,nside,chans=12):
        # if chans=12 healpix sinon chans=1

        outname='BRD%d'%(self.R_off)
        try:
            fidx =np.load('%s/%s_%d_%d_FIDX.npy'%(self.TEMPLATE_PATH,outname,nside,chans))
            fidx1=np.load('%s/%s_%d_%d_FIDX1.npy'%(self.TEMPLATE_PATH,outname,nside,chans))
            fidx2=np.load('%s/%s_%d_%d_FIDX2.npy'%(self.TEMPLATE_PATH,outname,nside,chans))
            fidx3=np.load('%s/%s_%d_%d_FIDX3.npy'%(self.TEMPLATE_PATH,outname,nside,chans))
            fidx4=np.load('%s/%s_%d_%d_FIDX4.npy'%(self.TEMPLATE_PATH,outname,nside,chans))
        except:
            nstep=int(np.log(nside)/np.log(2))
            yy=(np.arange(chans*nside*nside)//nside)%nside
            xx=nside-1-np.arange(chans*nside*nside)%nside
            idx=(nside*nside)*(np.arange(chans*nside*nside)//(nside*nside))

            if chans==12:
                for i in range(nstep):
                    idx=idx+(((xx)//(2**i))%2)*(4**i)+2*((yy//(2**i))%2)*(4**i)
            else:
                idx=yy*nside+nside-1-xx
                
            off=self.R_off

            fidx=idx

            if chans==12:
                tab=np.array([[1,3,4,5],[2,0,5,6],[3,1,6,7],[0,2,7,4],
                              [0,3,11,8],[1,0,8,9],[2,1,9,10],[3,2,10,11],
                              [5,4,11,9],[6,5,8,10],[7,6,9,11],[4,7,10,8]])

                #if self.Idx_Neighbours[nside] is None:
                r_idx=self.init_index(nside,kernel=5)
                r_idx2=self.init_index(nside,kernel=3)

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
                    fidx3[k,0,0]=np.where(fidx==r_idx[lidx,24])[0]
                    fidx3[k,0,1]=np.where(fidx==r_idx[lidx,23])[0]
                    fidx3[k,1,0]=np.where(fidx==r_idx[lidx,19])[0]
                    fidx3[k,1,1]=np.where(fidx==r_idx2[lidx,8])[0]
                    #print('+++',k)
                    #print(fidx.reshape(12,nside,nside)[k,0:3,0:3],':',fidx[fidx1[k,:,0:3]],':',fidx[fidx3[k,0:5,:]])
                    #print(r_idx[lidx,:].reshape(5,5))
                    #print(r_idx2[lidx,:].reshape(3,3))
                    lidx=fidx.reshape(12,nside,nside)[k,-1,0]
                    fidx3[k,-1,0]=np.where(fidx==r_idx[lidx,4])[0]
                    fidx3[k,-1,1]=np.where(fidx==r_idx[lidx,3])[0]
                    fidx3[k,-2,0]=np.where(fidx==r_idx[lidx,9])[0]
                    fidx3[k,-2,1]=np.where(fidx==r_idx2[lidx,2])[0]
                    #print('====',k)
                    #print(fidx.reshape(12,nside,nside)[k,-3:,0:3],':',fidx[fidx2[k,:,0:3]],':',fidx[fidx3[k,-5:,:]])
                    #print(r_idx[lidx,:].reshape(5,5))
                    #print(r_idx2[lidx,:].reshape(3,3))
                    #fidx4[k,off-1,0]=np.where(fidx==r_idx[lidx,6])[0]
                    lidx=fidx.reshape(12,nside,nside)[k,0,-1]
                    fidx4[k,0,0]=np.where(fidx==r_idx[lidx,21])[0]
                    fidx4[k,0,1]=np.where(fidx==r_idx[lidx,20])[0]
                    fidx4[k,1,1]=np.where(fidx==r_idx[lidx,15])[0]
                    fidx4[k,1,0]=np.where(fidx==r_idx2[lidx,6])[0]
                    #print('====',k)
                    #print(fidx.reshape(12,nside,nside)[k,0:3,-3:],':',fidx[fidx1[k,:,-2:]],':',fidx[fidx4[k,0:5,:]])
                    #print(r_idx[lidx,:].reshape(5,5))
                    #print(r_idx2[lidx,:].reshape(3,3))
                    #fidx4[k,off-1,0]=np.where(fidx==r_idx[lidx,6])[0]
                    lidx=fidx.reshape(12,nside,nside)[k,-1,-1]
                    fidx4[k,-1,1]=np.where(fidx==r_idx[lidx,0])[0]
                    fidx4[k,-1,0]=np.where(fidx==r_idx[lidx,1])[0]
                    fidx4[k,-2,1]=np.where(fidx==r_idx[lidx,5])[0]
                    fidx4[k,-2,0]=np.where(fidx==r_idx2[lidx,0])[0]
                    #print('+++',k)
                    #print(fidx.reshape(12,nside,nside)[k,-3:,-3:],':',fidx[fidx2[k,:,-3:]],':',fidx[fidx4[k,-5:,:]])
                    #print(r_idx[lidx,:].reshape(5,5))
                    #print(r_idx2[lidx,:].reshape(3,3))
                    #fidx4[k,-off,0]=np.where(fidx==r_idx[lidx,0])[0]

                fidx = (fidx+12*nside*nside)%(12*nside*nside)
                fidx1 = (fidx1+12*nside*nside)%(12*nside*nside)
                fidx2 = (fidx2+12*nside*nside)%(12*nside*nside)
                fidx3 = (fidx3+12*nside*nside)%(12*nside*nside)
                fidx4 = (fidx4+12*nside*nside)%(12*nside*nside)
            else:
                ll_idx=np.arange(nside,dtype='int')
                fidx1=np.zeros([1,off,nside],dtype='int')
                fidx2=np.zeros([1,off,nside],dtype='int')
                fidx3=np.zeros([1,nside+2*off,off],dtype='int')
                fidx4=np.zeros([1,nside+2*off,off],dtype='int')
                for i in range(2):
                    fidx1[0,i,:] = (nside-off+i)*nside+ll_idx
                    fidx2[0,i,:] = i*nside+ll_idx
                    fidx3[0,off:-off,i] = nside-2+i+nside*ll_idx
                    fidx4[0,off:-off,i] = i+nside*ll_idx

                    for j in range(2):
                        fidx3[0,j,i]=nside-2+i+nside*(nside-2+j)
                        fidx3[0,nside+off+j,i]=nside-2+i+nside*(j)
                        fidx4[0,j,i]=i+nside*(nside-2+j)
                        fidx4[0,nside+off+j,i]=i+nside*(j)
                    
                fidx = (fidx+nside*nside)%(nside*nside)
                fidx1 = (fidx1+nside*nside)%(nside*nside)
                fidx2 = (fidx2+nside*nside)%(nside*nside)
                fidx3 = (fidx3+nside*nside)%(nside*nside)
                fidx4 = (fidx4+nside*nside)%(nside*nside)
            
            np.save('%s/%s_%d_%d_FIDX.npy'%(self.TEMPLATE_PATH,outname,nside,chans),fidx)
            print('%s/%s_%d_%d_FIDX.npy COMPUTED'%(self.TEMPLATE_PATH,outname,nside,chans))
            np.save('%s/%s_%d_%d_FIDX1.npy'%(self.TEMPLATE_PATH,outname,nside,chans),fidx1)
            print('%s/%s_%d_%d_FIDX1.npy COMPUTED'%(self.TEMPLATE_PATH,outname,nside,chans))
            np.save('%s/%s_%d_%d_FIDX2.npy'%(self.TEMPLATE_PATH,outname,nside,chans),fidx2)
            print('%s/%s_%d_%d_FIDX2.npy COMPUTED'%(self.TEMPLATE_PATH,outname,nside,chans))
            np.save('%s/%s_%d_%d_FIDX3.npy'%(self.TEMPLATE_PATH,outname,nside,chans),fidx3)
            print('%s/%s_%d_%d_FIDX3.npy COMPUTED'%(self.TEMPLATE_PATH,outname,nside,chans))
            np.save('%s/%s_%d_%d_FIDX4.npy'%(self.TEMPLATE_PATH,outname,nside,chans),fidx4)
            print('%s/%s_%d_%d_FIDX4.npy COMPUTED'%(self.TEMPLATE_PATH,outname,nside,chans))
            sys.stdout.flush()
            
        self.nest2R[nside]=fidx
        self.nest2R1[nside]=fidx1
        self.nest2R2[nside]=fidx2
        self.nest2R3[nside]=fidx3
        self.nest2R4[nside]=fidx4
    
    def calc_R_inv_index(self,nside,chans=12):
        nstep=int(np.log(nside)/np.log(2))
        idx=np.arange(nside*nside)
        
        if chans==1:
            return (self.R_off+idx//nside)*(nside+2*self.R_off)+self.R_off+idx%nside
        
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
            self.calc_R_index(nside,chans=self.chans)
            
        if axis==0:
            im_center=im.get()[:,self.R_off:-self.R_off,self.R_off:-self.R_off]
        if axis==1:
            im_center=im.get()[:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
        if axis==2:
            im_center=im.get()[:,:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
        if axis==3:
            im_center=im.get()[:,:,:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]

        shape=list(im.shape)
        
        oshape=shape[0:axis]+[self.chans*nside*nside]
        if len(shape)>axis+3:
            oshape=oshape+shape[axis+3:]

        l_im=self.bk_reshape(im_center,oshape)
        v1=self.bk_gather(l_im,self.nest2R1[nside],axis=axis)
        v2=self.bk_gather(l_im,self.nest2R2[nside],axis=axis)
        v3=self.bk_gather(l_im,self.nest2R3[nside],axis=axis)
        v4=self.bk_gather(l_im,self.nest2R4[nside],axis=axis)
            
        imout=self.bk_concat([v1,im_center,v2],axis=axis+1)
        imout=self.bk_concat([v3,imout,v4],axis=axis+2)
        return Rformat(imout,self.R_off,axis,chans=self.chans)
        
    def to_R_center(self,im,axis=0,chans=12):
        if isinstance(im,Rformat):
            return im
        
        if chans==12:
            nside=int(np.sqrt(im.shape[axis]//chans))
        else:
            nside=im.shape[axis]
        
        if chans==1:
            lim=self.reduce_dim(im,axis=axis)
        else:
            lim=im
            
        if self.nest2R[nside] is None:
            self.calc_R_index(nside,chans=chans)
        
        im_center=self.bk_gather(lim,self.nest2R[nside],axis=axis)
        
        return self.bk_reshape(im_center,[chans*nside*nside])
        
    def to_R(self,im,axis=0,only_border=False,chans=12):
        if isinstance(im,Rformat):
            return im

        if chans==1 and len(im.shape)>1:
            nside=im.shape[axis]
        else:
            nside=int(np.sqrt(im.shape[axis]//chans))
        
        if self.nest2R[nside] is None:
            self.calc_R_index(nside,chans=chans)
                
        if only_border:
            
            if chans==1 and len(im.shape)>1:
                lim=self.reduce_dim(im,axis=axis)
            else:
                lim=im
                
            v1=self.bk_gather(lim,self.nest2R1[nside],axis=axis)
            v2=self.bk_gather(lim,self.nest2R2[nside],axis=axis)
            v3=self.bk_gather(lim,self.nest2R3[nside],axis=axis)
            v4=self.bk_gather(lim,self.nest2R4[nside],axis=axis)
            
            if axis==0:
                im_center=self.bk_reshape(im,[chans,nside,nside])
            if axis==1:
                im_center=self.bk_reshape(im,[im.shape[0],chans,nside,nside])
            if axis==2:
                im_center=self.bk_reshape(im,[im.shape[0],im.shape[1],chans,nside,nside])
            if axis==3:
                im_center=self.bk_reshape(im,[im.shape[0],im.shape[1],im.shape[2],chans,nside,nside])
                
            imout=self.bk_concat([v1,im_center,v2],axis=axis+1)
            imout=self.bk_concat([v3,imout,v4],axis=axis+2)

            return Rformat(imout,self.R_off,axis,chans=chans)
        
        else:
            if chans==1:
                im_center=self.reduce_dim(im,axis=axis)
            else:
                im_center=self.bk_gather(im,self.nest2R[nside],axis=axis)
            v1=self.bk_gather(im_center,self.nest2R1[nside],axis=axis)
            v2=self.bk_gather(im_center,self.nest2R2[nside],axis=axis)
            v3=self.bk_gather(im_center,self.nest2R3[nside],axis=axis)
            v4=self.bk_gather(im_center,self.nest2R4[nside],axis=axis)

            shape=list(im.shape)
            oshape=shape[0:axis]+[chans,nside,nside]

            if chans==1:
                if axis+2<len(shape):
                    oshape=oshape+shape[axis+2:]
            else:
                if axis+1<len(shape):
                    oshape=oshape+shape[axis+1:]
            
            im_center=self.bk_reshape(im_center,oshape)

            imout=self.bk_concat([v1,im_center,v2],axis=axis+1)
            imout=self.bk_concat([v3,imout,v4],axis=axis+2)
            return Rformat(imout,self.R_off,axis,chans=chans)
    
    def from_R(self,im,axis=0):
        if not isinstance(im,Rformat):
            print('fromR function only works with Rformat class')
            
        image=im.get()
        if im.chans==1:
            if axis==0:
                im_center=im.get()[0,self.R_off:-self.R_off,self.R_off:-self.R_off]
            if axis==1:
                im_center=im.get()[0,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
            if axis==2:
                im_center=im.get()[0,:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
            if axis==3:
                im_center=im.get()[0,:,:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
            return im_center
        else:
            nside=image.shape[axis+1]-self.R_off*2
        
            if self.inv_nest2R[nside] is None:
                self.inv_nest2R[nside]=self.calc_R_inv_index(nside,chans=im.chans)

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
                l_image=self.to_R(im,axis=axis,chans=self.chans).get()

            lout=int(l_image.shape[axis+1]-2*self.R_off)
            
            if self.nest2R[lout//2] is None:
                self.calc_R_index(lout//2,chans=self.chans)
                
            shape=list(im.shape)
            
            if axis==0:
                l_image=l_image[:,self.R_off:-self.R_off,self.R_off:-self.R_off]
                oshape=[self.chans,lout//2,2,lout//2,2]
                oshape2=[self.chans*(lout//2)*(lout//2)]
                if len(shape)>3:
                    oshape=oshape+shape[3:]
                    oshape2=oshape2+shape[3:]
            if axis==1:
                l_image=l_image[:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
                oshape=[shape[0],self.chans,lout//2,2,lout//2,2]
                oshape2=[shape[0],self.chans*(lout//2)*(lout//2)]
                if len(shape)>4:
                    oshape=oshape+shape[4:]
                    oshape2=oshape2+shape[4:]
            if axis==2:
                l_image=l_image[:,:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
                oshape=[shape[0],shape[1],self.chans,lout//2,2,lout//2,2]
                oshape2=[shape[0],shape[1],self.chans*(lout//2)*(lout//2)]
                if len(shape)>5:
                    oshape=oshape+shape[5:]
                    oshape2=oshape2+shape[5:]
            if axis==3:
                l_image=l_image[:,:,:,:,self.R_off:-self.R_off,self.R_off:-self.R_off]
                oshape=[shape[0],shape[1],shape[2],self.chans,lout//2,2,lout//2,2]
                oshape2=[shape[0],shape[1],shape[2],self.chans*(lout//2)*(lout//2)]
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

            imout=Rformat(imout,self.R_off,axis,chans=self.chans)
            
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
        y10=self.backend.roll(res,-1,axis=axis+1)
        y01=self.backend.roll(res,-1,axis=axis+2)
        y11=self.backend.roll(y10,-1,axis=axis+2)
        #imout is [....,12,2*nside+4*R_off,2*nside+4*R_off,...]
        imout=(0.25*y00+0.25*y10+0.25*y01+0.25*y11)
        
        y10=self.backend.roll(res,1,axis=axis+1)
        y01=self.backend.roll(res,1,axis=axis+2)
        y11=self.backend.roll(y10,1,axis=axis+2)
        #imout is [....,12,2*nside+4*R_off,2*nside+4*R_off,...]
        imout=imout+(0.25*y00+0.25*y10+0.25*y01+0.25*y11)
        
        y10=self.backend.roll(res,1,axis=axis+1)
        y01=self.backend.roll(res,-1,axis=axis+2)
        y11=self.backend.roll(y10,-1,axis=axis+2)
        #imout is [....,12,2*nside+4*R_off,2*nside+4*R_off,...]
        imout=imout+(0.25*y00+0.25*y10+0.25*y01+0.25*y11)
        
        y10=self.backend.roll(res,-1,axis=axis+1)
        y01=self.backend.roll(res,1,axis=axis+2)
        y11=self.backend.roll(y10,1,axis=axis+2)
        #imout is [....,12,2*nside+4*R_off,2*nside+4*R_off,...]
        imout=imout+(0.25*y00+0.25*y10+0.25*y01+0.25*y11)

        imout=imout/4
        
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
                l_image=self.to_R(im,axis=axis,chans=self.chans).get()

            lout=int(l_image.shape[axis+1]-2*self.R_off)
            
            nscale=int(np.log(nout//lout)/np.log(2))

            if lout==nout:
                imout=l_image
            else:
                imout=self.up_grade_2_R_format(l_image,axis=axis)
                for i in range(1,nscale):
                    imout=self.up_grade_2_R_format(imout,axis=axis)
                        
            imout=Rformat(imout,self.R_off,axis,chans=self.chans)
            
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
    def computeWigner(self,nside,lmax=1.5,all_nmax=128):
        
        nmax=np.min([12*nside**2,all_nmax])
        lidx=hp.ring2nest(nside,np.arange(12*nside**2))
        th,ph=hp.pix2ang(nside,np.arange(12*nside**2),nest=True)
        a=np.exp(-0.1*(nside**2)*((th-np.pi/2)**2+(ph-np.pi)**2))+ \
           np.exp(-0.1*(nside**2)*((th-np.pi/2)**2+(ph+np.pi)**2))
        filter=a*np.cos(lmax*(nside+1)*(th-np.pi/2))
        tmp=hp.anafast(filter[lidx])
        norm=1/np.sqrt(tmp.max())
        filter*=norm
        tmp=hp.anafast(filter[lidx])
        tot[0:tmp.shape[0]]+=tmp
        plt.plot(tmp)
        wr=np.zeros([12*nside**2,nmax,4])
        wi=np.zeros([12*nside**2,nmax,4])
        iii=np.zeros([12*nside**2,nmax],dtype='int')
            
        for l in range(12*nside**2):
            wwr=np.zeros([12*nside**2,4])
            wwi=np.zeros([12*nside**2,4])
            for k in range(4):
                r=hp.Rotator(rot=((np.pi-ph[l])/np.pi*180,(np.pi/2-th[l])/np.pi*180,45*k))
                th2,ph2=r(th,ph)
                a=np.exp(-0.1*(nside**2)*((th2-np.pi/2)**2+(ph2-np.pi)**2))+ \
                   np.exp(-0.1*(nside**2)*((th2-np.pi/2)**2+(ph2+np.pi)**2))
                wwr[:,k]=norm*a*np.cos(lmax*(nside+1)*(th2-np.pi/2))
                wwi[:,k]=norm*a*np.sin(lmax*(nside+1)*(th2-np.pi/2))

            idx=np.argsort(-np.sum(abs(wwr+complex(0,1)*wwi),1))
            wr[l,:,:]=wwr[idx[0:nmax],:]
            wi[l,:,:]=wwi[idx[0:nmax],:]
            iii[l,:]=idx[0:nmax]

        print('Write ALLWAVE_%d_%d_W.npy'%(nside,k),wr.shape[1]/(12*nside**2))
        np.save('ALLWAVE_%d_%d_Wr.npy'%(nside,all_nmax),wr)
        np.save('ALLWAVE_%d_%d_Wi.npy'%(nside,all_nmax),wi)
        np.save('ALLWAVE_%d_%d_I.npy'%(nside,all_nmax),iii)
        
    # ---------------------------------------------−---------
    def init_index(self,nside,kernel=-1):
        
        if kernel==-1:
            l_kernel=self.KERNELSZ
        else:
            l_kernel=kernel
            
        try:
            tmp=np.load('%s/W%d_%d_IDX.npy'%(self.TEMPLATE_PATH,l_kernel**2,nside))
        except:
            if l_kernel**2==9:
                if self.rank==0:
                    self.comp_idx_w9(nside)
            elif l_kernel**2==25:
                if self.rank==0:
                    self.comp_idx_w25(nside)
            else:
                if self.rank==0:
                    print('Only 3x3 and 5x5 kernel have been developped for Healpix and you ask for %dx%d'%(KERNELSZ,KERNELSZ))
                    exit(0)
            self.barrier()
            tmp=np.load('%s/W%d_%d_IDX.npy'%(self.TEMPLATE_PATH,l_kernel**2,nside))

                
        if kernel==-1:
            if self.BACKEND==self.TORCH:
                self.Idx_Neighbours[nside]=tmp.as_type('int64')
            else:
                self.Idx_Neighbours[nside]=tmp

        
        if self.do_wigner:
            try:
                self.Idx_convol[nside]=np.load('ALLWAVE_%d_%d_I.npy'%(nside,128))
            except:
                self.computeWigner(nside)
        else:
                self.Idx_convol[nside]=self.Idx_Neighbours[nside]
                
        if kernel!=-1:
            return tmp
        
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
            
            if l_x.get().dtype==self.all_cbk_type:
                l_mask=self.bk_complex(l_mask,0.0)
                
            return self.bk_reduce_sum(self.bk_reduce_sum(self.bk_reduce_sum(l_mask*l_x.get(),axis=axis+1),axis=axis+1),axis=axis+1)/(12*nside*nside)
        else:
            for i in range(axis+1,len(x.shape)):
                l_mask=self.bk_expand_dims(l_mask,-1)

            if l_x.dtype==self.all_cbk_type:
                l_mask=self.bk_complex(l_mask,0.0)
            return self.bk_reduce_mean(l_mask*l_x,axis=axis+1)
        
    # ---------------------------------------------−---------
    # convert tensor x [....,a,b,....] to [....,a*b,....]
    def reduce_dim(self,x,axis=0):
        shape=list(x.shape)
        
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
        else:
            o_shape=image.shape[0]
            
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
            
            if l_image.dtype=='complex128' or l_image.dtype=='complex64':
                r=self.backend.nn.conv2d(self.bk_real(l_image),
                                         l_ww,
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
                i=self.backend.nn.conv2d(self.bk_imag(l_image),
                                         l_ww,
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
                res=self.backend.complex(r,i)
            else:
                res=self.backend.nn.conv2d(l_image,l_ww,strides=[1, 1, 1, 1],padding='SAME')

            res=self.bk_reshape(res,[o_shape,shape[axis+1],shape[axis+2],ishape,norient])
        else:
            oshape=[o_shape,shape[axis+1],shape[axis+2],1]
            l_ww=self.bk_reshape(ww,[self.KERNELSZ,self.KERNELSZ,1,norient])

            tmp=self.bk_reshape(image,oshape)
            if tmp.dtype=='complex128' or tmp.dtype=='complex64':
                r=self.backend.nn.conv2d(self.bk_real(tmp),
                                         l_ww,
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
                i=self.backend.nn.conv2d(self.bk_imag(tmp),
                                         l_ww,
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
                res=self.backend.complex(r,i)
            else:
                res=self.backend.nn.conv2d(tmp,
                                           l_ww,
                                           strides=[1, 1, 1, 1],
                                           padding='SAME')

        return self.bk_reshape(res,shape+[norient])
    
    # ---------------------------------------------−---------
    def convol(self,in_image,axis=0):

        image=self.bk_cast(in_image)
        
        if self.use_R_format:
            
            if isinstance(image, Rformat):
                l_image=image
            else:
                l_image=self.to_R(image,axis=axis,chans=self.chans)
            
            nside=l_image.shape[axis+1]-2*self.R_off
            
            rr=self.conv2d(l_image.get(),self.ww_RealT[nside],axis=axis)
            ii=self.conv2d(l_image.get(),self.ww_ImagT[nside],axis=axis)

            if rr.dtype==self.all_cbk_type:
                res=self.bk_complex(1.0,0.0)*rr+self.bk_complex(0.0,1.0)*ii
            else:
                res=self.bk_complex(rr,ii) 
                
            res=Rformat(res,self.R_off,axis,chans=self.chans)
            
            if not isinstance(image, Rformat):
                res=self.from_R(res,axis=axis)
                
        else:
            nside=int(np.sqrt(image.shape[axis]//12))

            if self.Idx_convol[nside] is None:
                self.init_index(nside)

            imX9=self.bk_expand_dims(self.bk_gather(self.bk_cast(image),
                                                    self.Idx_convol[nside],axis=axis),-1)

            l_ww_real=self.ww_Real[nside]
            l_ww_imag=self.ww_Imag[nside]

            if self.do_wigner:
                for i in range(axis):
                    l_ww_real=self.bk_expand_dims(l_ww_real,0)
                    l_ww_imag=self.bk_expand_dims(l_ww_imag,0)
            else:
                for i in range(axis+1):
                    l_ww_real=self.bk_expand_dims(l_ww_real,0)
                    l_ww_imag=self.bk_expand_dims(l_ww_imag,0)
                    

            for i in range(axis+2,len(imX9.shape)-1):
                l_ww_real=self.bk_expand_dims(l_ww_real,axis+2)
                l_ww_imag=self.bk_expand_dims(l_ww_imag,axis+2)

            rr=self.bk_reduce_sum(l_ww_real*imX9,axis+1)
            ii=self.bk_reduce_sum(l_ww_imag*imX9,axis+1)

            res=self.backend.complex(1,0)*rr+self.backend.complex(0,1)*ii
            
        return(res)
        
            
    # ---------------------------------------------−---------
    def smooth(self,in_image,axis=0):

        image=self.bk_cast(in_image)
        
        if self.use_R_format:
            if isinstance(image, Rformat):
                l_image=image.get()
            else:
                l_image=self.to_R(image,axis=axis,chans=self.chans).get()
                
            nside=l_image.shape[axis+1]-2*self.R_off
            
            res=self.conv2d(l_image,self.ww_SmoothT,axis=axis)

            res=self.bk_reshape(res,l_image.shape)
            
            res=Rformat(res,self.R_off,axis,chans=self.chans)
            
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
    def get_ww(self,nside=1):
        return(self.ww_Real[nside],self.ww_Imag[nside])
    
    # ---------------------------------------------−---------
    def plot_ww(self):
        c,s=self.get_ww()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16,6))
        npt=int(np.sqrt(c.shape[0]))
        for i in range(c.shape[1]):
            plt.subplot(2,c.shape[1],1+i)
            plt.imshow(c[:,i].reshape(npt,npt),cmap='jet',vmin=-c.max(),vmax=c.max())
            plt.subplot(2,c.shape[1],1+i+c.shape[1])
            plt.imshow(s[:,i].reshape(npt,npt),cmap='jet',vmin=-c.max(),vmax=c.max())
            sys.stdout.flush()
        plt.show()
        
    
    
    
