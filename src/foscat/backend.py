import sys
import numpy as np

class foscat_backend:
    
    def __init__(self,name,mpi_rank=0,all_type='float64',gpupos=0):
        
        self.TENSORFLOW=1
        self.TORCH=2
        self.NUMPY=3
        
        self.BACKEND=name
        
        if name not in ['tensorflow','torch','numpy']:
            print('Backend %s not yet implemented')
            exit(0)
            
        if self.BACKEND=='tensorflow':
            import tensorflow as tf
            
            self.backend=tf
            self.BACKEND=self.TENSORFLOW
            #tf.config.threading.set_inter_op_parallelism_threads(1)
            #tf.config.threading.set_intra_op_parallelism_threads(1)

        if self.BACKEND=='torch':
            import torch
            self.BACKEND=self.TORCH
            self.backend=torch
            
        if self.BACKEND=='numpy':
            self.BACKEND=self.NUMPY
            self.backend=np
            
        self.float64=self.backend.float64
        self.float32=self.backend.float32
        self.int64=self.backend.int64
        self.int32=self.backend.int32
        self.complex64=self.backend.complex128
        self.complex128=self.backend.complex64
        
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
            if self.BACKEND==self.TENSORFLOW:
                print("Num GPUs Available: ", len(self.backend.config.experimental.list_physical_devices('GPU')))
            sys.stdout.flush()
        
        if self.BACKEND==self.TENSORFLOW:
            self.backend.debugging.set_log_device_placement(False)
            self.backend.config.set_soft_device_placement(True)
        
            gpus = self.backend.config.experimental.list_physical_devices('GPU')
            
        if self.BACKEND==self.TORCH:
            gpus=torch.cuda.is_available()
            
        if self.BACKEND==self.NUMPY:
            gpus=[]
        gpuname='CPU:0'
        self.gpulist={}
        self.gpulist[0]=gpuname
        self.ngpu=1
        
        if gpus:
            try:
                if self.BACKEND==self.TENSORFLOW:
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
                if self.BACKEND==self.TORCH:
                    self.ngpu=torch.cuda.device_count()
                    self.gpulist={}
                    for k in range(self.ngpu):
                        self.gpulist[k]=torch.cuda.get_device_name(0)

            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        
    # ---------------------------------------------−---------
    # --             BACKEND DEFINITION                    --
    # ---------------------------------------------−---------
    def bk_SparseTensor(self,indice,w,dense_shape=[]):
        if self.BACKEND==self.TENSORFLOW:
            return(self.backend.SparseTensor(indice,w,dense_shape=dense_shape))
        if self.BACKEND==self.TORCH:
            return(self.backend.SparseTensor(indice,w,dense_shape=dense_shape))
        if self.BACKEND==self.NUMPY:
            return np.sparse_matrix(indice,w,dense_shape=dense_shape)

    def bk_sparse_dense_matmul(self,smat,mat):
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.sparse.sparse_dense_matmul(smat,mat) 
        if self.BACKEND==self.TORCH:
            return self.backend.sparse.sparse_dense_matmul(smat,mat)
        if self.BACKEND==self.NUMPY:
            return np.sparse.sparse_dense_matmul(smat,mat)

    def conv2d(self,x,w,strides=[1, 1, 1, 1],padding='SAME'):
        if self.BACKEND==self.TENSORFLOW:
                return self.backend.nn.conv2d(x,w,
                                               strides=strides,
                                               padding=padding)
        # to be written!!!
        if self.BACKEND==self.TORCH:
            return x
        if self.BACKEND==self.NUMPY:
            return x

    def bk_threshold(self,x,threshold,greater=True):

        if self.BACKEND==self.TENSORFLOW:
            return(self.backend.cast(x>threshold,x.dtype)*x)
        if self.BACKEND==self.TORCH:
            return(self.backend.cast(x>threshold,x.dtype)*x)
        if self.BACKEND==self.NUMPY:
            return (x>threshold)*x

    def bk_maximum(self,x1,x2):
        if self.BACKEND==self.TENSORFLOW:
            return(self.backend.maximum(x1,x2))
        if self.BACKEND==self.TORCH:
            return(self.backend.maximum(x1,x2))
        if self.BACKEND==self.NUMPY:
            return x1*(x1>x2)+x2*(x2>x1)
        
    def bk_device(self,device_name):
        return self.backend.device(device_name)
        
    def bk_ones(self,shape,dtype=None):
        if dtype is None:
            dtype=self.all_type
        return(self.backend.ones(shape,dtype=dtype))
    
    def bk_conv1d(self,x,w):
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.nn.conv1d(x,w, stride=[1,1,1], padding='SAME')
        if self.BACKEND==self.TORCH:
            return self.backend.nn.conv1d(x,w, stride=1, padding='SAME')
        if self.BACKEND==self.NUMPY:
            return self.backend.nn.conv1d(x,w, stride=1, padding='SAME')
        
    def bk_flattenR(self,x):
        if self.BACKEND==self.TENSORFLOW or self.BACKEND==self.TORCH:
            if x.dtype=='complex32' or x.dtype=='complex64':
                rr=self.backend.reshape(self.bk_real(x),[np.prod(np.array(list(x.shape)))])
                ii=self.backend.reshape(self.bk_imag(x),[np.prod(np.array(list(x.shape)))])
                return self.bk_concat([rr,ii],axis=0) 
            else:
                return self.backend.reshape(x,[np.prod(np.array(list(x.shape)))])
            
        if self.BACKEND==self.NUMPY:
            if x.dtype=='complex32' or x.dtype=='complex64':
                return np.concatenate([x.real.flatten(),x.imag.flatten()],0)
            else:
                return x.flatten()
        
    def bk_flatten(self,x):
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.reshape(x,[np.prod(np.array(list(x.shape)))])
        if self.BACKEND==self.TORCH:
            return self.backend.reshape(x,[np.prod(np.array(list(x.shape)))])
        if self.BACKEND==self.NUMPY:
            return x.flatten()
        
    def bk_resize_image(self,x,shape):
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.image.resize(x,shape, method='bilinear')
        if self.BACKEND==self.TORCH:
            return self.backend.image.resize(x,shape, method='bilinear')
        if self.BACKEND==self.NUMPY:
            return self.backend.image.resize(x,shape, method='bilinear')
        
    def bk_L1(self,x):
        if x.dtype==self.all_cbk_type:
            xr=self.bk_real(x)
            xi=self.bk_imag(x)
                
            r=self.backend.sign(xr)*self.backend.sqrt(self.backend.sign(xr)*xr)
            i=self.backend.sign(xi)*self.backend.sqrt(self.backend.sign(xi)*xi)
            return self.bk_complex(r,i)
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
        
    def constant(self,data):
        
        if self.BACKEND==self.TENSORFLOW:
            return(self.backend.constant(data))
        return(data)

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

    def bk_reduce_std(self,data,axis=None):
        
        if axis is None:
            if self.BACKEND==self.TENSORFLOW:
                return(self.backend.math.reduce_std(data))
            if self.BACKEND==self.TORCH:
                return(self.backend.std(data))
            if self.BACKEND==self.NUMPY:
                return(np.std(data))
        else:
            if self.BACKEND==self.TENSORFLOW:
                return(self.backend.math.reduce_std(data,axis=axis))
            if self.BACKEND==self.TORCH:
                return(self.backend.std(data,axis))
            if self.BACKEND==self.NUMPY:
                return(np.std(data,axis))
        
    
    def bk_sqrt(self,data):
        
        return(self.backend.sqrt(self.backend.abs(data)))
    
    def bk_abs(self,data):
        return(self.backend.abs(data))

    def bk_is_complex(self,data):
        if self.BACKEND==self.TENSORFLOW:
            return data.dtype==self.all_cbk_type
        if self.BACKEND==self.TORCH:
            return data.dtype==self.all_cbk_type
        if self.BACKEND==self.NUMPY:
            return data.dtype==self.all_cbk_type
        
    def bk_norm(self,data):
        if self.bk_is_complex(data):
            res=self.bk_square(self.bk_real(data))+self.bk_square(self.bk_imag(data))
            return self.bk_sqrt(res)

        else:
            return self.bk_abs(data)
        
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

    def bk_matmul(self,a,b):
        if self.BACKEND==self.TENSORFLOW:
            return(self.backend.matmul(a,b))
        if self.BACKEND==self.TORCH:
            return(self.backend.matmul(a,b))
        if self.BACKEND==self.NUMPY:
            return(np.dot(a,b))

    def bk_tensor(self,data):
        if self.BACKEND==self.TENSORFLOW:
            return(self.backend.constant(data))
        if self.BACKEND==self.TORCH:
            return(self.backend.constant(data))
        if self.BACKEND==self.NUMPY:
            return(data)
        
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

    def bk_exp(self,data):
        
        return(self.backend.exp(data))
    
    def bk_min(self,data):
        
        return(self.backend.reduce_min(data))
    
    def bk_argmin(self,data):
        
        return(self.backend.argmin(data))
    
    def bk_tanh(self,data):
        
        return(self.backend.math.tanh(data))
    
    def bk_max(self,data):
        
        return(self.backend.reduce_max(data))
    
    def bk_argmax(self,data):
        
        return(self.backend.argmax(data))
    
    def bk_reshape(self,data,shape):
        return(self.backend.reshape(data,shape))
    
    def bk_repeat(self,data,nn,axis=0):
        return(self.backend.repeat(data,nn,axis=axis))
    
    def bk_tile(self,data,nn,axis=0):
        return(self.backend.tile(data,nn))
    
    def bk_roll(self,data,nn,axis=0):
        return(self.backend.roll(data,nn,axis=axis))

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

    
    def bk_conjugate(self,data):
                
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.math.conj(data)
        if self.BACKEND==self.TORCH:
            return self.backend.conjugate(data)
        if self.BACKEND==self.NUMPY:
            return data.conjugate()
        
    def bk_real(self,data):
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.math.real(data)
        if self.BACKEND==self.TORCH:
            return self.backend.real(data)
        if self.BACKEND==self.NUMPY:
            return self.backend.real(data)

    def bk_imag(self,data):
        if self.BACKEND==self.TENSORFLOW:
            return self.backend.math.imag(data)
        if self.BACKEND==self.TORCH:
            return self.backend.imag(data)
        if self.BACKEND==self.NUMPY:
            return self.backend.imag(data)
        
    def bk_relu(self,x):
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
        if isinstance(x,np.float64):
            if self.all_bk_type=='float32':
                return(np.float32(x))
            else:
                return(x)
        if isinstance(x,np.float32):
            if self.all_bk_type=='float64':
                return(np.float64(x))
            else:
                return(x)

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
