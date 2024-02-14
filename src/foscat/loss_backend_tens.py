import tensorflow as tf
import numpy as np
import sys

class loss_backend:
    
    def __init__(self,backend,curr_gpu,mpi_rank):
        
        self.bk=backend
        self.curr_gpu=curr_gpu
        self.mpi_rank=mpi_rank

    
    def check_dense(self,data,datasz):
        if isinstance(data, tf.Tensor):
            return data
        
        return data.to_dense()
        
    # ---------------------------------------------−---------
    
    @tf.function
    def loss(self,x,batch,loss_function,KEEP_TRACK):

        operation=loss_function.scat_operator

        nx=1
        if len(x.shape)>1:
            nx=x.shape[0]
            
        with tf.device(operation.gpulist[(operation.gpupos+self.curr_gpu)%operation.ngpu]):
            print('%s Run [PROC=%04d] on GPU %s'%(loss_function.name,self.mpi_rank,
                                      operation.gpulist[(operation.gpupos+self.curr_gpu)%operation.ngpu]))
            sys.stdout.flush()

            l_x=x
            """
            if nx>1:
                l_x={}
            for i in range(nx):
            """
            
            if nx==1:    
                ndata=x.shape[0]
            else:
                ndata=x.shape[0]*x.shape[1]
                
            if KEEP_TRACK is not None:
                l,linfo=loss_function.eval(l_x,batch,return_all=True)
            else:
                l=loss_function.eval(l_x,batch)
                
            g=tf.gradients(l,x)[0]
            g=self.check_dense(g,ndata)
            self.curr_gpu=self.curr_gpu+1   
            
        if KEEP_TRACK is not None:
            return l,g,linfo
        else:
            return l,g
