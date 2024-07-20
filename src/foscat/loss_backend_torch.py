import torch
from torch.autograd import grad
import numpy as np
import sys

class loss_backend:
    
    def __init__(self,backend,curr_gpu,mpi_rank):
        
        self.bk=backend
        self.curr_gpu=curr_gpu
        self.mpi_rank=mpi_rank

    
    def check_dense(self,data,datasz):
        if isinstance(data, torch.Tensor):
            return data
        """
        idx=tf.cast(data.indices, tf.int32)
        data=tf.math.bincount(idx,weights=data.values,
                              minlength=datasz)
        """
        return data
        
    # ---------------------------------------------−---------

    def loss(self,x,batch,loss_function,KEEP_TRACK):

        operation=loss_function.scat_operator

        nx=1
        if len(x.shape)>1:
            nx=x.shape[0]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            with torch.cuda.device((operation.gpupos+self.curr_gpu)%operation.ngpu):

                l_x=x.clone().detach().requires_grad_(True)
            
                if nx==1:    
                    ndata=x.shape[0]
                else:
                    ndata=x.shape[0]*x.shape[1]
                
                if KEEP_TRACK is not None:
                    l,linfo=loss_function.eval(l_x,batch,return_all=True)
                else:
                    l=loss_function.eval(l_x,batch)

                l.backward()
            
                g=l_x.grad
            
                self.curr_gpu=self.curr_gpu+1
        else:
            l_x=x.clone().detach().requires_grad_(True)

            if nx==1:    
                ndata=x.shape[0]
            else:
                ndata=x.shape[0]*x.shape[1]

            if KEEP_TRACK is not None:
                l,linfo=loss_function.eval(l_x,batch,return_all=True)
            else:
                """
                sx=operation.eval(l_x)
                tmp=(sx.C01-1.0)
    
                l=operation.backend.bk_reduce_sum(tmp*tmp) #loss_function.eval(l_x,batch)
                """
                
                l=loss_function.eval(l_x,batch)

            l.backward()
            
            g=l_x.grad
            
        if KEEP_TRACK is not None:
            return l.detach(),g,linfo
        else:
            return l.detach(),g

