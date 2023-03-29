import tensorflow as tf
import numpy as np
import time
import sys
import os
from datetime import datetime
from packaging import version
from threading import Thread
from threading import Event

class Loss:
    
    def __init__(self,function,scat_operator,*param,Rformat=True,name=''):

        self.loss_function=function
        self.scat_operator=scat_operator
        self.args=param
        self.Rformat=Rformat
        self.name=name

    def eval(self,x):
        return self.loss_function(x,self.scat_operator,self.args)
    
class Synthesis:
    def __init__(self,
                 loss_list,
                 eta=0.03,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-7,
                 decay_rate = 0.999,
                 MAXNUMLOSS=10):

        self.loss_class=loss_list
        self.number_of_loss=len(loss_list)
        self.nlog=0
        self.m_dw, self.v_dw = 0.0, 0.0
        self.beta1 = beta1
        self.beta2 = beta2
        self.pbeta1 = beta1
        self.pbeta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.history=np.zeros([10])
        self.curr_gpu=0
        self.event = Event()
        self.operation=loss_list[0].scat_operator
        self.mpi_size=self.operation.mpi_size
        self.mpi_rank=self.operation.mpi_rank
        self.MAXNUMLOSS=MAXNUMLOSS
    
    # ---------------------------------------------−---------
    def get_gpu(self,event,delay):

        isnvidia=os.system('which nvidia-smi &> /dev/null')

        while (1):
            if event.is_set():
                break
            time.sleep(delay)
            if isnvidia==0:
                try:
                    os.system("nvidia-smi | awk '$2==\"N/A\"{print substr($9,1,length($9)-3),substr($11,1,length($11)-3),substr($13,1,length($13)-1)}' > smi_tmp.txt")
                except:
                    nogpu=1
       
    def stop_synthesis(self):
        # stop thread that catch GPU information
        self.event.set()
        
        try:
            self.gpu_thrd.join()
        except:
            print('No thread to stop, everything is ok')
            sys.stdout.flush()

    # ---------------------------------------------−---------
    def check_dense(self,data,datasz):
        if isinstance(data, tf.Tensor):
            return data
        
        idx=tf.cast(data.indices, tf.int32)
        data=tf.math.bincount(idx,weights=data.values,
                              minlength=datasz)
        return data

    @tf.function
    def loss(self,x,loss_function):

        operation=loss_function.scat_operator

        nx=1
        if len(x.shape)>1:
            nx=x.shape[0]
            
        with tf.device(operation.gpulist[(operation.gpupos+self.curr_gpu)%operation.ngpu]):
            print('%s Run on GPU %s'%(loss_function.name,
                                      operation.gpulist[(operation.gpupos+self.curr_gpu)%operation.ngpu]))

            if nx>1:
                l_x={}
            for i in range(nx):
                if nx==1:
                    if operation.get_use_R() and loss_function.Rformat:
                        l_x=operation.to_R(x,only_border=True,chans=operation.chans)
                    else:
                        l_x=x
                    ndata=x.shape[0]
                else:
                    if operation.get_use_R() and loss_function.Rformat:
                        l_x[i]=operation.to_R(x[i],only_border=True,chans=operation.chans)
                    else:
                        l_x[i]=x[i]

                    ndata=x.shape[0]*x.shape[1]
                    
                
            l=loss_function.eval(l_x)
            g=self.check_dense(tf.gradients(l,x)[0],ndata)
            self.curr_gpu=self.curr_gpu+1
            
            
        return l,g
    #---------------------------------------------------------------
    
    def gradient(self,x):

        if self.operation.get_use_R():
            x=self.operation.to_R_center(x,chans=operation.chans)
            
        g_tot=None
        l_tot=0.0
        for k in range(self.number_of_loss):
            l,g=self.loss(x,self.loss_class[k])
            if g_tot is None:
                g_tot=g
            else:
                g_tot=g_tot+g

        if self.mpi_size==1:
            grad=g_tot
        else:
            grad=np.zeros([g_tot.shape[0]],dtype='float32')
            comm.Allreduce((g_tot.numpy(),MPI.FLOAT),(grad,MPI.FLOAT))
            
        operation=self.operation
        if operation.get_use_R():
            l_x=operation.to_R(grad,only_border=True,chans=operation.chans)
            x=operation.from_R(l_x)
        else:
            x=grad
            
        return x
        
    #---------------------------------------------------------------
    
    def update(self, dw):
        ## dw are from current minibatch
        ## momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)

        ## bias correction
        m_dw_corr = self.m_dw/(1-self.pbeta1)
        v_dw_corr = self.v_dw/(1-self.pbeta2)

        self.pbeta1 = self.beta1*self.pbeta1
        self.pbeta2 = self.beta2*self.pbeta2

        self.eta    = self.eta*self.decay_rate

        ## update weights and biases
        return self.eta*(m_dw_corr/(tf.sqrt(v_dw_corr)+self.epsilon))

    # ---------------------------------------------−---------
    def getgpumem(self):
        try:
            return np.loadtxt('smi_tmp.txt')
        except:
            return(np.zeros([1,3]))
        
    # ---------------------------------------------−---------
    def run(self,
            in_x,
            NUM_EPOCHS = 1000,
            DECAY_RATE=0.95,
            EVAL_FREQUENCY = 100,
            DEVAL_STAT_FREQUENCY = 1000,
            LEARNING_RATE = 0.03,
            EPSILON = 1E-7,
            grd_mask=None,
            SHOWGPU=False,
            MESSAGE='',
            axis=0):
        
        self.eta=LEARNING_RATE
        self.epsilon=EPSILON
        self.decay_rate = DECAY_RATE
        self.nlog=0

        if self.operation.get_use_R():
            if axis==0:
                x=self.operation.to_R_center(self.operation.bk_cast(in_x),chans=self.operation.chans)
            else:
                tmp_x=self.operation.to_R_center(self.operation.bk_cast(in_x[0]),chans=self.operation.chans)
                x=np.zeros([in_x.shape[0],tmp_x.shape[0]],dtype=self.operation.all_type)
                x[0]=tmp_x
                del tmp_x
                for i in range(1,in_x.shape[0]):
                    x[i]=self.operation.to_R_center(self.operation.bk_cast(in_x[i]),chans=self.operation.chans)
            
        self.curr_gpu=self.curr_gpu+self.mpi_rank
        
        if self.mpi_size>1:
            print('Work with MPI')
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            
        if self.mpi_rank==0 and SHOWGPU:
            # start thread that catch GPU information
            try:
                self.gpu_thrd = Thread(target=self.get_gpu, args=(self.event,1,))
                self.gpu_thrd.start()
            except:
                print("Error: unable to start thread for GPU survey")
            
        start = time.time()

        if self.mpi_size>1:
            num_loss=np.zeros([1],dtype='int32')
            total_num_loss=np.zeros([1],dtype='int32')
            num_loss[0]=self.number_of_loss
            comm.Allreduce((num_loss,MPI.INT),(total_num_loss,MPI.INT))
            total_num_loss=total_num_loss[0]
        else:
            total_num_loss=self.number_of_loss
            
        if self.mpi_rank==0:
            print('Total number of loss ',total_num_loss)
            sys.stdout.flush()
        
        l_log=np.zeros([self.mpi_size*self.MAXNUMLOSS],dtype='float32')
        l_log[self.mpi_rank*self.MAXNUMLOSS:(self.mpi_rank+1)*self.MAXNUMLOSS]=-1.0
        ltot=1.0*l_log
        
        for itt in range(NUM_EPOCHS):
            g_tot=None
            l_tot=0.0
            for k in range(self.number_of_loss):
                l,g=self.loss(x,self.loss_class[k])
                if grd_mask is not None:
                    g=grd_mask*g.numpy()
                else:
                    g=g.numpy()
                g[np.isnan(g)]=0.0
                if g_tot is None:
                    g_tot=g
                else:
                    g_tot=g_tot+g

                    
                l_tot=l_tot+l.numpy()
            
                l_log[self.mpi_rank*self.MAXNUMLOSS+k]=l.numpy()

            
            if self.mpi_size==1:
                ltot=l_log
            else:
                comm.Allreduce((l_log,MPI.FLOAT),(ltot,MPI.FLOAT))
            

            if self.mpi_size==1:
                grad=g_tot
            else:
                if axis==0:
                    grad=np.zeros([g_tot.shape[0]],dtype=self.operation.get_type())
                else:
                    grad=np.zeros([g_tot.shape[0],g_tot.shape[1]],dtype=self.operation.get_type())
                    
                comm.Allreduce((g_tot.astype(self.operation.get_type()),self.operation.get_mpi_type()),
                               (grad,self.operation.get_mpi_type()))
            
            if self.nlog==self.history.shape[0]:
                new_log=np.zeros([self.history.shape[0]*2])
                new_log[0:self.nlog]=self.history
                self.history=new_log
                
            self.history[self.nlog]=ltot[ltot!=-1].sum()
            self.nlog=self.nlog+1
                
            x=x-self.update(grad)
            
            if itt%EVAL_FREQUENCY==0 and self.mpi_rank==0:
                end = time.time()
                cur_loss='%.3g ('%(ltot[ltot!=-1].sum())
                for k in ltot[ltot!=-1]:
                    cur_loss=cur_loss+'%.3g '%(k)
                cur_loss=cur_loss+')'
                
                mess=''
                if SHOWGPU:
                    info_gpu=self.getgpumem()
                    for k in range(info_gpu.shape[0]):
                        mess=mess+'[GPU%d %.0f/%.0f MB %.0f%%]'%(k,info_gpu[k,0],info_gpu[k,1],info_gpu[k,2])
                
                print('%sItt %d L=%s %.3fs %s'%(MESSAGE,itt,cur_loss,(end-start),mess))
                sys.stdout.flush()
                start = time.time()

        if self.mpi_rank==0 and SHOWGPU:
            self.stop_synthesis()
        
        operation=self.operation
        if operation.get_use_R():
            if axis==0:
                l_x=operation.to_R(x,only_border=True,chans=self.operation.chans)
                x=operation.from_R(l_x)
            else:
                l_x=operation.to_R(x[0],only_border=True,chans=self.operation.chans)
                tmp_x=operation.from_R(l_x)
                out_x=np.zeros([x.shape[0],tmp_x.shape[0]])
                out_x[0]=tmp_x
                del tmp_x
                for i in range(1,in_x.shape[0]):
                    l_x=operation.to_R(x[i],only_border=True,chans=self.operation.chans)
                    out_x[i]=operation.from_R(l_x)
                x=out_x
                    
        return(x)

    def get_history(self):
        return(self.history[0:self.nlog])
