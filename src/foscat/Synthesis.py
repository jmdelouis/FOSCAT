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
    
    def __init__(self,function,scat_operator,*param):

        self.loss_function=function
        self.scat_operator=scat_operator
        self.args=param

    def eval(self,x):
        return self.loss_function(x,self.scat_operator,self.args)
    
class Synthesis:
    def __init__(self,
                 loss_list,
                 eta=0.03,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-7,
                 decay_rate = 0.999):

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
        
        with tf.device(operation.gpulist[(operation.gpupos+self.curr_gpu)%operation.ngpu]):
            print('Run on GPU %s'%(operation.gpulist[(operation.gpupos+self.curr_gpu)%operation.ngpu]))
            
            if operation.get_use_R():
                l_x=operation.to_R(x,only_border=True)
            else:
                l_x=x
                
            l=loss_function.eval(l_x)
            
            g=self.check_dense(tf.gradients(l,x)[0],x.shape[0])
            
            self.curr_gpu=self.curr_gpu+1
            
        return l,g
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
            x,
            NUM_EPOCHS = 1000,
            DECAY_RATE=0.95,
            EVAL_FREQUENCY = 100,
            DEVAL_STAT_FREQUENCY = 1000,
            LEARNING_RATE = 0.03,
            EPSILON = 1E-7,
            mpi_size=1,
            mpi_rank=0):
        
        self.eta=LEARNING_RATE
        self.epsilon=EPSILON
        self.decay_rate = DECAY_RATE
        self.nlog=0

        if self.operation.get_use_R():
            x=self.operation.to_R_center(x)
            
        self.curr_gpu=self.curr_gpu+mpi_rank
        
        if mpi_size>1:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            
        if mpi_rank==0:
            # start thread that catch GPU information
            try:
                self.gpu_thrd = Thread(target=self.get_gpu, args=(self.event,1,))
                self.gpu_thrd.start()
            except:
                print("Error: unable to start thread for GPU survey")
            
        start = time.time()

        if mpi_size>1:
            num_loss=np.zeros([1],dtype='int32')
            total_num_loss=np.zeros([1],dtype='int32')
            num_loss[0]=self.number_of_loss
            comm.Allreduce((num_loss,MPI.INT),(total_num_loss,MPI.INT))
            total_num_loss=total_num_loss[0]
        else:
            total_num_loss=self.number_of_loss
            
        if mpi_rank==0:
            print('Total number of loss ',total_num_loss)
            
        for itt in range(NUM_EPOCHS):
            g_tot=None
            l_tot=0.0
            for k in range(self.number_of_loss):
                l,g=self.loss(x,self.loss_class[k])
                if g_tot is None:
                    g_tot=g
                else:
                    g_tot=g_tot+g
                l_tot=l_tot+l.numpy()
                
            l_log=np.zeros([mpi_size],dtype='float32')
            ltot=np.zeros([mpi_size],dtype='float32')
            l_log[mpi_rank]=l.numpy()
            
            if mpi_size==1:
                ltot=l_log
            else:
                comm.Allreduce((l_log,MPI.FLOAT),(ltot,MPI.FLOAT))
            

            if mpi_size==1:
                grad=g
            else:
                grad=np.zeros([g.shape[0]],dtype='float32')
                comm.Allreduce((g.numpy(),MPI.FLOAT),(grad,MPI.FLOAT))
            
            if self.nlog==self.history.shape[0]:
                new_log=np.zeros([self.history.shape[0]*2])
                new_log[0:self.nlog]=self.history
                self.history=new_log
                
            self.history[self.nlog]=ltot.sum()
            self.nlog=self.nlog+1
                
            x=x-self.update(grad)
            
            if itt%EVAL_FREQUENCY==0 and mpi_rank==0:
                end = time.time()
                cur_loss='%.3g ('%(ltot.sum())
                for k in range(ltot.shape[0]):
                    cur_loss=cur_loss+'%.3g '%(ltot[k])
                cur_loss=cur_loss+')'
                
                info_gpu=self.getgpumem()
                mess=''
                for k in range(info_gpu.shape[0]):
                    mess=mess+'[GPU%d %.0f/%.0f MB %.0f%%]'%(k,info_gpu[k,0],info_gpu[k,1],info_gpu[k,2])
                
                print('Itt %d L=%s %.3fs %s'%(itt,cur_loss,(end-start),mess))
                sys.stdout.flush()
                start = time.time()

        if mpi_rank==0:
            self.stop_synthesis()
        
        operation=self.operation
        if operation.get_use_R():
            l_x=operation.to_R(x,only_border=True)
            x=operation.from_R(l_x)
            
        return(x)

    def get_history(self):
        return(self.history[0:self.nlog])
