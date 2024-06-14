import numpy as np
import time
import sys
import os
from datetime import datetime
from packaging import version
from threading import Thread
from threading import Event
import scipy.optimize as opt

class Loss:
    
    def __init__(self,function,scat_operator,*param,
                 name='',
                 batch=None,
                 batch_data=None,
                 batch_update=None,
                 info_callback=False):

        self.loss_function=function
        self.scat_operator=scat_operator
        self.args=param
        self.name=name
        self.batch=batch
        self.batch_data=batch_data
        self.batch_update=batch_update
        self.info=info_callback
        
    def eval(self,x,batch,return_all=False):
        if self.batch is None:
            if self.info:
                return self.loss_function(x,self.scat_operator,self.args,return_all=return_all)
            else:
                return self.loss_function(x,self.scat_operator,self.args)
        else:
            if self.info:
                return self.loss_function(x,batch,self.scat_operator,self.args,return_all=return_all)
            else:
                return self.loss_function(x,batch,self.scat_operator,self.args)
        
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
        self.__iteration__=1234
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
        self.KEEP_TRACK=None
        self.MAXNUMLOSS=len(loss_list)
    
        if self.operation.BACKEND=='tensorflow':
            import foscat.loss_backend_tens as fbk
            self.bk=fbk.loss_backend(self.operation,self.curr_gpu,self.mpi_rank)
            
        if self.operation.BACKEND=='torch':
            import foscat.loss_backend_torch as fbk
            self.bk=fbk.loss_backend(self.operation,self.curr_gpu,self.mpi_rank)
            
        if self.operation.BACKEND=='numpy':
            print('Synthesis does not work with numpy. Please select Torch or Tensorflow FOSCAT backend')
            return None

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
    def getgpumem(self):
        try:
            return np.loadtxt('smi_tmp.txt')
        except:
            return(np.zeros([1,3]))
        
    # ---------------------------------------------−---------
    def info_back(self,x):
            
        self.nlog=self.nlog+1
        self.itt2=0
            
        if self.itt%self.EVAL_FREQUENCY==0 and self.mpi_rank==0:
            end = time.time()
            cur_loss='%10.3g ('%(self.ltot[self.ltot!=-1].mean())
            for k in self.ltot[self.ltot!=-1]:
                cur_loss=cur_loss+'%10.3g '%(k)
                
            cur_loss=cur_loss+')'
                
            mess=''
                
            if self.SHOWGPU:
                info_gpu=self.getgpumem()
                for k in range(info_gpu.shape[0]):
                    mess=mess+'[GPU%d %.0f/%.0f MB %.0f%%]'%(k,info_gpu[k,0],info_gpu[k,1],info_gpu[k,2])
                
            
            print('%sItt %6d L=%s %.3fs %s'%(self.MESSAGE,self.itt,cur_loss,(end-self.start),mess))
            sys.stdout.flush()
            if self.KEEP_TRACK is not None:
                print(self.last_info)
                sys.stdout.flush()
                
            self.start = time.time()
            
        self.itt=self.itt+1

    # ---------------------------------------------−---------
    def calc_grad(self,in_x):
        
        g_tot=None
        l_tot=0.0

            
        if self.do_all_noise and self.totalsz>self.batchsz:
            nstep=self.totalsz//self.batchsz
        else:
            nstep=1

        x=self.operation.backend.bk_reshape(self.operation.backend.bk_cast(in_x),self.oshape)
        
        self.l_log[self.mpi_rank*self.MAXNUMLOSS:(self.mpi_rank+1)*self.MAXNUMLOSS]=-1.0
        
        for istep in range(nstep):
                    
            for k in range(self.number_of_loss):
                if self.loss_class[k].batch is None:
                    l_batch=None
                else:
                    l_batch=self.loss_class[k].batch(self.loss_class[k].batch_data,istep)

                if self.KEEP_TRACK is not None:
                    l,g,linfo=self.bk.loss(x,l_batch,self.loss_class[k],self.KEEP_TRACK)
                    self.last_info=self.KEEP_TRACK(linfo,self.mpi_rank,add=True)
                else:
                    l,g=self.bk.loss(x,l_batch,self.loss_class[k],self.KEEP_TRACK)

                if g_tot is None:
                    g_tot=g
                else:
                    g_tot=g_tot+g

                l_tot=l_tot+l.numpy()

                if self.l_log[self.mpi_rank*self.MAXNUMLOSS+k]==-1:
                    self.l_log[self.mpi_rank*self.MAXNUMLOSS+k]=l.numpy()/nstep
                else:
                    self.l_log[self.mpi_rank*self.MAXNUMLOSS+k]=self.l_log[self.mpi_rank*self.MAXNUMLOSS+k]+l.numpy()/nstep
                
        grd_mask=self.grd_mask
            
        if grd_mask is not None:
            g_tot=grd_mask*g_tot.numpy()
        else:
            g_tot=g_tot.numpy()
            
        g_tot[np.isnan(g_tot)]=0.0

        self.imin=self.imin+self.batchsz

        if self.mpi_size==1:
            self.ltot=self.l_log
        else:
            local_log=(self.l_log).astype('float64')
            self.ltot=np.zeros(self.l_log.shape,dtype='float64')
            self.comm.Allreduce((local_log,self.MPI.DOUBLE),(self.ltot,self.MPI.DOUBLE))
            
        if self.mpi_size==1:
            grad=g_tot
        else:
            if self.operation.backend.bk_is_complex( g_tot):
                grad=np.zeros(self.oshape,dtype=gtot.dtype)

                self.comm.Allreduce((g_tot),(grad))
            else:
                grad=np.zeros(self.oshape,dtype='float64')

                self.comm.Allreduce((g_tot.astype('float64'),self.MPI.DOUBLE),
                                    (grad,self.MPI.DOUBLE))
        
        if self.nlog==self.history.shape[0]:
            new_log=np.zeros([self.history.shape[0]*2])
            new_log[0:self.nlog]=self.history
            self.history=new_log

        l_tot=self.ltot[self.ltot!=-1].mean()
        
        self.history[self.nlog]=l_tot

        g_tot=grad.flatten()

        if self.operation.backend.bk_is_complex( g_tot):
            return l_tot.astype('float64'),g_tot
        
        return l_tot.astype('float64'),g_tot.astype('float64')

    # ---------------------------------------------−---------
    def xtractmap(self,x,axis):
        x=self.operation.backend.bk_reshape(x,self.oshape)
        
        return x

    # ---------------------------------------------−---------
    def run(self,
            in_x,
            NUM_EPOCHS = 100,
            DECAY_RATE=0.95,
            EVAL_FREQUENCY = 100,
            DEVAL_STAT_FREQUENCY = 1000,
            NUM_STEP_BIAS = 1,
            LEARNING_RATE = 0.03,
            EPSILON = 1E-7,
            KEEP_TRACK=None,
            grd_mask=None,
            SHOWGPU=False,
            MESSAGE='',
            factr=10.0,
            batchsz=1,
            totalsz=1,
            do_lbfgs=True,
            axis=0):
        
        self.KEEP_TRACK=KEEP_TRACK
        self.track={}
        self.ntrack=0
        self.eta=LEARNING_RATE
        self.epsilon=EPSILON
        self.decay_rate = DECAY_RATE
        self.nlog=0
        self.itt2=0
        self.batchsz=batchsz
        self.totalsz=totalsz
        self.grd_mask=grd_mask
        self.EVAL_FREQUENCY=EVAL_FREQUENCY
        self.MESSAGE=MESSAGE
        self.SHOWGPU=SHOWGPU
        self.axis=axis
        self.in_x_nshape=in_x.shape[0]
        self.seed=1234

        np.random.seed(self.mpi_rank*7+1234)
            
        x=in_x        
                    
        self.curr_gpu=self.curr_gpu+self.mpi_rank
        
        if self.mpi_size>1:
            from mpi4py import MPI
            

            comm = MPI.COMM_WORLD
            self.comm=comm
            self.MPI=MPI
            if self.mpi_rank==0:
                print('Work with MPI')
                sys.stdout.flush()
            
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
        self.ltot=l_log.copy()
        self.l_log=l_log
        
        self.imin=0
        self.start=time.time()
        self.itt=0
        
        self.oshape=list(x.shape)
        
        if not isinstance(x,np.ndarray):
            x=x.numpy()
            
        x=x.flatten()

        self.do_all_noise=False

        self.do_all_noise=True

        self.noise_idx=None

        for k in range(self.number_of_loss):
            if self.loss_class[k].batch is not None:
                l_batch=self.loss_class[k].batch(self.loss_class[k].batch_data,0,init=True)
        
        l_tot,g_tot=self.calc_grad(x)

        self.info_back(x)

        maxitt=NUM_EPOCHS

        start_x=x.copy()

        for iteration in range(NUM_STEP_BIAS):
        
            x,l,i=opt.fmin_l_bfgs_b(self.calc_grad,
                                    x.astype('float64'),
                                    callback=self.info_back,
                                    pgtol=1E-32,
                                    factr=factr,
                                    maxiter=maxitt)

            # update bias input data
            if iteration<NUM_STEP_BIAS-1:
                #if self.mpi_rank==0:
                #    print('%s Hessian restart'%(self.MESSAGE))

                omap=self.xtractmap(x,axis)

                for k in range(self.number_of_loss):
                    if self.loss_class[k].batch_update is not None:
                        self.loss_class[k].batch_update(self.loss_class[k].batch_data,omap)
                    if self.loss_class[k].batch is not None:
                        l_batch=self.loss_class[k].batch(self.loss_class[k].batch_data,0,init=True)
                #x=start_x.copy()

        if self.mpi_rank==0 and SHOWGPU:
            self.stop_synthesis()

        if self.KEEP_TRACK is not None:
            self.last_info=self.KEEP_TRACK(None,self.mpi_rank,add=False)

        x=self.xtractmap(x,axis)
        return(x)

    def get_history(self):
        return(self.history[0:self.nlog])
