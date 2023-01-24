import tensorflow as tf
import numpy as np
import time

class Loss:
    
    def __init__(self,function,*param):

        self.loss_function=function
        self.args=param

    def eval(self,x):
        return self.loss_function(x,self.args)
    
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
        
    
    # ---------------------------------------------−---------
    def check_dense(self,data,datasz):
        s='%s'%(type(data))
        if 'Index' in s:
            idx=tf.cast(data.indices, tf.int32)
            print(idx)
            data=tf.math.bincount(idx,weights=data.values,
                                  minlength=datasz)
        return data

    @tf.function
    def loss(self,x,loss_function):

        l=loss_function.eval(x)
        
        g=self.check_dense(tf.gradients(l,x)[0],x)
            
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
    def run(self,
            x,
            NUM_EPOCHS = 1000,
            DECAY_RATE=0.95,
            EVAL_FREQUENCY = 100,
            DEVAL_STAT_FREQUENCY = 1000,
            LEARNING_RATE = 0.03,
            EPSILON = 1E-7):
        
        self.eta=LEARNING_RATE
        self.epsilon=EPSILON
        self.decay_rate = DECAY_RATE
        self.nlog=0
        
        start = time.time()
        
        for itt in range(NUM_EPOCHS):
            grad=None
            ltot=np.zeros([self.number_of_loss])
            for k in range(self.number_of_loss):
                l,g=self.loss(x,self.loss_class[k])
                if grad is None:
                    grad=g
                else:
                    grad=grad+g

                ltot[k]=l.numpy()
                    
            if self.nlog==self.history.shape[0]:
                new_log=np.zeros([self.history.shape[0]*2])
                new_log[0:self.nlog]=self.history
                self.history=new_log
            self.history[self.nlog]=ltot.sum()
            self.nlog=self.nlog+1
                
            x=x-self.update(g)
            
            if itt%EVAL_FREQUENCY==0:
                end = time.time()
                cur_loss='%.3g ('%(ltot.sum())
                for k in range(self.number_of_loss):
                    cur_loss=cur_loss+'%.3g '%(ltot[k])
                cur_loss=cur_loss+')'
                
                print('Itt %d L=%s %.3fs'%(itt,cur_loss,(end-start)))
                start = time.time()
                
        return(x)

    def get_history(self):
        return(self.history[0:self.nlog])
