import numpy as np
import tensorflow as tf

class adam():
    def __init__(self,
                 eta=0.03,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-7,
                 decay_rate = 0.999):
        self.m_dw, self.v_dw = 0.0, 0.0
        self.beta1 = beta1
        self.beta2 = beta2
        self.pbeta1 = beta1
        self.pbeta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.decay_rate = decay_rate

    def get_lr(self):
        return self.eta
    
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
