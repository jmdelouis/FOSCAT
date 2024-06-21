import numpy as np
import pickle
import foscat.scat_cov as sc
  

class GCNN:
        
    def __init__(self,
                 scat_operator=None,
                 nparam=1,
                 nscale=1,
                 chanlist=[],
                 in_nside=1,
                 n_chan_out=1,
                 nbatch=1,
                 SEED=1234,
                 hidden=None,
                 filename=None):

        if filename is not None:

            outlist=pickle.load(open("%s.pkl"%(filename),"rb"))
        
            self.scat_operator=sc.funct(KERNELSZ=outlist[3],all_type=outlist[7])
            self.KERNELSZ= self.scat_operator.KERNELSZ
            self.all_type= self.scat_operator.all_type
            self.npar=outlist[2]
            self.nscale=outlist[5]
            self.chanlist=outlist[0]
            self.in_nside=outlist[4] 
            self.nbatch=outlist[1]
            self.n_chan_out=outlist[8]
            if len(outlist[9])>0:
                self.hidden=outlist[9]
            else:
                self.hidden=None
                
            self.x=self.scat_operator.backend.bk_cast(outlist[6])
        else:
            self.nscale=nscale
            self.nbatch=nbatch
            self.npar=nparam
            self.n_chan_out=n_chan_out
            self.scat_operator=scat_operator
        
            if len(chanlist)!=nscale+1:
                print('len of chanlist (here %d) should of nscale+1 (here %d)'%(len(chanlist),nscale+1))
                return None
            
            self.chanlist=chanlist
            self.KERNELSZ= scat_operator.KERNELSZ
            self.all_type= scat_operator.all_type
            self.in_nside=in_nside
            self.hidden=hidden

            np.random.seed(SEED)
            self.x=scat_operator.backend.bk_cast(np.random.randn(self.get_number_of_weights())/(self.KERNELSZ*self.KERNELSZ))

    def save(self,filename):

        if self.hidden is None:
            tabh=[]
        else:
            tabh=self.hidden

        www= self.get_weights()
        
        if not isinstance(www,np.ndarray):
            www=www.numpy()
            
        outlist=[self.chanlist, \
                 self.nbatch, \
                 self.npar, \
                 self.KERNELSZ, \
                 self.in_nside, \
                 self.nscale, \
                 www, \
                 self.all_type, \
                 self.n_chan_out, \
                 tabh]
        
        myout=open("%s.pkl"%(filename),"wb")
        pickle.dump(outlist,myout)
        myout.close()
    
    def get_number_of_weights(self):
        totnchan=0
        szk=self.KERNELSZ*self.KERNELSZ
        if self.hidden is not None:
            totnchan=totnchan+self.hidden[0]*self.npar
            for i in range(1,len(self.hidden)):
                totnchan=totnchan+self.hidden[i]*self.hidden[i-1]
            totnchan=totnchan+self.hidden[len(self.hidden)-1]*12*self.in_nside**2*self.chanlist[0]
        else:
            totnchan=self.npar*12*self.in_nside**2*self.chanlist[0]
            
        for i in range(self.nscale):
            totnchan=totnchan+self.chanlist[i]*self.chanlist[i+1]*szk
            
        return totnchan+self.chanlist[i+1]*self.n_chan_out*szk

    def set_weights(self,x):
        self.x=x
        
    def get_weights(self):
        return self.x
        
    def eval(self,param,indices=None,weights=None,axis=0):

        x=self.x
        

        if axis==0:
            nval=1
        else:
            nval=param.shape[0]
        
        nn=0
        im=self.scat_operator.backend.bk_reshape(param,[nval,self.npar])
        if self.hidden is not None:
            ww=self.scat_operator.backend.bk_reshape(x[nn:nn+self.npar*self.hidden[0]], \
                                                     [self.npar,self.hidden[0]])
            im=self.scat_operator.backend.bk_matmul(im,ww)
            im=self.scat_operator.backend.bk_relu(im)
            nn+=self.npar*self.hidden[0]
            
            for i in range(1,len(self.hidden)):
                ww=self.scat_operator.backend.bk_reshape(x[nn:nn+self.hidden[i]*self.hidden[i-1]], \
                                                     [self.hidden[i-1],self.hidden[i]])
                im=self.scat_operator.backend.bk_matmul(im,ww)
                im=self.scat_operator.backend.bk_relu(im)
                nn+=self.hidden[i]*self.hidden[i-1]
            
            ww=self.scat_operator.backend.bk_reshape(x[nn:nn+self.hidden[len(self.hidden)-1]*12*self.in_nside**2*self.chanlist[0]], \
                                                     [self.hidden[len(self.hidden)-1],
                                                      12*self.in_nside**2*self.chanlist[0]])
            im=self.scat_operator.backend.bk_matmul(im,ww)
            im=self.scat_operator.backend.bk_reshape(im,[nval,12*self.in_nside**2,self.chanlist[0]])
            im=self.scat_operator.backend.bk_relu(im)
            nn+=self.hidden[len(self.hidden)-1]*12*self.in_nside**2*self.chanlist[0]
            
        else:
            ww=self.scat_operator.backend.bk_reshape(x[0:self.npar*12*self.in_nside**2*self.chanlist[0]], \
                                                     [self.npar,12*self.in_nside**2*self.chanlist[0]])
            im=self.scat_operator.backend.bk_matmul(im,ww)
            im=self.scat_operator.backend.bk_reshape(im,[nval,12*self.in_nside**2,self.chanlist[0]])
            im=self.scat_operator.backend.bk_relu(im)

            nn=self.npar*12*self.chanlist[0]*self.in_nside**2

        
        for k in range(self.nscale):
            ww=self.scat_operator.backend.bk_reshape(x[nn:nn+self.KERNELSZ*self.KERNELSZ*self.chanlist[k]*self.chanlist[k+1]],
                                                [self.KERNELSZ*self.KERNELSZ,self.chanlist[k],self.chanlist[k+1]])
            nn=nn+self.KERNELSZ*self.KERNELSZ*self.chanlist[k]*self.chanlist[k+1]
            if indices is None:
                im=self.scat_operator.healpix_layer_transpose(im,ww,axis=1)
            else:
                im=self.scat_operator.healpix_layer_transpose(im,ww,indices=indices[k],weights=weights[k],axis=1)
            im=self.scat_operator.backend.bk_relu(im)

        ww=self.scat_operator.backend.bk_reshape(x[nn:],[self.KERNELSZ*self.KERNELSZ,self.chanlist[self.nscale],self.n_chan_out])
        if indices is None:
            im=self.scat_operator.healpix_layer(im,ww,axis=1)
        else:
            im=self.scat_operator.healpix_layer(im,ww,indices=indices[self.nscale],weights=weights[self.nscale],axis=1)
            
        if axis==0:
            im=self.scat_operator.backend.bk_reshape(im,[im.shape[1],im.shape[2]])
        return im

        
