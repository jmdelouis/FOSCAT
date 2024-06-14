import foscat.FoCUS as FOC
import numpy as np
import pickle
import foscat.backend as bk
import sys

# Vérifier si TensorFlow est importé et défini
tf_defined = 'tensorflow' in sys.modules

if tf_defined:
    import tensorflow as tf
    tf_function = tf.function  # Facultatif : si vous voulez utiliser TensorFlow dans ce script
else:
    def tf_function(func):
        return func
    
def read(filename):
    thescat=scat1D(1,1,1,1,1,[0],[0])
    return thescat.read(filename)
    
class scat1D:
    def __init__(self,p00,s0,s1,s2,s2l,j1,j2,cross=False,backend=None):
        self.P00=p00
        self.S0=s0
        self.S1=s1
        self.S2=s2
        self.S2L=s2l
        self.j1=j1
        self.j2=j2
        self.cross=cross
        self.backend=backend

    # ---------------------------------------------−---------
    def build_flat(self,table):
        shape=table.shape
        ndata=1
        for k in range(1,len(table.shape)):
            ndata=ndata*table.shape[k]
        return self.backend.bk_reshape(table,[table.shape[0],ndata])
        
    # ---------------------------------------------−---------
    def flatten(self,S2L=True,P00=True):
        tmp=[self.build_flat(self.S0),self.build_flat(self.S1)]
        
        if P00:
            tmp=tmp+[self.build_flat(self.P00)]

        tmp=tmp+[self.build_flat(self.S2)]
        
        if S2L:
            tmp=tmp+[self.build_flat(self.S2L)]
            
        if isinstance(self.P00,np.ndarray):
            return np.concatenate(tmp,1)
        else:
            return self.backend.bk_concat(tmp,1)
    # ---------------------------------------------−---------
    def flatten_name(self,S2L=True,P00=True):
        
        tmp=['S0']

        
        tmp=tmp+['S1_%d'%(k) for k in range(self.S1.shape[-1])]
        
        if P00:
            tmp=tmp+['P00_%d'%(k) for k in range(self.P00.shape[-1])]

        j1,j2=self.get_j_idx()
        
        tmp=tmp+['S2_%d-%d'%(j1[k],j2[k]) for k in range(self.S2.shape[-1])]
        
        if S2L:
            tmp=tmp+['S2L_%d-%d'%(j1[k],j2[k]) for k in range(self.S2.shape[-1])]
            
        return tmp
        
    def get_j_idx(self):
        return self.j1,self.j2
    
    def get_S0(self):
        return(self.S0)

    def get_S1(self):
        return(self.S1)
    
    def get_S2(self):
        return(self.S2)

    def get_S2L(self):
        return(self.S2L)

    def get_P00(self):
        return(self.P00)

    def reset_P00(self):
        self.P00=0*self.P00

    def constant(self):
        return scat1D(self.backend.constant(self.P00 ), \
                    self.backend.constant(self.S0  ), \
                    self.backend.constant(self.S1  ), \
                    self.backend.constant(self.S2  ), \
                    self.backend.constant(self.S2L ), \
                    self.j1  , \
                    self.j2  ,backend=self.backend)

    def domult(self,x,y):
        if x.dtype==y.dtype:
            return x*y
        
        if self.backend.bk_is_complex(x):
            
            return self.backend.bk_complex(self.backend.bk_real(x)*y,self.backend.bk_imag(x)*y)
        else:
            return self.backend.bk_complex(self.backend.bk_real(y)*x,self.backend.bk_imag(y)*x)
        
    def dodiv(self,x,y):
        if x.dtype==y.dtype:
            return x/y
        if self.backend.bk_is_complex(x):
            
            return self.backend.bk_complex(self.backend.bk_real(x)/y,self.backend.bk_imag(x)/y)
        else:
            return self.backend.bk_complex(x/self.backend.bk_real(y),x/self.backend.bk_imag(y))
        
    def domin(self,x,y):
        if x.dtype==y.dtype:
            return x-y
        
        if self.backend.bk_is_complex(x):
            
            return self.backend.bk_complex(self.backend.bk_real(x)-y,self.backend.bk_imag(x)-y)
        else:
            return self.backend.bk_complex(x-self.backend.bk_real(y),x-self.backend.bk_imag(y))
        
    def doadd(self,x,y):
        if x.dtype==y.dtype:
            return x+y
        
        if self.backend.bk_is_complex(x):
            
            return self.backend.bk_complex(self.backend.bk_real(x)+y,self.backend.bk_imag(x)+y)
        else:
            return self.backend.bk_complex(x+self.backend.bk_real(y),x+self.backend.bk_imag(y))
        
    def relu(self):
        
        return scat1D(self.backend.bk_relu(self.P00), \
                    self.backend.bk_relu(self.S0), \
                    self.backend.bk_relu(self.S1), \
                    self.backend.bk_relu(self.S2), \
                    self.backend.bk_relu(self.S2L), \
                    self.j1,self.j2,backend=self.backend)

    def __add__(self,other):
        assert isinstance(other, float) or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, scat1D)
        
        if isinstance(other, scat1D):
            return scat1D(self.doadd(self.P00,other.P00), \
                        self.doadd(self.S0, other.S0), \
                        self.doadd(self.S1, other.S1), \
                        self.doadd(self.S2, other.S2), \
                        self.doadd(self.S2L, other.S2L), \
                        self.j1,self.j2,backend=self.backend)
        else:
            return scat1D((self.P00+ other), \
                        (self.S0+ other), \
                        (self.S1+ other), \
                        (self.S2+ other), \
                        (self.S2L+ other), \
                        self.j1,self.j2,backend=self.backend)
            
    def toreal(self,value):
        if value is None:
            return None
        
        return self.backend.bk_real(value)

    def addcomplex(self,value,amp):
        if value is None:
            return None
        
        return self.backend.bk_complex(value,amp*value)
 
    def add_complex(self,amp):
        return scat1D(self.addcomplex(self.P00,amp), \
                    self.addcomplex(self.S0,amp), \
                    self.addcomplex(self.S1,amp), \
                    self.addcomplex(self.S2,amp), \
                    self.addcomplex(self.S2L,amp), \
                    self.j1,self.j2,backend=self.backend)
 
    def real(self):
        return scat1D(self.toreal(self.P00), \
                    self.toreal(self.S0), \
                    self.toreal(self.S1), \
                    self.toreal(self.S2), \
                    self.toreal(self.S2L), \
                    self.j1,self.j2,backend=self.backend)

    def __radd__(self,other):
        return self.__add__(other)

    def __truediv__(self,other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, scat1D)
        
        if isinstance(other, scat1D):
            return scat1D(self.dodiv(self.P00, other.P00), \
                        self.dodiv(self.S0, other.S0), \
                        self.dodiv(self.S1, other.S1), \
                        self.dodiv(self.S2, other.S2), \
                        self.dodiv(self.S2L, other.S2L), \
                        self.j1,self.j2,backend=self.backend)
        else:
            return scat1D((self.P00/ other), \
                        (self.S0/ other), \
                        (self.S1/ other), \
                        (self.S2/ other), \
                        (self.S2L/ other), \
                        self.j1,self.j2,backend=self.backend)
        

    def __rtruediv__(self,other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, scat1D)
        
        if isinstance(other, scat1D):
            return scat1D(self.dodiv(other.P00, self.P00), \
                        self.dodiv(other.S0 , self.S0), \
                        self.dodiv(other.S1 , self.S1), \
                        self.dodiv(other.S2 , self.S2), \
                        self.dodiv(other.S2L , self.S2L), \
                        self.j1,self.j2,backend=self.backend)
        else:
            return scat1D((other/ self.P00), \
                        (other / self.S0), \
                        (other / self.S1), \
                        (other / self.S2), \
                        (other / self.S2L), \
                        self.j1,self.j2,backend=self.backend)
        
    def __sub__(self,other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, scat1D)
        
        if isinstance(other, scat1D):
            return scat1D(self.domin(self.P00, other.P00), \
                        self.domin(self.S0, other.S0), \
                        self.domin(self.S1, other.S1), \
                        self.domin(self.S2, other.S2), \
                        self.domin(self.S2L, other.S2L), \
                        self.j1,self.j2,backend=self.backend)
        else:
            return scat1D((self.P00- other), \
                        (self.S0- other), \
                        (self.S1- other), \
                        (self.S2- other), \
                        (self.S2L- other), \
                        self.j1,self.j2,backend=self.backend)
        
    def __rsub__(self,other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, scat1D)
        
        if isinstance(other, scat1D):
            return scat1D(self.domin(other.P00,self.P00), \
                        self.domin(other.S0, self.S0), \
                        self.domin(other.S1, self.S1), \
                        self.domin(other.S2, self.S2), \
                        self.domin(other.S2L, self.S2L), \
                        self.j1,self.j2,backend=self.backend)
        else:
            return scat1D((other-self.P00), \
                        (other-self.S0), \
                        (other-self.S1), \
                        (other-self.S2), \
                        (other-self.S2L), \
                        self.j1,self.j2,backend=self.backend)
        
    def __mul__(self,other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, scat1D)
        
        if isinstance(other, scat1D):
            return scat1D(self.domult(self.P00, other.P00), \
                        self.domult(self.S0, other.S0), \
                        self.domult(self.S1, other.S1), \
                        self.domult(self.S2, other.S2), \
                        self.domult(self.S2L, other.S2L), \
                        self.j1,self.j2,backend=self.backend)
        else:
            return scat1D((self.P00* other), \
                        (self.S0* other), \
                        (self.S1* other), \
                        (self.S2* other), \
                        (self.S2L* other), \
                        self.j1,self.j2,backend=self.backend)
    def relu(self):
        return scat1D(self.backend.bk_relu(self.P00),
                    self.backend.bk_relu(self.S0),
                    self.backend.bk_relu(self.S1),
                    self.backend.bk_relu(self.S2),
                    self.backend.bk_relu(self.S2L), \
                    self.j1,self.j2,backend=self.backend)


    def __rmul__(self,other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, scat1D)
        
        if isinstance(other, scat1D):
            return scat1D(self.domult(self.P00, other.P00), \
                        self.domult(self.S0, other.S0), \
                        self.domult(self.S1, other.S1), \
                        self.domult(self.S2, other.S2), \
                        self.domult(self.S2L, other.S2L), \
                        self.j1,self.j2,backend=self.backend)
        else:
            return scat1D((self.P00* other), \
                        (self.S0* other), \
                        (self.S1* other), \
                        (self.S2* other), \
                        (self.S2L* other), \
                        self.j1,self.j2,backend=self.backend)

    def l1_abs(self,x):
        y=self.get_np(x)
        if self.backend.bk_is_complex(y):
            tmp=y.real*y.real+y.imag*y.imag
            tmp=np.sign(tmp)*np.sqrt(np.fabs(tmp))
            y=tmp
        
        return(y)
    
    def plot(self,name=None,hold=True,color='blue',lw=1,legend=True):

        import matplotlib.pyplot as plt

        j1,j2=self.get_j_idx()
        
        if name is None:
            name=''

        if hold:
            plt.figure(figsize=(16,8))
        
        test=None
        plt.subplot(2, 2, 1)
        tmp=abs(self.get_np(self.S1))
        if len(tmp.shape)==4:
            for i1 in range(tmp.shape[0]):
                for i2 in range(tmp.shape[1]):
                    if test is None:
                        test=1
                        plt.plot(tmp[i1,i2,:],color=color, label=r'%s $S_{1}$' % (name), lw=lw)
                    else:
                        plt.plot(tmp[i1,i2,:],color=color, lw=lw)
        else:
            for i1 in range(tmp.shape[0]):
                if test is None:
                    test=1
                    plt.plot(tmp[i1,:],color=color, label=r'%s $S_{1}$' % (name), lw=lw)
                else:
                    plt.plot(tmp[i1,:],color=color, lw=lw)
        plt.yscale('log')
        plt.ylabel('S1')
        plt.xlabel(r'$j_{1}$')
        plt.legend()

        test=None
        plt.subplot(2, 2, 2)
        tmp=abs(self.get_np(self.P00))
        if len(tmp.shape)==4:
            for i1 in range(tmp.shape[0]):
                for i2 in range(tmp.shape[0]):
                    if test is None:
                        test=1
                        plt.plot(tmp[i1,i2,:],color=color, label=r'%s $P_{00}$' % (name), lw=lw)
                    else:
                        plt.plot(tmp[i1,i2,:],color=color, lw=lw)
        else:
            for i1 in range(tmp.shape[0]):
                if test is None:
                    test=1
                    plt.plot(tmp[i1,:],color=color, label=r'%s $P_{00}$' % (name), lw=lw)
                else:
                    plt.plot(tmp[i1,:],color=color, lw=lw)
        plt.yscale('log')
        plt.ylabel('P00')
        plt.xlabel(r'$j_{1}$')
        plt.legend()
        
        ax1=plt.subplot(2, 2, 3)
        ax2 = ax1.twiny()
        n=0
        tmp=abs(self.get_np(self.S2))
        lname=r'%s $S_{2}$' % (name)
        ax1.set_ylabel(r'$S_{2}$ [L1 norm]')
        test=None
        tabx=[]
        tabnx=[]
        tab2x=[]
        tab2nx=[]
        if len(tmp.shape)==5:
            for i0 in range(tmp.shape[0]):
                for i1 in range(tmp.shape[1]):
                    for i2 in range(j1.max()+1):
                        if j2[j1==i2].shape[0]==1:
                            ax1.plot(j2[j1==i2]+n,tmp[i0,i1,j1==i2],'.', \
                                     color=color, lw=lw)
                        else:
                            if legend and test is None:
                                ax1.plot(j2[j1==i2]+n,tmp[i0,i1,j1==i2], \
                                         color=color, label=lname, lw=lw)
                                test=1
                            ax1.plot(j2[j1==i2]+n,tmp[i0,i1,j1==i2], \
                                     color=color, lw=lw)
                    tabnx=tabnx+[r'%d'%(k) for k in j2[j1==i2]]
                    tabx=tabx+[k+n for k in j2[j1==i2]]
                    tab2x=tab2x+[(j2[j1==i2]+n).mean()]
                    tab2nx=tab2nx+['%d'%(i2)]
                    ax1.axvline((j2[j1==i2]+n).max()+0.5,ls=':',color='gray') 
                    n=n+j2[j1==i2].shape[0]-1
        else:
            for i0 in range(tmp.shape[0]):
                for i2 in range(j1.max()+1):
                    if j2[j1==i2].shape[0]==1:
                        ax1.plot(j2[j1==i2]+n,tmp[i0,j1==i2],'.', \
                                 color=color, lw=lw)
                    else:
                        if legend and test is None:
                            ax1.plot(j2[j1==i2]+n,tmp[i0,j1==i2], \
                                     color=color, label=lname, lw=lw)
                            test=1
                        ax1.plot(j2[j1==i2]+n,tmp[i0,j1==i2], \
                                         color=color, lw=lw)
                    tabnx=tabnx+[r'%d'%(k) for k in j2[j1==i2]]
                    tabx=tabx+[k+n for k in j2[j1==i2]]
                    tab2x=tab2x+[(j2[j1==i2]+n).mean()]
                    tab2nx=tab2nx+['%d'%(i2)]
                    ax1.axvline((j2[j1==i2]+n).max()+0.5,ls=':',color='gray') 
                    n=n+j2[j1==i2].shape[0]-1
        plt.yscale('log')
        ax1.set_xlim(0,n+2)
        ax1.set_xticks(tabx)
        ax1.set_xticklabels(tabnx,fontsize=6)
        ax1.set_xlabel(r"$j_{2}$ ",fontsize=6)
        
        # Move twinned axis ticks and label from top to bottom
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        # Offset the twin axis below the host
        ax2.spines["bottom"].set_position(("axes", -0.15))

        # Turn on the frame for the twin axis, but then hide all 
        # but the bottom spine
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)

        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.spines["bottom"].set_visible(True)
        ax2.set_xlim(0,n+2)
        ax2.set_xticks(tab2x)
        ax2.set_xticklabels(tab2nx,fontsize=6)
        ax2.set_xlabel(r"$j_{1}$",fontsize=6)
        ax1.legend(frameon=0)
        
        ax1=plt.subplot(2, 2, 4)
        ax2 = ax1.twiny()
        n=0
        tmp=abs(self.get_np(self.S2L))
        lname=r'%s $S2_{2}$' % (name)
        ax1.set_ylabel(r'$S_{2}$ [L2 norm]')
        test=None
        tabx=[]
        tabnx=[]
        tab2x=[]
        tab2nx=[]
        if len(tmp.shape)==5:
            for i0 in range(tmp.shape[0]):
                for i1 in range(tmp.shape[1]):
                    for i2 in range(j1.max()+1):
                        if j2[j1==i2].shape[0]==1:
                            ax1.plot(j2[j1==i2]+n,tmp[i0,i1,j1==i2],'.', \
                                     color=color, lw=lw)
                        else:
                            if legend and test is None:
                                ax1.plot(j2[j1==i2]+n,tmp[i0,i1,j1==i2], \
                                         color=color, label=lname, lw=lw)
                                test=1
                            ax1.plot(j2[j1==i2]+n,tmp[i0,i1,j1==i2], \
                                     color=color, lw=lw)
                        tabnx=tabnx+[r'%d'%(k) for k in j2[j1==i2]]
                        tabx=tabx+[k+n for k in j2[j1==i2]]
                        tab2x=tab2x+[(j2[j1==i2]+n).mean()]
                        tab2nx=tab2nx+['%d'%(i2)]
                        ax1.axvline((j2[j1==i2]+n).max()+0.5,ls=':',color='gray') 
                        n=n+j2[j1==i2].shape[0]-1
        else:
            for i0 in range(tmp.shape[0]):
                for i2 in range(j1.max()+1):
                    if j2[j1==i2].shape[0]==1:
                        ax1.plot(j2[j1==i2]+n,tmp[i0,j1==i2],'.', \
                                 color=color, lw=lw)
                    else:
                        if legend and test is None:
                            ax1.plot(j2[j1==i2]+n,tmp[i0,j1==i2], \
                                     color=color, label=lname, lw=lw)
                            test=1
                        ax1.plot(j2[j1==i2]+n,tmp[i0,j1==i2], \
                                 color=color, lw=lw)
                    tabnx=tabnx+[r'%d'%(k) for k in j2[j1==i2]]
                    tabx=tabx+[k+n for k in j2[j1==i2]]
                    tab2x=tab2x+[(j2[j1==i2]+n).mean()]
                    tab2nx=tab2nx+['%d'%(i2)]
                    ax1.axvline((j2[j1==i2]+n).max()+0.5,ls=':',color='gray') 
                    n=n+j2[j1==i2].shape[0]-1
        plt.yscale('log')
        ax1.set_xlim(-1,n+3)
        ax1.set_xticks(tabx)
        ax1.set_xticklabels(tabnx,fontsize=6)
        ax1.set_xlabel(r"$j_{2}$",fontsize=6)
        
        # Move twinned axis ticks and label from top to bottom
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        # Offset the twin axis below the host
        ax2.spines["bottom"].set_position(("axes", -0.15))

        # Turn on the frame for the twin axis, but then hide all 
        # but the bottom spine
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)

        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.spines["bottom"].set_visible(True)
        ax2.set_xlim(0,n+3)
        ax2.set_xticks(tab2x)
        ax2.set_xticklabels(tab2nx,fontsize=6)
        ax2.set_xlabel(r"$j_{1}$",fontsize=6)
        ax1.legend(frameon=0)
        
    def save(self,filename):
        outlist=[self.get_S0().numpy(), \
                 self.get_S1().numpy(), \
                 self.get_S2().numpy(), \
                 self.get_S2L().numpy(), \
                 self.get_P00().numpy(), \
                 self.j1, \
                 self.j2]

        myout=open("%s.pkl"%(filename),"wb")
        pickle.dump(outlist,myout)
        myout.close()

        
    def read(self,filename):
        
        outlist=pickle.load(open("%s.pkl"%(filename),"rb"))

        return scat1D(outlist[4],outlist[0],outlist[1],outlist[2],outlist[3],outlist[5],outlist[6],backend=bk.foscat_backend('numpy'))
    
    def get_np(self,x):
        if isinstance(x, np.ndarray):
            return x
        else:
            return x.numpy()

    def std(self):
        return np.sqrt(((abs(self.get_np(self.S0)).std())**2+ \
                        (abs(self.get_np(self.S1)).std())**2+ \
                        (abs(self.get_np(self.S2)).std())**2+ \
                        (abs(self.get_np(self.S2L)).std())**2+ \
                        (abs(self.get_np(self.P00)).std())**2)/4)

    def mean(self):
        return abs(self.get_np(self.S0).mean()+ \
                   self.get_np(self.S1).mean()+ \
                   self.get_np(self.S2).mean()+ \
                   self.get_np(self.S2L).mean()+ \
                   self.get_np(self.P00).mean())/3



    # ---------------------------------------------−---------
    def cleanval(self,x):
        x=x.numpy()
        x[np.isfinite(x)==False]=np.median(x[np.isfinite(x)])
        return x

    def filter_inf(self):
        S1  = self.cleanval(self.S1)
        S0  = self.cleanval(self.S0)
        P00 = self.cleanval(self.P00)
        S2  = self.cleanval(self.S2)
        S2L = self.cleanval(self.S2L)

        return scat1D(P00,S0,S1,S2,S2L,self.j1,self.j2,backend=self.backend)

    # ---------------------------------------------−---------
    def interp(self,nscale,extend=False,constant=False,threshold=1E30):
        
        if nscale+2>self.S1.shape[1]:
            print('Can not *interp* %d with a statistic described over %d'%(nscale,self.S1.shape[1]))
            return scat1D(self.P00,self.S0,self.S1,self.S2,self.S2L,self.j1,self.j2,backend=self.backend)
        if isinstance(self.S1,np.ndarray):
            s1=self.S1
            p0=self.P00
            s2=self.S2
            s2l=self.S2L
        else:
            s1=self.S1.numpy()
            p0=self.P00.numpy()
            s2=self.S2.numpy()
            s2l=self.S2L.numpy()

        if isinstance(threshold,scat1D):
            if isinstance(threshold.S1,np.ndarray):
                s1th=threshold.S1
                p0th=threshold.P00
                s2th=threshold.S2
                s2lth=threshold.S2L
            else:
                s1th=threshold.S1.numpy()
                p0th=threshold.P00.numpy()
                s2th=threshold.S2.numpy()
                s2lth=threshold.S2L.numpy()
        else:
            s1th=threshold+0*s1
            p0th=threshold+0*p0
            s2th=threshold+0*s2
            s2lth=threshold+0*s2l

        for k in range(nscale):
            if constant:
                s1[:,nscale-1-k,:]=s1[:,nscale-k,:]
                p0[:,nscale-1-k,:]=p0[:,nscale-k,:]
            else:
                idx=np.where((s1[:,nscale+1-k,:]>0)*(s1[:,nscale+2-k,:]>0)*(s1[:,nscale-k,:]<s1th[:,nscale-k,:]))
                if len(idx[0])>0:
                    s1[idx[0],nscale-1-k,idx[1]]=np.exp(3*np.log(s1[idx[0],nscale+1-k,idx[1]])-2*np.log(s1[idx[0],nscale+2-k,idx[1]]))
                idx=np.where((s1[:,nscale-k,:]>0)*(s1[:,nscale+1-k,:]>0)*(s1[:,nscale-1-k,:]<s1th[:,nscale-1-k,:]))
                if len(idx[0])>0:
                    s1[idx[0],nscale-1-k,idx[1]]=np.exp(2*np.log(s1[idx[0],nscale-k,idx[1]])-np.log(s1[idx[0],nscale+1-k,idx[1]]))

                idx=np.where((p0[:,nscale+1-k,:]>0)*(p0[:,nscale+2-k,:]>0)*(p0[:,nscale-k,:]<p0th[:,nscale-k,:]))
                if len(idx[0])>0:
                    p0[idx[0],nscale-1-k,idx[1]]=np.exp(3*np.log(p0[idx[0],nscale+1-k,idx[1]])-2*np.log(p0[idx[0],nscale+2-k,idx[1]]))

                idx=np.where((p0[:,nscale-k,:]>0)*(p0[:,nscale+1-k,:]>0)*(p0[:,nscale-1-k,:]<p0th[:,nscale-1-k,:]))
                if len(idx[0])>0:
                    p0[idx[0],nscale-1-k,idx[1]]=np.exp(2*np.log(p0[idx[0],nscale-k,idx[1]])-np.log(p0[idx[0],nscale+1-k,idx[1]]))
                
        
        j1,j2=self.get_j_idx()

        for k in range(nscale):

            """
            i0=np.where((j1==nscale-1-k)*(j2>=nscale+1-k))[0]
            i1=np.where((j1==nscale-k)*(j2>=nscale+1-k))[0]
            i2=np.where((j1==nscale+1-k)*(j2>=nscale+1-k))[0]
            print(i0,j1[i0],j2[i0],j1[i1],j2[i1],j1[i2],j2[i2])
            s2[:,i0]=np.exp(2*np.log(s2[:,i1])-np.log(s2[:,i2]))
            s2l[:,i0]=np.exp(2*np.log(s2l[:,i1])-np.log(s2l[:,i2]))
            """

            for l in range(nscale-k):
                i0=np.where((j1==nscale-1-k-l)*(j2==nscale-1-k))[0]
                i1=np.where((j1==nscale-1-k-l)*(j2==nscale  -k))[0]
                i2=np.where((j1==nscale-1-k-l)*(j2==nscale+1-k))[0]
                i3=np.where((j1==nscale-1-k-l)*(j2==nscale+2-k))[0]
                
                if constant:
                    s2[:,i0]=s2[:,i1]
                    s2l[:,i0]=s2l[:,i1]
                else:
                    idx=np.where((s2[:,i2]>0)*(s2[:,i3]>0)*(s2[:,i2]<s2th[:,i2]))
                    if len(idx[0])>0:
                        s2[idx[0],i0,idx[1],idx[2]]=np.exp(3*np.log(s2[idx[0],i2,idx[1],idx[2]])-2*np.log(s2[idx[0],i3,idx[1],idx[2]]))

                    idx=np.where((s2[:,i1]>0)*(s2[:,i2]>0)*(s2[:,i1]<s2th[:,i1]))
                    if len(idx[0])>0:
                        s2[idx[0],i0,idx[1],idx[2]]=np.exp(2*np.log(s2[idx[0],i1,idx[1],idx[2]])-np.log(s2[idx[0],i2,idx[1],idx[2]]))
                    
                    idx=np.where((s2l[:,i2]>0)*(s2l[:,i3]>0)*(s2l[:,i2]<s2lth[:,i2]))
                    if len(idx[0])>0:
                        s2l[idx[0],i0,idx[1],idx[2]]=np.exp(3*np.log(s2l[idx[0],i2,idx[1],idx[2]])-2*np.log(s2l[idx[0],i3,idx[1],idx[2]]))

                    idx=np.where((s2l[:,i1]>0)*(s2l[:,i2]>0)*(s2l[:,i1]<s2lth[:,i1]))
                    if len(idx[0])>0:
                        s2l[idx[0],i0,idx[1],idx[2]]=np.exp(2*np.log(s2l[idx[0],i1,idx[1],idx[2]])-np.log(s2l[idx[0],i2,idx[1],idx[2]]))
                    
        if extend:
            for k in range(nscale):
                for l in range(1,nscale):
                    i0=np.where((j1==2*nscale-1-k)*(j2==2*nscale-1-k-l))[0]
                    i1=np.where((j1==2*nscale-1-k)*(j2==2*nscale  -k-l))[0]
                    i2=np.where((j1==2*nscale-1-k)*(j2==2*nscale+1-k-l))[0]
                    i3=np.where((j1==2*nscale-1-k)*(j2==2*nscale+2-k-l))[0]
                    if constant:
                        s2[:,i0]=s2[:,i1]
                        s2l[:,i0]=s2l[:,i1]
                    else:
                        idx=np.where((s2[:,i2]>0)*(s2[:,i3]>0)*(s2[:,i2]<s2th[:,i2]))
                        if len(idx[0])>0:
                            s2[idx[0],i0,idx[1],idx[2]]=np.exp(3*np.log(s2[idx[0],i2,idx[1],idx[2]])-2*np.log(s2[idx[0],i3,idx[1],idx[2]]))
                        idx=np.where((s2[:,i1]>0)*(s2[:,i2]>0)*(s2[:,i1]<s2th[:,i1]))
                        if len(idx[0])>0:
                            s2[idx[0],i0,idx[1],idx[2]]=np.exp(2*np.log(s2[idx[0],i1,idx[1],idx[2]])-np.log(s2[idx[0],i2,idx[1],idx[2]]))

                        idx=np.where((s2l[:,i2]>0)*(s2l[:,i3]>0)*(s2l[:,i2]<s2lth[:,i2]))
                        if len(idx[0])>0:
                            s2l[idx[0],i0,idx[1],idx[2]]=np.exp(3*np.log(s2l[idx[0],i2,idx[1],idx[2]])-2*np.log(s2l[idx[0],i3,idx[1],idx[2]]))
                        idx=np.where((s2l[:,i1]>0)*(s2l[:,i2]>0)*(s2l[:,i1]<s2lth[:,i1]))
                        if len(idx[0])>0:
                            s2l[idx[0],i0,idx[1],idx[2]]=np.exp(2*np.log(s2l[idx[0],i1,idx[1],idx[2]])-np.log(s2l[idx[0],i2,idx[1],idx[2]]))
        
        s1[np.isnan(s1)]=0.0
        p0[np.isnan(p0)]=0.0
        s2[np.isnan(s2)]=0.0
        s2l[np.isnan(s2l)]=0.0

        return scat1D(self.backend.constant(p0),self.S0,
                    self.backend.constant(s1),
                    self.backend.constant(s2),
                    self.backend.constant(s2l),self.j1,self.j2,backend=self.backend)

    # ---------------------------------------------−---------
    def model(self,i__y,add=0,dx=3,dell=2,weigth=None,inverse=False):

        if i__y.shape[0]<dx+1:
            l__dx=i__y.shape[0]-1
        else:
            l__dx=dx

        if i__y.shape[0]<dell:
            l__dell=0
        else:
            l__dell=dell

        if l__dx<2:
            res=np.zeros([i__y.shape[0]+add])
            if inverse:
                res[:-add]=i__y
            else:
                res[add:]=i__y[0:]
            return res

        if weigth is None:
            w=2**(np.arange(l__dx))
        else:
            if not inverse:
                w=weigth[0:l__dx]
            else:
                w=weigth[-l__dx:]

        x=np.arange(l__dx)+1
        if not inverse:
            y=np.log(i__y[1:l__dx+1])
        else:
            y=np.log(i__y[-(l__dx+1):-1])

        r=np.polyfit(x,y,1,w=w)

        if inverse:
            res=np.exp(r[0]*(np.arange(i__y.shape[0]+add)-1)+r[1])
            res[:-(l__dell+add)]=i__y[:-l__dell]
        else:
            res=np.exp(r[0]*(np.arange(i__y.shape[0]+add)-add)+r[1])
            res[l__dell+add:]=i__y[l__dell:]
        return res

    def findn(self,n):
        d=np.sqrt(1+8*n)
        return int((d-1)/2)

    def findidx(self,s2):
        i1=np.zeros([s2.shape[1]],dtype='int')
        i2=np.zeros([s2.shape[1]],dtype='int')
        n=0
        for k in range(self.findn(s2.shape[1])):
            i1[n:n+k+1]=np.arange(k+1)
            i2[n:n+k+1]=k
            n=n+k+1
        return i1,i2

    def extrapol_s2(self,add,lnorm=1):
        if lnorm==1:
            s2=self.S2.numpy()
        if lnorm==2:
            s2=self.S2L.numpy()
        i1,i2=self.findidx(s2)

        so2=np.zeros([s2.shape[0],(self.findn(s2.shape[1])+add)*(self.findn(s2.shape[1])+add+1)//2,s2.shape[2],s2.shape[3]])
        oi1,oi2=self.findidx(so2)
        for l in range(s2.shape[0]):
            for k in range(self.findn(s2.shape[1])):
                for i in range(s2.shape[2]):
                    for j in range(s2.shape[3]):
                        tmp=self.model(s2[l,i2==k,i,j],dx=4,dell=1,add=add,weigth=np.array([1,2,2,2]))
                        tmp[np.isnan(tmp)]=0.0
                        so2[l,oi2==k+add,i,j]=tmp


        for l in range(s2.shape[0]):
            for k in range(add+1,-1,-1):
                lidx=np.where(oi2-oi1==k)[0]
                lidx2=np.where(oi2-oi1==k+1)[0]
                for i in range(s2.shape[2]):
                    for j in range(s2.shape[3]):
                        so2[l,lidx[0:add+2-k],i,j]=so2[l,lidx2[0:add+2-k],i,j]

        return(so2)

    def extrapol_s1(self,i_s1,add):
        s1=i_s1.numpy()
        so1=np.zeros([s1.shape[0],s1.shape[1]+add,s1.shape[2]])
        for k in range(s1.shape[0]):
            for i in range(s1.shape[2]):
                so1[k,:,i]=self.model(s1[k,:,i],dx=4,dell=1,add=add)
                so1[k,np.isnan(so1[k,:,i]),i]=0.0
        return so1

    def extrapol(self,add):
        return scat1D(self.extrapol_s1(self.P00,add), \
                    self.S0, \
                    self.extrapol_s1(self.S1,add), \
                    self.extrapol_s2(add,lnorm=1), \
                    self.extrapol_s2(add,lnorm=2),self.j1,self.j2,backend=self.backend)
        
        
        
        
    
class funct(FOC.FoCUS):

    def fill(self,im,nullval=0):
        return self.fill_1d(im,nullval=nullval)

    def ud_grade(self,im,nout,axis=0):
        return self.ud_grade_1d(im,nout,axis=axis)
    
    def up_grade(self,im,nout,axis=0):
        return self.up_grade_1d(im,nout,axis=axis)
    
    def smooth(self,data,axis=0):
        return self.smooth_1d(data,axis=axis)

    def convol(self,data,axis=0):
        return self.convol_1d(data,axis=axis)
    
    def eval(self, image1, image2=None,mask=None,Auto=True,s0_off=1E-6,Add_R45=False,axis=0):
        # Check input consistency
        if mask is not None:
            if len(image1.shape)==1:
                if image1.shape[0]!=mask.shape[1]:
                    print('The mask should have the same size than the input timeline to eval Scattering')
                    return None
            else:
                if image1.shape[1]!=mask.shape[1]:
                    print('The mask should have the same size than the input timeline to eval Scattering')
                    return None
            
        ### AUTO OR CROSS
        cross = False
        if image2 is not None:
            cross = True
            all_cross=not Auto
        else:
            all_cross=False
        
        # determine jmax and nside corresponding to the input map
        im_shape = image1.shape

        nside=im_shape[len(image1.shape)-1]
        npix=nside
        
        jmax=int(np.log(nside)/np.log(2)) #-self.OSTEP

        ### LOCAL VARIABLES (IMAGES and MASK)
        # Check if image1 is [Npix] or [Nbatch,Npix]
        if len(image1.shape)==1:
            # image1 is [Nbatch, Npix]
            I1 = self.backend.bk_cast(self.backend.bk_expand_dims(image1,0))  # Local image1 [Nbatch, Npix]
            if cross:
                I2 = self.backend.bk_cast(self.backend.bk_expand_dims(image2,0))  # Local image2 [Nbatch, Npix]
        else:
            I1=self.backend.bk_cast(image1)
            if cross:
                I2=self.backend.bk_cast(image2)
        
        # self.mask is [Nmask, Npix]
        
        if mask is None:
            vmask = self.backend.bk_ones([1, npix], dtype=self.all_type)
        else:
            vmask = self.backend.bk_cast(mask)  # [Nmask, Npix]

        if self.KERNELSZ>3:
            # if the kernel size is bigger than 3 increase the binning before smoothing
            l_image1=self.up_grade_1d(I1,nside*2,axis=axis+1)
            vmask=self.up_grade_1d(vmask,nside*2,axis=1)
                
            if cross:
                l_image2=self.up_grade_1d(I2,nside*2,axis=axis+1)
        else:
            l_image1=I1
            if cross:
                l_image2=I2
        if len(image1.shape)==1:
            s0 = self.backend.bk_reduce_sum(l_image1*vmask,axis=axis+1)
            if cross and Auto==False:
                s0 = self.backend.bk_concat([s0,self.backend.bk_reduce_sum(l_image2*vmask,axis=axis)])
        else:
            lmask=self.backend.bk_expand_dims(vmask,0)
            lim=self.backend.bk_expand_dims(l_image1,1)
            s0 = self.backend.bk_reduce_sum(lim*lmask,axis=axis+2)
            if cross and Auto==False:
                lim=self.backend.bk_expand_dims(l_image2,1)
                s0 = self.backend.bk_concat([s0,self.backend.bk_reduce_sum(lim*lmask,axis=axis+2)])
            
        

        s1=None
        s2=None
        s2l=None
        p00=None
        s2j1=None
        s2j2=None

        l2_image=None
        l2_image_imag=None

        for j1 in range(jmax):
            if j1<jmax-self.OSTEP: # stop to add scales 
                # Convol image along the axis defined by 'axis' using the wavelet defined at
                # the foscat initialisation
                #c_image_real is [....,Npix_j1,....,Norient]
                c_image1=self.convol_1d(l_image1,axis=axis+1)
                if cross:
                    c_image2=self.convol_1d(l_image2,axis=axis+1)
                else:
                    c_image2=c_image1

                # Compute (a+ib)*(a+ib)* the last c_image column is the real and imaginary part
                conj=c_image1*self.backend.bk_conjugate(c_image2)
            
                if Auto:
                    conj=self.backend.bk_real(conj)

                # Compute l_p00 [....,....,Nmask,1]
                l_p00 = self.backend.bk_expand_dims(self.backend.bk_reduce_sum(conj*vmask,axis=1),-1)

                conj=self.backend.bk_L1(conj)

                # Compute l_s1 [....,....,Nmask,1]
                l_s1 = self.backend.bk_expand_dims(self.backend.bk_reduce_sum(conj*vmask,axis=1),-1)
                
                # Concat S1,P00 [....,....,Nmask,j1] 
                if s1 is None:
                    s1=l_s1
                    p00=l_p00
                else:
                    s1=self.backend.bk_concat([s1,l_s1],axis=-1)
                    p00=self.backend.bk_concat([p00,l_p00],axis=-1)

                # Concat l2_image [....,Npix_j1,....,j1]
                if l2_image is None:
                    l2_image=self.backend.bk_expand_dims(conj,axis=1)
                else:
                    l2_image=self.backend.bk_concat([self.backend.bk_expand_dims(conj,axis=1),l2_image],axis=1)

            # Convol l2_image [....,Npix_j1,....,j1,Norient,Norient]
            c2_image=self.convol_1d(self.backend.bk_relu(l2_image),axis=axis+2)

            conj2p=c2_image*self.backend.bk_conjugate(c2_image)
            conj2pl1=self.backend.bk_L1(conj2p)

            if Auto:
                conj2p=self.backend.bk_real(conj2p)
                conj2pl1=self.backend.bk_real(conj2pl1)

            c2_image=self.convol_1d(self.backend.bk_relu(-l2_image),axis=axis+2)

            conj2m=c2_image*self.backend.bk_conjugate(c2_image)
            conj2ml1=self.backend.bk_L1(conj2m)

            if Auto:
                conj2m=self.backend.bk_real(conj2m)
                conj2ml1=self.backend.bk_real(conj2ml1)
                
            # Convol l_s2 [....,....,Nmask,j1]
            l_s2 = self.backend.bk_reduce_sum((conj2p-conj2m)*self.backend.bk_expand_dims(vmask,1),axis=axis+2)
            l_s2l1 = self.backend.bk_reduce_sum((conj2pl1-conj2ml1)*self.backend.bk_expand_dims(vmask,1),axis=axis+2)

            # Concat l_s2 [....,....,Nmask,j1*(j1+1)/2,Norient,Norient]
            if s2 is None:
                s2l=l_s2
                s2=l_s2l1
                s2j1=np.arange(l_s2.shape[axis],dtype='int')
                s2j2=j1*np.ones(l_s2.shape[axis],dtype='int')
            else:
                s2=self.backend.bk_concat([s2,l_s2l1],axis=-1)
                s2l=self.backend.bk_concat([s2l,l_s2],axis=-1)
                s2j1=np.concatenate([s2j1,np.arange(l_s2.shape[axis+1],dtype='int')],0)
                s2j2=np.concatenate([s2j2,j1*np.ones(l_s2.shape[axis+1],dtype='int')],0)

            if j1!=jmax-1:
                # Rescale vmask [Nmask,Npix_j1//4]   
                vmask = self.smooth_1d(vmask,axis=1)
                vmask = self.ud_grade_1d(vmask,vmask.shape[1]//2,axis=1)
                if self.mask_thres is not None:
                    vmask = self.backend.bk_threshold(vmask,self.mask_thres)

                # Rescale l2_image [....,Npix_j1//4,....,j1,Norient]   
                l2_image = self.smooth_1d(l2_image,axis=axis+2)
                l2_image = self.ud_grade_1d(l2_image,l2_image.shape[axis+2]//2,axis=axis+2)

                # Rescale l_image [....,Npix_j1//4,....]  
                l_image1 = self.smooth_1d(l_image1,axis=axis+1)
                l_image1 = self.ud_grade_1d(l_image1,l_image1.shape[axis+1]//2,axis=axis+1)
                if cross:
                    l_image2 = self.smooth_1d(l_image2,axis=axis+2)
                    l_image2 = self.ud_grade_1d(l_image2,l_image2.shape[axis+2]//2,axis=axis+2)

        return(scat1D(p00,s0,s1,s2,s2l,s2j1,s2j2,cross=cross,backend=self.backend))

    def square(self,x):
        # the abs make the complex value usable for reduce_sum or mean
        return scat1D(self.backend.bk_square(self.backend.bk_abs(x.P00)),
                    self.backend.bk_square(self.backend.bk_abs(x.S0)),
                    self.backend.bk_square(self.backend.bk_abs(x.S1)),
                    self.backend.bk_square(self.backend.bk_abs(x.S2)),
                    self.backend.bk_square(self.backend.bk_abs(x.S2L)),x.j1,x.j2,backend=self.backend)
    
    def sqrt(self,x):
        # the abs make the complex value usable for reduce_sum or mean
        return scat1D(self.backend.bk_sqrt(self.backend.bk_abs(x.P00)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.S0)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.S1)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.S2)),
                    self.backend.bk_sqrt(self.backend.bk_abs(x.S2L)),x.j1,x.j2,backend=self.backend)

    def reduce_mean(self,x,axis=None):
        if axis is None:
            tmp=self.backend.bk_abs(self.backend.bk_reduce_sum(x.P00))+ \
                 self.backend.bk_abs(self.backend.bk_reduce_sum(x.S0))+ \
                 self.backend.bk_abs(self.backend.bk_reduce_sum(x.S1))+ \
                 self.backend.bk_abs(self.backend.bk_reduce_sum(x.S2))+ \
                 self.backend.bk_abs(self.backend.bk_reduce_sum(x.S2L))
            
            ntmp=np.array(list(x.P00.shape)).prod()+ \
                  np.array(list(x.S0.shape)).prod()+ \
                  np.array(list(x.S1.shape)).prod()+ \
                  np.array(list(x.S2.shape)).prod()
            
            return  tmp/ntmp
        else:
            tmp=self.backend.bk_abs(self.backend.bk_reduce_sum(x.P00,axis=axis))+ \
                 self.backend.bk_abs(self.backend.bk_reduce_sum(x.S0,axis=axis))+ \
                 self.backend.bk_abs(self.backend.bk_reduce_sum(x.S1,axis=axis))+ \
                 self.backend.bk_abs(self.backend.bk_reduce_sum(x.S2,axis=axis))+ \
                 self.backend.bk_abs(self.backend.bk_reduce_sum(x.S2L,axis=axis))
            
            ntmp=np.array(list(x.P00.shape)).prod()+ \
                  np.array(list(x.S0.shape)).prod()+ \
                  np.array(list(x.S1.shape)).prod()+ \
                  np.array(list(x.S2.shape)).prod()+ \
                  np.array(list(x.S2L.shape)).prod()
            
            return  tmp/ntmp

    def reduce_sum(self,x,axis=None):
        if axis is None:
            return  self.backend.bk_reduce_sum(self.backend.bk_abs(x.P00))+ \
                self.backend.bk_reduce_sum(self.backend.bk_abs(x.S0))+ \
                self.backend.bk_reduce_sum(self.backend.bk_abs(x.S1))+ \
                self.backend.bk_reduce_sum(self.backend.bk_abs(x.S2))+ \
                self.backend.bk_reduce_sum(self.backend.bk_abs(x.S2L))
        else:
            return scat1D(self.backend.bk_reduce_sum(x.P00,axis=axis),
                        self.backend.bk_reduce_sum(x.S0,axis=axis),
                        self.backend.bk_reduce_sum(x.S1,axis=axis),
                        self.backend.bk_reduce_sum(x.S2,axis=axis),
                        self.backend.bk_reduce_sum(x.S2L,axis=axis),x.j1,x.j2,backend=self.backend)
        
    def ldiff(self,sig,x):
        return scat1D(x.domult(sig.P00,x.P00)*x.domult(sig.P00,x.P00),
                    x.domult(sig.S0,x.S0)*x.domult(sig.S0,x.S0),
                    x.domult(sig.S1,x.S1)*x.domult(sig.S1,x.S1),
                    x.domult(sig.S2,x.S2)*x.domult(sig.S2,x.S2),
                    x.domult(sig.S2L,x.S2L)*x.domult(sig.S2L,x.S2L),x.j1,x.j2,backend=self.backend)

    def log(self,x):
        return scat1D(self.backend.bk_log(x.P00),
                    self.backend.bk_log(x.S0),
                    self.backend.bk_log(x.S1),
                    self.backend.bk_log(x.S2),
                    self.backend.bk_log(x.S2L),x.j1,x.j2,backend=self.backend)
    def abs(self,x):
        return scat1D(self.backend.bk_abs(x.P00),
                    self.backend.bk_abs(x.S0),
                    self.backend.bk_abs(x.S1),
                    self.backend.bk_abs(x.S2),
                    self.backend.bk_abs(x.S2L),x.j1,x.j2,backend=self.backend)
    def inv(self,x):
        return scat1D(1/(x.P00),1/(x.S0),1/(x.S1),1/(x.S2),1/(x.S2L),x.j1,x.j2,backend=self.backend)

    def one(self):
        return scat1D(1.0,1.0,1.0,1.0,1.0,[0],[0],backend=self.backend)

    def eval_comp_fast(self, image1, image2=None,mask=None,Auto=True,s0_off=1E-6):

        res=self.eval_fast(image1, image2=image2,mask=mask,Auto=Auto,s0_off=s0_off)
        return res.P00,res.S0,res.S1,res.S2,res.S2L,res.j1,res.j2

    @tf_function
    def eval_fast(self, image1, image2=None,mask=None,Auto=True,s0_off=1E-6):
        p0,s0,s1,s2,s2l,j1,j2=self.eval_comp_fast(image1, image2=image2,mask=mask,Auto=Auto,s0_off=s0_off)
        return scat1D(p0,s0,s1,s2,s2l,j1,j2,backend=self.backend)
        
        
        
