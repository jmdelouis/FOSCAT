import foscat.FoCUS as FOC
import numpy as np
import foscat.backend as bk
import foscat.Rformat as Rformat


def read(filename):
    thescat = scat_cov(1, 1, 1)
    return thescat.read(filename)


testwarn=0

class scat_cov:
    def __init__(self, p00, c01, c11, s1=None, c10=None,backend=None):
        self.P00     = p00
        self.C01     = c01
        self.C11     = c11
        self.S1      = s1
        self.C10     = c10
        self.backend = backend

    def numpy(self):
        if self.S1 is None:
            s1 = None
        else:
            s1=self.S1.numpy()
        if self.C10 is None:
            c10 = None
        else:
            c10=self.C10.numpy()
        
        return scat_cov((self.P00.numpy()),
                        (self.C01.numpy()),
                        (self.C11.numpy()),
                        s1=s1, c10=c10,backend=self.backend)
        
    def get_S1(self):
        return self.S1

    def get_P00(self):
        return self.P00

    def reset_P00(self):
        self.P00=0*self.P00

    def get_C01(self):
        return self.C01

    def get_C10(self):
        return self.C10

    def get_C11(self):
        return self.C11

    def __add__(self, other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
               isinstance(other, bool) or isinstance(other, scat_cov)

        if self.S1 is None:
            s1 = None
        else:
            if isinstance(other, scat_cov):
                if other.S1 is None:
                    s1=None
                else:
                    s1 = self.S1 + other.S1
            else:
                s1 = self.S1 + other

        if self.C10 is None:
            c10 = None
        else:
            if isinstance(other, scat_cov):
                if other.C10 is None:
                    c10=None
                else:
                    c10 = self.doadd(self.C10 , other.C10)
            else:
                c10 = self.C10 + other
                
        if self.C11 is None:
            c11 = None
        else:
            if isinstance(other, scat_cov):
                if other.C11 is None:
                    c11 = None
                else:
                    c11 = self.doadd(self.C11, other.C11 )
            else:
                c11 = self.C11+other 

        if isinstance(other, scat_cov):
            return scat_cov(self.doadd(self.P00,other.P00),
                            (self.C01 + other.C01),
                            c11,s1=s1, c10=c10,backend=self.backend)
        else:
            return scat_cov((self.P00 + other),
                            (self.C01 + other),
                            c11,s1=s1, c10=c10,backend=self.backend)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __truediv__(self, other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
               isinstance(other, bool) or isinstance(other, scat_cov)

        if self.S1 is None:
            s1 = None
        else:
            if isinstance(other, scat_cov):
                if other.S1 is None:
                    s1 = None
                else:
                    s1 = self.S1 / other.S1
            else:
                s1 = self.S1 / other

        if self.C10 is None:
            c10 = None
        else:
            if isinstance(other, scat_cov):
                if other.C10 is None:
                    c10 = None
                else:
                    c10 = self.dodiv(self.C10 , other.C10)
            else:
                c10 = self.C10 / other
                
        if self.C11 is None:
            c11 = None
        else:
            if isinstance(other, scat_cov):
                if other.C11 is None:
                    c11 = None
                else:
                    c11 = self.dodiv(self.C11, other.C11 )
            else:
                c11 = self.C11/other 

        if isinstance(other, scat_cov):
            return scat_cov(self.dodiv(self.P00,other.P00),
                            (self.C01 / other.C01),
                            c11,s1=s1, c10=c10,backend=self.backend)
        else:
            return scat_cov((self.P00 / other),
                            (self.C01 / other),
                            c11,s1=s1, c10=c10,backend=self.backend)

    def __rtruediv__(self, other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
               isinstance(other, bool) or isinstance(other, scat_cov)

        if self.S1 is None:
            s1 = None
        else:
            if isinstance(other, scat_cov):
                s1 = other.S1 / self.S1
            else:
                s1 = other/self.S1

        if self.C10 is None:
            c10 = None
        else:
            if isinstance(other, scat_cov):
                c10 = self.dodiv(other.C10 , self.C10)
            else:
                c10 = other/self.C10
                
        if self.C11 is None:
            c11 = None
        else:
            if isinstance(other, scat_cov):
                if other.C11 is None:
                    c11 = None
                else:
                    c11 = self.dodiv( other.C11,self.C11 )
            else:
                c11 = other/self.C11

        if isinstance(other, scat_cov):
            return scat_cov(self.dodiv(other.P00,self.P00),
                            (other.C01 / self.C01),
                            c11,s1=s1, c10=c10,backend=self.backend)
        else:
            return scat_cov((other/self.P00 ),
                            (other/self.C01 ),
                            (other/self.C11 ),
                            s1=s1, c10=c10,backend=self.backend)

    def __rsub__(self, other):

        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
               isinstance(other, bool) or isinstance(other, scat_cov)

        if self.S1 is None:
            s1 = None
        else:
            if isinstance(other, scat_cov):
                if other.S1 is None:
                    s1 = None
                else:
                    s1 = other.S1 - self.S1
            else:
                s1 = other - self.S1

        if self.C10 is None:
            c10 = None
        else:
            if isinstance(other, scat_cov):
                if other.C10 is None:
                    c10 = None
                else:
                    c10 = self.domin(other.C10 , self.C10 )
            else:
                c10 = other - self.C10
                
        if self.C11 is None:
            c11 = None
        else:
            if isinstance(other, scat_cov):
                if other.C11 is None:
                    c11 = None
                else:
                    c11 = self.domin( other.C11,self.C11 )
            else:
                c11 = other - self.C11

        if isinstance(other, scat_cov):
            return scat_cov(self.domin(other.P00,self.P00),
                            (other.C01 - self.C01),
                            c11,s1=s1, c10=c10,
                            backend=self.backend)
        else:
            return scat_cov((other-self.P00),
                            (other-self.C01),
                            c11,s1=s1, c10=c10,
                            backend=self.backend)
        
    def __sub__(self, other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
               isinstance(other, bool) or isinstance(other, scat_cov)

        if self.S1 is None:
            s1 = None
        else:
            if isinstance(other, scat_cov):
                if other.S1 is None:
                    s1 = None
                else:
                    s1 = self.S1 - other.S1
            else:
                s1 = self.S1 - other

        if self.C10 is None:
            c10 = None
        else:
            if isinstance(other, scat_cov):
                if other.C10 is None:
                    c10 = None
                else:
                    c10 = self.domin(self.C10 , other.C10)
            else:
                c10 = self.C10 - other
                
        if self.C11 is None:
            c11 = None
        else:
            if isinstance(other, scat_cov):
                if other.C11 is None:
                    c11 = None
                else:
                    c11 = self.domin(self.C11 , other.C11)
            else:
                c11 = self.C11 - other

        if isinstance(other, scat_cov):
            return scat_cov(self.domin(self.P00,other.P00),
                            (self.C01 - other.C01),
                            c11,
                            s1=s1, c10=c10,backend=self.backend)
        else:
            return scat_cov((self.P00 - other),
                            (self.C01 - other),
                            c11,
                            s1=s1, c10=c10,backend=self.backend)
        
    def domult(self,x,y):
        if x.dtype==y.dtype:
            return x*y
        if x.dtype=='complex64' or x.dtype=='complex128':
            
            return self.backend.bk_complex(self.backend.bk_real(x)*y,self.backend.bk_imag(x)*y)
        else:
            return self.backend.bk_complex(self.backend.bk_real(y)*x,self.backend.bk_imag(y)*x)
        
    def dodiv(self,x,y):
        if x.dtype==y.dtype:
            return x/y
        if x.dtype=='complex64' or x.dtype=='complex128':
            
            return self.backend.bk_complex(self.backend.bk_real(x)/y,self.backend.bk_imag(x)/y)
        else:
            return self.backend.bk_complex(x/self.backend.bk_real(y),x/self.backend.bk_imag(y))
        
    def domin(self,x,y):
        if x.dtype==y.dtype:
            return x-y
        if x.dtype=='complex64' or x.dtype=='complex128':
            
            return self.backend.bk_complex(self.backend.bk_real(x)-y,self.backend.bk_imag(x)-y)
        else:
            return self.backend.bk_complex(x-self.backend.bk_real(y),x-self.backend.bk_imag(y))
        
    def doadd(self,x,y):
        if x.dtype==y.dtype:
            return x+y
        if x.dtype=='complex64' or x.dtype=='complex128':
            
            return self.backend.bk_complex(self.backend.bk_real(x)+y,self.backend.bk_imag(x)+y)
        else:
            return self.backend.bk_complex(x+self.backend.bk_real(y),x+self.backend.bk_imag(y))
                
            
    def __mul__(self, other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
               isinstance(other, bool) or isinstance(other, scat_cov)

        if self.S1 is None:
            s1 = None
        else:
            if isinstance(other, scat_cov):
                if other.S1 is None:
                    s1 = None
                else:
                    s1 = self.S1 * other.S1
            else:
                s1 = self.S1 * other

        if self.C10 is None:
            c10 = None
        else:
            if isinstance(other, scat_cov):
                if other.C10 is None:
                    c10 = None
                else:
                    c10 = self.domult(self.C10 , other.C10)
            else:
                c10 = self.C10 * other

        if self.C11 is None:
            c11 = None
        else:
            if isinstance(other, scat_cov):
                if other.C11 is None:
                    c11 = None
                else:
                    c11 = self.domult(self.C11 , other.C11)
            else:
                c11 = self.C11 * other

        if isinstance(other, scat_cov):
            return scat_cov(self.domult(self.P00,other.P00),
                            self.domult(self.C01,other.C01),
                            c11,
                            s1=s1, c10=c10,backend=self.backend)
        else:
            return scat_cov((self.P00 * other),
                            (self.C01 * other),
                            c11,
                            s1=s1, c10=c10,backend=self.backend)

    
    def __rmul__(self, other):
        return self.__mul__(other)

    def plot(self, name=None, hold=True, color='blue', lw=1, legend=True):

        import matplotlib.pyplot as plt

        if name is None:
            name = ''

        if hold:
            plt.figure(figsize=(8, 8))

        if self.S1 is not None:
            plt.subplot(2, 2, 1)
            if legend:
                plt.plot(abs(self.get_np(self.S1)).flatten(), color=color, label=r'%s $S_1$' % (name), lw=lw)
            else:
                plt.plot(abs(self.get_np(self.S1)).flatten(), color=color, lw=lw)
            plt.yscale('log')
            plt.legend()

        plt.subplot(2, 2, 2)
        if legend:
            plt.plot(abs(self.get_np(self.P00)).flatten(), color=color, label=r'%s $P_{00}$' % (name), lw=lw)
        else:
            plt.plot(abs(self.get_np(self.P00)).flatten(), color=color, lw=lw)
        plt.yscale('log')
        plt.legend()

        plt.subplot(2, 2, 3)
        if legend:
            plt.plot(abs(self.get_np(self.C01)).flatten(), color=color, label=r'%s $C_{01}$' % (name), lw=lw)
            if self.C10 is not None:
                plt.plot(abs(self.get_np(self.C10)).flatten(), color=color, label=r'%s $C_{10}$' % (name), lw=lw)
        else:
            plt.plot(abs(self.get_np(self.C01)).flatten(), color=color, lw=lw)
            if self.C10 is not None:
                plt.plot(abs(self.get_np(self.C10)).flatten(), color=color, lw=lw)
        plt.yscale('log')
        plt.legend()

        plt.subplot(2, 2, 4)
        if legend:
            plt.plot(abs(self.get_np(self.C11)).flatten(), color=color, label=r'%s $C_{11}$' % (name), lw=lw)
        else:
            plt.plot(abs(self.get_np(self.C11)).flatten(), color=color, lw=lw)
        plt.yscale('log')
        plt.legend()

    def get_np(self, x):
        if x is not None:
            if isinstance(x, np.ndarray):
                return x
            else:
                return x.numpy()
        
    def save(self, filename):
        if self.S1 is not None:  # Auto
            np.save('%s_s1.npy' % filename, self.get_np(self.get_S1()))
        if self.C10 is not None:  # Cross
            np.save('%s_c10.npy' % filename, self.get_np(self.get_C10()))
        np.save('%s_c01.npy' % filename, self.get_np(self.get_C01()))
        np.save('%s_c11.npy' % filename, self.get_np(self.get_C11()))
        np.save('%s_p0.npy' % filename, self.get_np(self.get_P00()))

    def read(self, filename):
        try:
            s1 = np.load('%s_s1.npy' % filename)
        except:
            s1 = None

        try:
            c10 = np.load('%s_c10.npy' % filename)
        except:
            c10 = None

        c01 = np.load('%s_c01.npy' % filename)
        c11 = np.load('%s_c11.npy' % filename)
        p0 = np.load('%s_p0.npy' % filename)

        return scat_cov(p0, c01, c11, s1=s1, c10=c10,backend=self.backend)

    def std(self):
        if self.S1 is not None:  # Auto
            return np.sqrt(((abs(self.get_np(self.S1)).std()) ** 2 +
                            (abs(self.get_np(self.C01)).std()) ** 2 +
                            (abs(self.get_np(self.C11)).std()) ** 2 +
                            (abs(self.get_np(self.P00)).std()) ** 2 ) / 4)
        else:  # Cross
            return np.sqrt(((abs(self.get_np(self.C01)).std()) ** 2 +
                            (abs(self.get_np(self.C10)).std()) ** 2 +
                            (abs(self.get_np(self.C11)).std()) ** 2 +
                            (abs(self.get_np(self.P00)).std()) ** 2) / 4)

    def mean(self):
        if self.S1 is not None:  # Auto
            return (abs(self.get_np(self.S1)).mean() +
                    abs(self.get_np(self.C01)).mean() +
                    abs(self.get_np(self.C11)).mean() +
                    abs(self.get_np(self.P00)).mean()) / 4
        else:  # Cross
            return (abs(self.get_np(self.C01)).mean() +
                    abs(self.get_np(self.C10)).mean() +
                    abs(self.get_np(self.C11)).mean() +
                    abs(self.get_np(self.P00)).mean()) / 4

    def get_nscale(self):
        return self.P00.shape[2]
    
    def get_norient(self):
        return self.P00.shape[3]

    def add_data_from_log_slope(self,y,n,ds=3):
        if len(y)<ds:
            if len(y)==1:
                return(np.repeat(y[0],n))
            if len(y)==2:
                a=np.polyfit(np.arange(2),np.log(y[0:2]),1)
        else:
            a=np.polyfit(np.arange(ds),np.log(y[0:ds]),1)
        return np.exp((np.arange(n)-1-n)*a[0]+a[1])

    def add_data_from_slope(self,y,n,ds=3):
        if len(y)<ds:
            if len(y)==1:
                return(np.repeat(y[0],n))
            if len(y)==2:
                a=np.polyfit(np.arange(2),y[0:2],1)
        else:
            a=np.polyfit(np.arange(ds),y[0:ds],1)
        return (np.arange(n)-1-n)*a[0]+a[1]
    
    def up_grade(self,nscale,ds=3):
        noff=nscale-self.P00.shape[2]
        if noff==0:
            return scat_cov((self.P00),
                            (self.C01),
                            (self.C11),
                            s1=self.S1,
                            c10=self.C10,backend=self.backend)
        
        inscale=self.P00.shape[2]
        p00=np.zeros([self.P00.shape[0],self.P00.shape[1],nscale,self.P00.shape[3]],dtype='complex')
        p00[:,:,noff:,:]=self.P00.numpy()
        for i in range(self.P00.shape[0]):
            for j in range(self.P00.shape[1]):
                for k in range(self.P00.shape[3]):
                    p00[i,j,0:noff,k]=self.add_data_from_log_slope(p00[i,j,noff:,k],noff,ds=ds)
                    
        s1=np.zeros([self.S1.shape[0],self.S1.shape[1],nscale,self.S1.shape[3]])
        s1[:,:,noff:,:]=self.S1.numpy()
        for i in range(self.S1.shape[0]):
            for j in range(self.S1.shape[1]):
                for k in range(self.S1.shape[3]):
                    s1[i,j,0:noff,k]=self.add_data_from_log_slope(s1[i,j,noff:,k],noff,ds=ds)

        nout=0
        for i in range(1,nscale):
            nout=nout+i
            
        c01=np.zeros([self.C01.shape[0],self.C01.shape[1], \
                      nout,self.C01.shape[3],self.C01.shape[4]],dtype='complex')
                     
        jo1=np.zeros([nout])
        jo2=np.zeros([nout])

        n=0
        for i in range(1,nscale):
            jo1[n:n+i]=np.arange(i)
            jo2[n:n+i]=i
            n=n+i
            
        j1=np.zeros([self.C01.shape[2]])
        j2=np.zeros([self.C01.shape[2]])
        
        n=0
        for i in range(1,self.P00.shape[2]):
            j1[n:n+i]=np.arange(i)
            j2[n:n+i]=i
            n=n+i

        for i in range(self.C01.shape[0]):
            for j in range(self.C01.shape[1]):
                for k in range(self.C01.shape[3]):
                    for l in range(self.C01.shape[4]):
                        for ij in range(noff+1,nscale):
                            idx=np.where(jo2==ij)[0]
                            c01[i,j,idx[noff:],k,l]=self.C01.numpy()[i,j,j2==ij-noff,k,l]
                            c01[i,j,idx[:noff],k,l]=self.add_data_from_slope(self.C01.numpy()[i,j,j2==ij-noff,k,l],noff,ds=ds)

                        for ij in range(nscale):
                            idx=np.where(jo1==ij)[0]
                            if idx.shape[0]>noff:
                                c01[i,j,idx[:noff],k,l]=self.add_data_from_slope(c01[i,j,idx[noff:],k,l],noff,ds=ds)
                            else:
                                c01[i,j,idx,k,l]=np.mean(c01[i,j,jo1==ij-1,k,l])

        
        nout=0
        for j3 in range(nscale):
            for j2 in range(0,j3):
                for j1 in range(0,j2):
                    nout=nout+1

        c11=np.zeros([self.C11.shape[0],self.C11.shape[1], \
                      nout,self.C11.shape[3], \
                      self.C11.shape[4],self.C11.shape[5]],dtype='complex')
                     
        jo1=np.zeros([nout])
        jo2=np.zeros([nout])
        jo3=np.zeros([nout])

        nout=0
        for j3 in range(nscale):
            for j2 in range(0,j3):
                for j1 in range(0,j2):
                    jo1[nout]=j1
                    jo2[nout]=j2
                    jo3[nout]=j3
                    nout=nout+1

        ncross=self.C11.shape[2]
        jj1=np.zeros([ncross])
        jj2=np.zeros([ncross])
        jj3=np.zeros([ncross])
        
        n=0
        for j3 in range(inscale):
            for j2 in range(0,j3):
                for j1 in range(0,j2):
                    jj1[n]=j1
                    jj2[n]=j2
                    jj3[n]=j3
                    n=n+1

        n=0
        for j3 in range(nscale):
            for j2 in range(j3):
                idx=np.where((jj3==j3)*(jj2==j2))[0]
                if idx.shape[0]>0:
                    idx2=np.where((jo3==j3+noff)*(jo2==j2+noff))[0]
                    for i in range(self.C11.shape[0]):
                        for j in range(self.C11.shape[1]):
                            for k in range(self.C11.shape[3]):
                                for l in range(self.C11.shape[4]):
                                    for m in range(self.C11.shape[5]):
                                        c11[i,j,idx2[noff:],k,l,m]=self.C11.numpy()[i,j,idx,k,l,m]
                                        c11[i,j,idx2[:noff],k,l,m]=self.add_data_from_log_slope(self.C11.numpy()[i,j,idx,k,l,m],noff,ds=ds)

        idx=np.where(abs(c11[0,0,:,0,0,0])==0)[0]
        for iii in idx:
            iii1=np.where((jo1==jo1[iii]+1)*(jo2==jo2[iii]+1)*(jo3==jo3[iii]+1))[0]
            iii2=np.where((jo1==jo1[iii]+2)*(jo2==jo2[iii]+2)*(jo3==jo3[iii]+2))[0]
            if iii2.shape[0]>0:
                for i in range(self.C11.shape[0]):
                    for j in range(self.C11.shape[1]):
                        for k in range(self.C11.shape[3]):
                            for l in range(self.C11.shape[4]):
                                for m in range(self.C11.shape[5]):
                                    c11[i,j,iii,k,l,m]=self.add_data_from_slope(c11[i,j,[iii1,iii2],k,l,m],1,ds=2)[0]
        
        idx=np.where(abs(c11[0,0,:,0,0,0])==0)[0]
        for iii in idx:
            iii1=np.where((jo1==jo1[iii])*(jo2==jo2[iii])*(jo3==jo3[iii]-1))[0]
            iii2=np.where((jo1==jo1[iii])*(jo2==jo2[iii])*(jo3==jo3[iii]-2))[0]
            if iii2.shape[0]>0:
                for i in range(self.C11.shape[0]):
                    for j in range(self.C11.shape[1]):
                        for k in range(self.C11.shape[3]):
                            for l in range(self.C11.shape[4]):
                                for m in range(self.C11.shape[5]):
                                    c11[i,j,iii,k,l,m]=self.add_data_from_slope(c11[i,j,[iii1,iii2],k,l,m],1,ds=2)[0]

        return scat_cov( (p00),
                         (c01),
                         (c11),
                        s1=(s1),backend=self.backend)
        
        

class funct(FOC.FoCUS):

    def eval(self, image1, image2=None, mask=None, norm=None, Auto=True):
        """
        Calculates the scattering correlations for a batch of images. Mean are done over pixels.
        mean of modulus:
                        S1 = <|I * Psi_j3|>
             Normalization : take the log
        power spectrum:
                        P00 = <|I * Psi_j3|^2>
            Normalization : take the log
        orig. x modulus:
                        C01 = < (I * Psi)_j3 x (|I * Psi_j2| * Psi_j3)^* >
             Normalization : divide by (P00_j2 * P00_j3)^0.5
        modulus x modulus:
                        C11 = <(|I * psi1| * psi3)(|I * psi2| * psi3)^*>
             Normalization : divide by (P00_j1 * P00_j2)^0.5
        Parameters
        ----------
        image1: tensor
            Image on which we compute the scattering coefficients [Nbatch, Npix, 1, 1]
        image2: tensor
            Second image. If not None, we compute cross-scattering covariance coefficients.
        mask:
        norm: None or str
            If None no normalization is applied, if 'auto' normalize by the reference P00,
            if 'self' normalize by the current P00.
        all_cross: False or True
            If False compute all the coefficient even the Imaginary part,
            If True return only the terms computable in the auto case.
        Returns
        -------
        S1, P00, C01, C11 normalized
        """
        ### AUTO OR CROSS
        cross = False
        if image2 is not None:
            cross = True
            all_cross=Auto
        else:
            all_cross=False

        ### PARAMETERS
        axis = 1
        # determine jmax and nside corresponding to the input map
        im_shape = image1.shape
        if self.use_R_format and isinstance(image1, Rformat.Rformat):
            if len(image1.shape) == 4:
                nside = im_shape[2] - 2 * self.R_off
            else:
                nside = im_shape[1] - 2 * self.R_off
            npix = 12 * nside * nside  # Number of pixels
        else:
            if self.chans==1:
                nside=im_shape[axis]
                npix=nside*nside
            else:
                npix = image1.shape[-1]  # image1 is [Npix] or [Nbatch, Npix]
                nside = int(np.sqrt(npix / 12))

        J = int(np.log(nside) / np.log(2))  # Number of j scales
        Jmax = J - self.OSTEP  # Number of steps for the loop on scales

        ### LOCAL VARIABLES (IMAGES and MASK)
        # Check if image1 is [Npix] or [Nbatch, Npix] or Rformat
        if len(image1.shape) == 1 or (len(image1.shape)==2 and self.chans==1) or (len(image1.shape) == 3 and isinstance(image1, Rformat.Rformat)):
            I1 = self.backend.bk_cast(self.backend.bk_expand_dims(image1, 0))  # Local image1 [Nbatch, Npix]
            if cross:
                I2 = self.backend.bk_cast(self.backend.bk_expand_dims(image2, 0))  # Local image2 [Nbatch, Npix]
        else:
            I1 = self.backend.bk_cast(image1)  # Local image1 [Nbatch, Npix]
            if cross:
                I2 = self.backend.bk_cast(image2)  # Local image2 [Nbatch, Npix]
                
        if mask is None:
            if self.chans==1:
                vmask = self.backend.bk_ones([1, nside, nside],dtype=self.all_type)
            else:
                vmask = self.backend.bk_ones([1, npix], dtype=self.all_type)
                
            if self.use_R_format:
                vmask = self.to_R(vmask, axis=1,chans=self.chans)
        else:
            vmask = self.backend.bk_cast(mask)  # [Nmask, Npix]
            if self.use_R_format:
                vmask = self.to_R(vmask, axis=1,chans=self.chans)

        if self.use_R_format:
            I1 = self.to_R(I1, axis=axis,chans=self.chans)
            if cross:
                I2 = self.to_R(I2, axis=axis,chans=self.chans)

        if self.KERNELSZ > 3:
            # if the kernel size is bigger than 3 increase the binning before smoothing
            I1 = self.up_grade(I1, nside * 2, axis=axis)
            vmask = self.up_grade(vmask, nside * 2, axis=1)
            if cross:
                I2 = self.up_grade(I2, nside * 2, axis=axis)
                
        # Normalize the masks because they have different pixel numbers
        # vmask /= self.backend.bk_reduce_sum(vmask, axis=1)[:, None]  # [Nmask, Npix]

        ### INITIALIZATION
        # Coefficients
        S1, P00, C01, C11, C10 = None, None, None, None, None

        # Dictionaries for C01 computation
        M1_dic = {}  # M stands for Module M1 = |I1 * Psi|
        if cross:
            M2_dic = {}

        # P00 for normalization
        cond_init_P1_dic = ((norm == 'self') or ((norm == 'auto') and (self.P1_dic is None)))
        if norm is None:
            pass
        elif cond_init_P1_dic:
            P1_dic = {}
            if cross:
                P2_dic = {}
        elif (norm == 'auto') and (self.P1_dic is not None):
            P1_dic = self.P1_dic
            if cross:
                P2_dic = self.P2_dic

        #### COMPUTE S1, P00, C01 and C11
        nside_j3 = nside  # NSIDE start (nside_j3 = nside / 2^j3)
        for j3 in range(Jmax):

            ####### S1 and P00
            ### Make the convolution I1 * Psi_j3
            conv1 = self.convol(I1, axis=1)  # [Nbatch, Npix_j3, Norient3]
            ### Take the module M1 = |I1 * Psi_j3|
            M1_square = conv1*self.backend.bk_conjugate(conv1) # [Nbatch, Npix_j3, Norient3]
            M1 = self.backend.bk_L1(M1_square)  # [Nbatch, Npix_j3, Norient3]
            # Store M1_j3 in a dictionary
            M1_dic[j3] = self.update_R_border(M1, axis=axis)

            if not cross:  # Auto
                M1_square=self.backend.bk_real(M1_square)
                
                ### P00_auto = < M1^2 >_pix
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                p00 = self.masked_mean(M1_square, vmask, axis=1)
                if cond_init_P1_dic:
                    # We fill P1_dic with P00 for normalisation of C01 and C11
                    P1_dic[j3] = p00  # [Nbatch, Nmask, Norient3]
                if norm == 'auto':  # Normalize P00
                    p00 /= P1_dic[j3]

                # We store P00_auto to return it [Nbatch, Nmask, NP00, Norient3]
                if P00 is None:
                    P00 = p00[:, :, None, :]  # Add a dimension for NP00
                else:
                    P00 = self.backend.bk_concat([P00, p00[:, :, None, :]], axis=2)

                #### S1_auto computation
                ### Image 1 : S1 = < M1 >_pix
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                s1 = self.masked_mean(M1, vmask, axis=1)  # [Nbatch, Nmask, Norient3]
                ### Normalize S1
                if norm is not None:
                    s1 /= (P1_dic[j3]) ** 0.5
                ### We store S1 for image1  [Nbatch, Nmask, NS1, Norient3]
                if S1 is None:
                    S1 = s1[:, :, None, :]  # Add a dimension for NS1
                else:
                    S1 = self.backend.bk_concat([S1, s1[:, :, None, :]], axis=2)

            else:  # Cross
                ### Make the convolution I2 * Psi_j3
                conv2 = self.convol(I2, axis=1)  # [Nbatch, Npix_j3, Norient3]
                ### Take the module M2 = |I2 * Psi_j3|
                M2_square = conv2*self.backend.bk_conjugate(conv2)  # [Nbatch, Npix_j3, Norient3]
                M2 = self.backend.bk_L1(M2_square)  # [Nbatch, Npix_j3, Norient3]
                # Store M2_j3 in a dictionary
                M2_dic[j3] = self.update_R_border(M2, axis=axis)

                ### P00_auto = < M2^2 >_pix
                # Not returned, only for normalization
                if cond_init_P1_dic:
                    # Apply the mask [Nmask, Npix_j3] and average over pixels
                    p1 = self.masked_mean(M1_square, vmask, axis=1)  # [Nbatch, Nmask, Norient3]
                    p2 = self.masked_mean(M2_square, vmask, axis=1)  # [Nbatch, Nmask, Norient3]
                    # We fill P1_dic with P00 for normalisation of C01 and C11
                    P1_dic[j3] = p1  # [Nbatch, Nmask, Norient3]
                    P2_dic[j3] = p2  # [Nbatch, Nmask, Norient3]

                ### P00_cross = < (I1 * Psi_j3) (I2 * Psi_j3)^* >_pix
                # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
                p00 = conv1 * self.backend.bk_conjugate(conv2)
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                p00 = self.masked_mean(p00, vmask, axis=1)

                ### Normalize P00_cross
                if norm == 'auto':
                    p00 /= (P1_dic[j3] * P2_dic[j3])**0.5

                ### Store P00_cross as complex [Nbatch, Nmask, NP00, Norient3]
                if not all_cross:
                    p00=self.backend.bk_real(p00)
                    
                if P00 is None:
                    P00 = p00[:,:,None,:]  # Add a dimension for NP00
                else:
                    P00 = self.backend.bk_concat([P00, p00[:,:,None,:]], axis=2)
                    
            # Initialize dictionaries for |I1*Psi_j| * Psi_j3
            M1convPsi_dic = {}
            if cross:
                # Initialize dictionaries for |I2*Psi_j| * Psi_j3
                M2convPsi_dic = {}

            ###### C01
            for j2 in range(0, j3):  # j2 < j3
                ### C01_auto = < (I1 * Psi)_j3 x (|I1 * Psi_j2| * Psi_j3)^* >_pix
                if not cross:
                    c01 = self._compute_C01(j2,
                                            conv1,
                                            vmask,
                                            M1_dic,
                                            M1convPsi_dic)  # [Nbatch, Nmask, Norient3, Norient2]
                    ### Normalize C01 with P00_j [Nbatch, Nmask, Norient_j]
                    if norm is not None:
                        c01 /= (P1_dic[j2][:, :, None, :] *
                                P1_dic[j3][:, :, :, None]) ** 0.5  # [Nbatch, Nmask, Norient3, Norient2]
                        
                    ### Store C01 as a complex [Nbatch, Nmask, NC01, Norient3, Norient2]
                    if C01 is None:
                        C01 = c01[:,:,None,:,:]  # Add a dimension for NC01
                    else:
                        C01 = self.backend.bk_concat([C01, c01[:, :, None, :, :]],
                                                 axis=2)  # Add a dimension for NC01

                ### C01_cross = < (I1 * Psi)_j3 x (|I2 * Psi_j2| * Psi_j3)^* >_pix
                ### C10_cross = < (I2 * Psi)_j3 x (|I1 * Psi_j2| * Psi_j3)^* >_pix
                else:
                    c01 = self._compute_C01(j2,
                                            conv1,
                                            vmask,
                                            M2_dic,
                                            M2convPsi_dic)
                    c10 = self._compute_C01(j2,
                                            conv2,
                                            vmask,
                                            M1_dic,
                                            M1convPsi_dic)
                    
                    ### Normalize C01 and C10 with P00_j [Nbatch, Nmask, Norient_j]
                    if norm is not None:
                        c01 /= (P2_dic[j2][:, :, None, :] *
                                P1_dic[j3][:, :, :, None]) ** 0.5  # [Nbatch, Nmask, Norient3, Norient2]
                        c10 /= (P1_dic[j2][:, :, None, :] *
                                P2_dic[j3][:, :, :, None]) ** 0.5  # [Nbatch, Nmask, Norient3, Norient2]
                        
                    ### Store C01 and C10 as a complex [Nbatch, Nmask, NC01, Norient3, Norient2]
                    if C01 is None:
                        C01 = c01[:, :, None, :, :] # Add a dimension for NC01
                    else:
                        C01 = self.backend.bk_concat([C01,c01[:, :, None, :, :]],axis=2)  # Add a dimension for NC01
                    if C10 is None:
                        C10 = c10[:, :, None, :, :]  # Add a dimension for NC01
                    else:
                        C10 = self.backend.bk_concat([C10,c10[:, :, None, :, :]], axis=2)  # Add a dimension for NC01
                        


                ##### C11
                for j1 in range(0, j2):  # j1 < j2
                    ### C11_auto = <(|I1 * psi1| * psi3)(|I1 * psi2| * psi3)^*>
                    if not cross:
                        c11 = self._compute_C11(j1, j2, vmask,
                                                M1convPsi_dic,
                                                M2convPsi_dic=None) # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        ### Normalize C11 with P00_j [Nbatch, Nmask, Norient_j]
                        if norm is not None:
                            c11 /= (P1_dic[j1][:, :, None, None, :] *
                                    P1_dic[j2][:, :, None, :,
                                               None]) ** 0.5  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        ### Store C11 as a complex [Nbatch, Nmask, NC11, Norient3, Norient2, Norient1]
                        if C11 is None:
                            C11 = c11[:, :, None, :, :, :]  # Add a dimension for NC11
                        else:
                            C11 = self.backend.bk_concat([C11,c11[:, :, None, :, :, :]],
                                                 axis=2)  # Add a dimension for NC11

                        ### C11_cross = <(|I1 * psi1| * psi3)(|I2 * psi2| * psi3)^*>
                    else:
                        c11 = self._compute_C11(j1, j2, vmask,
                                                M1convPsi_dic,
                                                M2convPsi_dic=M2convPsi_dic)  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        ### Normalize C11 with P00_j [Nbatch, Nmask, Norient_j]
                        if norm is not None:
                            c11 /= (P1_dic[j1][:, :, None, None, :] *
                                    P2_dic[j2][:, :, None, :, None]) ** 0.5  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        ### Store C11 as a complex [Nbatch, Nmask, NC11, Norient3, Norient2, Norient1]
                        if C11 is None:
                            C11 = c11[:, :, None, :, :, :]  # Add a dimension for NC11
                        else:
                            C11 = self.backend.bk_concat([C11,c11[:, :, None, :, :, :]],
                                                 axis=2)  # Add a dimension for NC11
            
            ###### Reshape for next iteration on j3
            ### Image I1,
            # downscale the I1 [Nbatch, Npix_j3]
            if j3 != Jmax - 1:
                I1_smooth = self.smooth(I1, axis=1)
                I1 = self.ud_grade_2(I1_smooth, axis=1)

                ### Image I2
                if cross:
                    I2_smooth = self.smooth(I2, axis=1)
                    I2 = self.ud_grade_2(I2_smooth, axis=1)

                ### Modules
                for j2 in range(0, j3 + 1):  # j2 =< j3
                    ### Dictionary M1_dic[j2]
                    M1_smooth = self.smooth(M1_dic[j2], axis=1)  # [Nbatch, Npix_j3, Norient3]
                    M1_dic[j2] = self.ud_grade_2(M1_smooth, axis=1)  # [Nbatch, Npix_j3, Norient3]

                    ### Dictionary M2_dic[j2]
                    if cross:
                        M2_smooth = self.smooth(M2_dic[j2], axis=1)  # [Nbatch, Npix_j3, Norient3]
                        M2_dic[j2] = self.ud_grade_2(M2_smooth, axis=1)  # [Nbatch, Npix_j3, Norient3]
                ### Mask
                vmask = self.ud_grade_2(vmask, axis=1)

                ### NSIDE_j3
                nside_j3 = nside_j3 // 2

        ### Store P1_dic and P2_dic in self
        if (norm == 'auto') and (self.P1_dic is None):
            self.P1_dic = P1_dic
            if cross:
                self.P2_dic = P2_dic
            
        if not cross:
            return scat_cov(P00, C01, C11, s1=S1,backend=self.backend)
        else:
            return scat_cov(P00, C01, C11, c10=C10,backend=self.backend)

    def clean_norm(self):
        self.P1_dic = None
        self.P2_dic = None
        return

    def _compute_C01(self, j2, conv,
                     vmask, M_dic,
                     MconvPsi_dic):
        """
        Compute the C01 coefficients (auto or cross)
        C01 = < (Ia * Psi)_j3 x (|Ib * Psi_j2| * Psi_j3)^* >_pix
        Parameters
        ----------
        Returns
        -------
        cc01, sc01: real and imag parts of C01 coeff
        """
        ### Compute |I1 * Psi_j2| * Psi_j3 = M1_j2 * Psi_j3
        # Warning: M1_dic[j2] is already at j3 resolution [Nbatch, Npix_j3, Norient3]
        MconvPsi = self.convol(M_dic[j2], axis=1)  # [Nbatch, Npix_j3, Norient3, Norient2]

        # Store it so we can use it in C11 computation
        MconvPsi_dic[j2] = MconvPsi  # [Nbatch, Npix_j3, Norient3, Norient2]

        ### Compute the product (I2 * Psi)_j3 x (M1_j2 * Psi_j3)^*
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        # cconv, sconv are [Nbatch, Npix_j3, Norient3]
        c01 = self.backend.bk_expand_dims(conv, -1) * self.backend.bk_conjugate(MconvPsi)   # [Nbatch, Npix_j3, Norient3, Norient2]

        ### Apply the mask [Nmask, Npix_j3] and sum over pixels
        c01 = self.masked_mean(c01, vmask, axis=1)  # [Nbatch, Nmask, Norient3, Norient2]
        return c01

    def _compute_C11(self, j1, j2, vmask,
                     M1convPsi_dic,
                     M2convPsi_dic=None):
        #### Simplify notations
        M1 = M1convPsi_dic[j1]  # [Nbatch, Npix_j3, Norient3, Norient1]
        
        # Auto or Cross coefficients
        if M2convPsi_dic is None:  # Auto
            M2 = M1convPsi_dic[j2]  # [Nbatch, Npix_j3, Norient3, Norient2]
        else:  # Cross
            M2 = M2convPsi_dic[j2]

        ### Compute the product (|I1 * Psi_j1| * Psi_j3)(|I2 * Psi_j2| * Psi_j3)
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        c11 = self.backend.bk_expand_dims(M1, -2) * self.backend.bk_conjugate(self.backend.bk_expand_dims(M2, -1))  # [Nbatch, Npix_j3, Norient3, Norient2, Norient1]

        ### Apply the mask and sum over pixels
        c11 = self.masked_mean(c11, vmask, axis=1)  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
        return c11

    def square(self, x):
        if isinstance(x, scat_cov):
            if x.S1 is None:
                return scat_cov(self.backend.bk_square(self.backend.bk_abs(x.P00)),
                                self.backend.bk_square(self.backend.bk_abs(x.C01)),
                                self.backend.bk_square(self.backend.bk_abs(x.C11)),backend=self.backend)
            else:
                return scat_cov(self.backend.bk_square(self.backend.bk_abs(x.P00)),
                                self.backend.bk_square(self.backend.bk_abs(x.C01)),
                                self.backend.bk_square(self.backend.bk_abs(x.C11)),
                                s1=self.backend.bk_square(self.backend.bk_abs(x.S1)),backend=self.backend)
        else:
            return self.backend.bk_abs(self.backend.bk_square(x))

    def sqrt(self, x):
        if isinstance(x, scat_cov):
            if x.S1 is None:
                return scat_cov(self.backend.bk_sqrt(self.backend.bk_abs(x.P00)),
                                self.backend.bk_sqrt(self.backend.bk_abs(x.C01)),
                                self.backend.bk_sqrt(self.backend.bk_abs(x.C11)),backend=self.backend)
            else:
                return scat_cov(self.backend.bk_sqrt(self.backend.bk_abs(x.P00)),
                                self.backend.bk_sqrt(self.backend.bk_abs(x.C01)),
                                self.backend.bk_sqrt(self.backend.bk_abs(x.C11)),
                                s1=self.backend.bk_sqrt(self.backend.bk_abs(x.S1)),backend=self.backend)
        else:
            return self.backend.bk_abs(self.backend.bk_sqrt(x))

    def reduce_mean(self, x):
        if isinstance(x, scat_cov):
            if x.S1 is None:
                result = (self.backend.bk_reduce_mean(self.backend.bk_abs(x.P00)) + \
                         self.backend.bk_reduce_mean(self.backend.bk_abs(x.C01)) + \
                          self.backend.bk_reduce_mean(self.backend.bk_abs(x.C11)))/3
            else:
                result = (self.backend.bk_reduce_mean(self.backend.bk_abs(x.P00)) + \
                         self.backend.bk_reduce_mean(self.backend.bk_abs(x.S1)) + \
                         self.backend.bk_reduce_mean(self.backend.bk_abs(x.C01)) + \
                         self.backend.bk_reduce_mean(self.backend.bk_abs(x.C11)))/4
        else:
            return self.backend.bk_reduce_mean(x)
        return result

    def reduce_sum(self, x):
        
        if isinstance(x, scat_cov):
            if x.S1 is None:
                result = self.backend.bk_reduce_sum(x.P00) + \
                         self.backend.bk_reduce_sum(x.C01) + \
                         self.backend.bk_reduce_sum(x.C11)
            else:
                result = self.backend.bk_reduce_sum(x.P00) + \
                         self.backend.bk_reduce_sum(x.S1) + \
                         self.backend.bk_reduce_sum(x.C01) + \
                         self.backend.bk_reduce_sum(x.C11)
        else:
            return self.backend.bk_reduce_sum(x)
        return result

        
    def ldiff(self,sig,x):

                
        if x.S1 is None:
            return scat_cov(x.domult(sig.P00,x.P00)*x.domult(sig.P00,x.P00),
                            x.domult(sig.C01,x.C01)*x.domult(sig.C01,x.C01),
                            x.domult(sig.C11,x.C11)*x.domult(sig.C11,x.C11),
                            backend=self.backend)
        else:
            return scat_cov(x.domult(sig.P00,x.P00)*x.domult(sig.P00,x.P00),
                            x.domult(sig.S1,x.S1)*x.domult(sig.S1,x.S1),
                            x.domult(sig.C01,x.C01)*x.domult(sig.C01,x.C01),
                            x.domult(sig.C11,x.C11)*x.domult(sig.C11,x.C11),
                            backend=self.backend)
    
    def log(self, x):
        if isinstance(x, scat_cov):

            if x.S1 is None:
                result = self.backend.bk_log(x.P00) + \
                         self.backend.bk_log(x.C01) + \
                         self.backend.bk_log(x.C11)
            else:
                result = self.backend.bk_log(x.P00) + \
                         self.backend.bk_log(x.S1) + \
                         self.backend.bk_log(x.C01) + \
                         self.backend.bk_log(x.C11)
        else:
            return self.backend.bk_log(x)
        
        return result
