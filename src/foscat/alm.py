import healpy as hp
import numpy as np

class alm():

    def __init__(self,backend=None,lmax=24,limit_range=1E7):
        self._logtab={}
        self.lmax=0
        for k in range(1,2*lmax+1):
            self._logtab[k]=np.log(k) 
        self._limit_range=1/limit_range
        self._log_limit_range=np.log(limit_range)
        if backend is None:
            import foscat.scat_cov as sc
            self.sc=sc.funct()
            self.backend=self.sc.backend
        else:
            self.backend=backend.backend
    
    def log(self,v):
        #return np.log(v)
        if isinstance(v,np.ndarray):
            return np.array([self.log(k) for k in v])
        if v<self.lmax*2+1:
            return self._logtab[v]
        else:
            self._logtab[v]=np.log(v)
        return self._logtab[v]

    # Fonction pour calculer la double factorielle
    def double_factorial_log(self,n):
        if n <= 0:
            return 0.0
        result = 0.0
        for i in range(n, 0, -2):
            result += self.log(i)
        return result

    # Calcul des P_{lm}(x) pour tout l inclus dans [m,lmax]
    def compute_legendre_m(self,x,m,lmax):
        # Étape 1 : Calcul de P_{mm}(x)
        if m == 0:
            Pmm = 1.0
        else:
            Pmm = (-1)**m * (1 - x**2)**(m/2)
        
        result=np.zeros([lmax-m+1,x.shape[0]])
        ratio=np.zeros([lmax-m+1,1])
        
        # Si l == m, c'est directement P_{mm}
        result[0]=Pmm
        ratio[0,0]= self.double_factorial_log(2*m - 1)-0.5*np.sum(self.log(1+np.arange(2*m)))
    
        if m == lmax:
            return result*np.exp(ratio)*np.sqrt((2*(np.arange(lmax-m+1)-m))/(4*np.pi)).reshape(lmax+1-m,1)
    
        # Étape 2 : Calcul de P_{l+1, m}(x)
        result[1] = x * (2*m + 1) * result[0]

        ratio[1,0]=ratio[0,0]-0.5*self.log(2*m+1)
    
        # Étape 3 : Récurence pour l > m + 1
        for l in range(m + 2, lmax+1):
            result[l-m] = ((2*l - 1) * x * result[l-m-1] - (l + m - 1) *  result[l-m-2]) / (l - m)
            ratio[l-m,0] = 0.5*self.log(l-m)-0.5*self.log(l+m)+ratio[l-m-1,0]
            if np.max(abs(result[l-m]))>self._limit_range:
                result[l-m-1]*=self._limit_range
                result[l-m]*=self._limit_range
                ratio[l-m-1,0]+=self._log_limit_range
                ratio[l-m,0]+=self._log_limit_range
        
        return result*np.exp(ratio)*(np.sqrt(4*np.pi*(2*(np.arange(lmax-m+1)+m)+1))).reshape(lmax+1-m,1)
    
    def comp_tf(self,im,ph):
        nside=int(np.sqrt(im.shape[0]//12))
        n=0
        ii=0
        ft_im=[]
        for k in range(nside-1):
            N=4*(k+1)
            ft_im.append(self.backend.bk_fft(im[n:n+N])[:N//2+1]*np.exp(-1J*np.arange(N//2+1)/N*ph[n]))
            ft_im.append(self.backend.bk_zeros((3*nside-N//2-1),dtype=self.backend.all_cbk_type))
            n+=N
            ii+=1
        for k in range(2*nside+1):
            N=4*nside
            ft_im.append(self.backend.bk_fft(im[n:n+N])[:N//2+1]*np.exp(-1J*np.arange(N//2+1)/N*ph[n]))
            ft_im.append(self.backend.bk_zeros((3*nside-N//2-1),dtype=self.backend.all_cbk_type))
            n+=N
            ii+=1
        for k in range(nside-1):
            N=4*(nside-1-k)
            ft_im.append(self.backend.bk_fft(im[n:n+N])[:N//2+1]*np.exp(-1J*np.arange(N//2+1)/N*ph[n]))
            ft_im.append(self.backend.bk_zeros((3*nside-N//2-1),dtype=self.backend.all_cbk_type))
            n+=N
            ii+=1
        return self.backend.bk_reshape(self.backend.bk_concat(ft_im,axis=0),[4*nside-1,3*nside])

    def anafast(self,im,map2=None):
        nside=int(np.sqrt(im.shape[0]//12))
        th,ph=hp.pix2ang(nside,np.arange(12*nside*nside))
        ft_im=self.comp_tf(self.backend.bk_complex(im,0*im),ph)
        if map2 is not None:
            ft_im2=self.comp_tf(self.backend.bk_complex(map2,0*im),ph)
            
        co_th=np.cos(np.unique(th))

        lmax=3*nside-1
    
        cl2=None
        cl2_L1=None
        for m in range(lmax+1):
            plm=self.compute_legendre_m(co_th,m,3*nside-1)/(12*nside**2)
            
            tmp=self.backend.bk_reduce_sum(plm*ft_im[:,m],1)
            
            if map2 is not None:
                tmp2=self.backend.bk_reduce_sum(plm*ft_im2[:,m],1)
            else:
                tmp2=tmp
                
            tmp=self.backend.bk_real((tmp*self.backend.bk_conjugate(tmp2)))
            if cl2 is None:
                cl2=tmp
                cl2_l1=self.backend.bk_L1(tmp)
            else:
                tmp=self.backend.bk_concat([self.backend.bk_zeros((m),dtype=self.backend.all_bk_type),tmp],axis=0)
                cl2+=2*tmp
                cl2_l1+=2*self.backend.bk_L1(tmp)
        cl2=cl2*(1+np.clip((np.arange(cl2.shape[0])-2*nside)/(3*nside),0,1))/(2*np.arange(cl2.shape[0])+1)* \
            (1+np.clip((np.arange(cl2.shape[0])-2.4*nside)/(2.5*nside),0,1))
        cl2_l1=cl2_l1*(1+np.clip((np.arange(cl2.shape[0])-2*nside)/(3*nside),0,1))/(2*np.arange(cl2.shape[0])+1)* \
            (1+np.clip((np.arange(cl2.shape[0])-2.4*nside)/(2.5*nside),0,1))
        return cl2,cl2_l1
