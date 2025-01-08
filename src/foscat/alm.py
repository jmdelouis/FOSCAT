import healpy as hp
import numpy as np

class alm():

    def __init__(self,backend=None,lmax=24,nside=None,limit_range=1E10):
        self._logtab={}
        self.lth={}
        if nside is not None:
            self.lmax=3*nside
            th,ph=hp.pix2ang(nside,np.arange(12*nside*nside))

            lth=np.unique(th)
            
            self.lth[nside]=lth
        else:
            self.lmax=lmax

        for k in range(1,2*self.lmax+1):
            self._logtab[k]=np.log(k)
        self._logtab[0]=0.0
        self._limit_range=1/limit_range
        self._log_limit_range=np.log(limit_range)
        
        if backend is None:
            import foscat.scat_cov as sc
            self.sc=sc.funct()
            self.backend=self.sc.backend
        else:
            self.backend=backend.backend

        self.Yp={}
        self.Ym={}

    def ring_th(self,nside):
        if nside not in self.lth:
            th,ph=hp.pix2ang(nside,np.arange(12*nside*nside))

            lth=np.unique(th)
        
            self.lth[nside]=lth
        return self.lth[nside]
        

    def init_Ys(self,s,nside):

        if (s,nside) not in self.Yp:
            import quaternionic
            import spherical

            ell_max = 3*nside-1  # Use the largest ℓ value you expect to need
            wigner = spherical.Wigner(ell_max)

            th,ph=hp.pix2ang(nside,np.arange(12*nside*nside))

            lth=self.ring_th(nside)

            R = quaternionic.array.from_spherical_coordinates(lth, 0*lth)
            self.Yp[s,nside]     = {}
            self.Ym[s,nside]     = {}
            iplus  = (wigner.sYlm( s, R)*(4*np.pi/(12*nside**2))).T.real
            imoins = (wigner.sYlm(-s, R)*(4*np.pi/(12*nside**2))).T.real
            
            for m in range(ell_max+1):
                idx=np.array([wigner.Yindex(k, m) for k in range(m,ell_max+1)])
                self.Yp[s,nside][m] = iplus[idx]
                self.Ym[s,nside][m] = imoins[idx]
                
            del(iplus)
            del(imoins)
            del(wigner)
            
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
        result=np.zeros([lmax-m+1,x.shape[0]])
        ratio=np.zeros([lmax-m+1,1])
        
        ratio[0,0] = self.double_factorial_log(2*m - 1)-0.5*np.sum(self.log(1+np.arange(2*m)))
        
        # Étape 1 : Calcul de P_{mm}(x)
        if m == 0:
            Pmm = 1.0
        else:
            #Pmm = (-1)**m * (1 - x**2)**(m/2)
            Pmm = (0.5-m%2)*2 * (1 - x**2)**(m/2)
        
        
        # Si l == m, c'est directement P_{mm}
        result[0]  = Pmm
    
        if m == lmax:
            return result*np.exp(ratio)*np.sqrt(4*np.pi*(2*(np.arange(lmax-m+1)+m)+1)).reshape(lmax+1-m,1)
    
        # Étape 2 : Calcul de P_{l+1, m}(x)
        result[1] = x * (2*m + 1) * result[0]

        ratio[1,0] = ratio[0,0]-0.5*self.log(2*m+1)
    
        # Étape 3 : Récurence pour l > m + 1
        for l in range(m + 2, lmax+1):
            result[l-m]  = ((2*l - 1) * x * result[l-m-1] - (l + m - 1) *  result[l-m-2]) / (l - m)
            ratio[l-m,0] = 0.5*self.log(l-m)-0.5*self.log(l+m)+ratio[l-m-1,0]
            if np.max(abs(result[l-m]))>self._limit_range:
                result[l-m-1]*=  self._limit_range
                result[l-m]*=    self._limit_range
                ratio[l-m-1,0]+= self._log_limit_range
                ratio[l-m,0]+=   self._log_limit_range
                
        return result*np.exp(ratio)*(np.sqrt(4*np.pi*(2*(np.arange(lmax-m+1)+m)+1))).reshape(lmax+1-m,1)

    
    # Calcul des s_P_{lm}(x) pour tout l inclus dans [m,lmax]
    def compute_legendre_spin2_m(self,co_th,si_th,m,lmax):
        result=np.zeros([lmax-m+2,co_th.shape[0]])
        ratio =np.zeros([lmax-m+2,1])
        
        ratio[1,0] = self.double_factorial_log(2*m - 1)-0.5*np.sum(self.log(1+np.arange(2*m)))
        # Étape 1 : Calcul de P_{mm}(x)
        if m == 0:
            Pmm = 1.0
        else:
            #Pmm = (-1)**m * (1 - x**2)**(m/2)
            Pmm = (0.5-m%2)*2 * (1 - co_th**2)**(m/2)
        
        
        # Si l == m, c'est directement P_{mm}
        result[1]  = Pmm
    
        if m == lmax:
            ylm=result*np.exp(ratio)
            ylm[1:]*=(np.sqrt(4*np.pi*(2*(np.arange(lmax-m+1)+m)+1))).reshape(lmax+1-m,1)

        else:
            # Étape 2 : Calcul de P_{l+1, m}(x)
            result[2] = co_th * (2*m + 1) * result[0]

            ratio[2,0] = ratio[1,0]-self.log(2*m+1)/2
    
            # Étape 3 : Récurence pour l > m + 1
            for l in range(m + 2, lmax+1):
                result[l-m+1]  = ((2*l - 1) * co_th * result[l-m] - (l + m - 1) *  result[l-m-1]) / (l - m)
                ratio[l-m+1,0] = (self.log(l-m)-self.log(l+m))/2+ratio[l-m,0]
                if np.max(abs(result[l-m+1]))>self._limit_range:
                    result[l-m]*=  self._limit_range
                    result[l-m+1]*=   self._limit_range
                    ratio[l-m,0]+= self._log_limit_range
                    ratio[l-m+1,0]+=   self._log_limit_range

            ylm=result*np.exp(ratio)
            ylm[1:]*=(np.sqrt(4*np.pi*(2*(np.arange(lmax-m+1)+m)+1))).reshape(lmax+1-m,1)

        ell=(np.arange(lmax+1-m)+m).reshape(lmax+1-m,1)
        
        cot_th=co_th/si_th
        si2_th=si_th*si_th

        a = (2*m**2-ell*(ell+1))/(si2_th.reshape(1,si2_th.shape[0]))+ell*(ell-1)*cot_th*cot_th
        b = 2*m*(ell-1)*cot_th/si_th
        w=np.zeros([lmax+1-m,1])
        l=ell[ell>1]
        w[ell>1]=np.sqrt(1/((l+2)*(l+1)*(l)*(l-1)))
        w=w.reshape(lmax+1-m,1)
        
        alpha_plus=w*(a+b)
        alpha_moins=w*(a-b)

        a=2*np.sqrt((2*ell+1)/(2*ell-1)*(ell*ell-m*m))
        b=m/si2_th
        
        beta_plus=w*a*(cot_th/si_th+b)
        beta_moins=w*a*(cot_th/si_th-b)

        ylm_plus  = alpha_plus*ylm[1:]+ beta_plus*ylm[:-1]  
        ylm_moins = alpha_moins*ylm[1:] + beta_moins*ylm[:-1]
                
        return ylm_plus,ylm_moins
    
    def comp_tf(self,im,ph):
        nside=int(np.sqrt(im.shape[0]//12))
        n=0
        ii=0
        ft_im=[]
        for k in range(nside-1):
            N=4*(k+1)
            l_n=N
            if l_n>3*nside:
                l_n=3*nside
            tmp=self.backend.bk_fft(im[n:n+N])[0:l_n]
            ft_im.append(tmp*np.exp(-1J*np.arange(l_n)*ph[n]))
            ft_im.append(self.backend.bk_zeros((3*nside-l_n),dtype=self.backend.all_cbk_type))
            # if N<3*nside fill the tf with rotational values to mimic alm_tools.F90 of healpix (Minor effect)
            #for m in range(l_n,3*nside,l_n):
            #    ft_im.append(tmp[0:np.min([3*nside-m,l_n])])
            n+=N
            ii+=1
        for k in range(2*nside+1):
            N=4*nside
            ft_im.append(self.backend.bk_fft(im[n:n+N])[:3*nside]*np.exp(-1J*np.arange(3*nside)*ph[n]))
            n+=N
            ii+=1
        for k in range(nside-1):
            N=4*(nside-1-k)
            l_n=N
            if l_n>3*nside:
                l_n=3*nside
            tmp=self.backend.bk_fft(im[n:n+N])[0:l_n]
            ft_im.append(tmp*np.exp(-1J*np.arange(l_n)*ph[n]))
            ft_im.append(self.backend.bk_zeros((3*nside-l_n),dtype=self.backend.all_cbk_type))
            # if N<3*nside fill the tf with rotational values to mimic alm_tools.F90 of healpix (Minor effect)
            #for m in range(l_n,3*nside,l_n):
            #    ft_im.append(tmp[0:np.min([3*nside-m,l_n])])
            n+=N
            ii+=1
        return self.backend.bk_reshape(self.backend.bk_concat(ft_im,axis=0),[4*nside-1,3*nside])

    def anafast(self,im,map2=None,nest=False):
        """The `anafast` function computes the L1 and L2 norm power spectra. 

        Currently, it is not optimized for single-pass computation due to the relatively inefficient computation of \(Y_{lm}\). 
        Nonetheless, it utilizes TensorFlow and can be integrated into gradient computations.

        Input:
        - `im`: a vector of size \([12 \times \text{Nside}^2]\) for scalar data, or of size \([3, 12 \times \text{Nside}^2]\) for polar data.
        - `map2` (optional): a vector of size \([12 \times \text{Nside}^2]\) for scalar data, or of size 
        \([3, 12 \times \text{Nside}^2]\) for polar data. If provided, cross power spectra will be computed.
        - `nest=True`: alters the ordering of the input maps.

        Output:
        -A tensor of size \([l_{\text{max}} \times (l_{\text{max}}-1)]\) formatted as \([6, \ldots]\), 
        ordered as TT, EE, BB, TE, EB.TBanafast function computes L1 and L2 norm powerspctra.

        """
        if len(im.shape)==1: # nopol
            nside=int(np.sqrt(im.shape[0]//12))
        else:
            nside=int(np.sqrt(im.shape[1]//12))
        th,ph=hp.pix2ang(nside,np.arange(12*nside*nside))
        if nest:
            idx=hp.ring2nest(nside,np.arange(12*nside**2))
            if len(im.shape)==1: # nopol
                ft_im=self.comp_tf(self.backend.bk_complex(self.backend.bk_gather(im,idx),0*im),ph)
                if map2 is not None:
                    ft_im2=self.comp_tf(self.backend.bk_complex(self.backend.bk_gather(map2,idx),0*im),ph)
            else:
                ft_im=self.comp_tf(self.backend.bk_complex(self.backend.bk_gather(im[0],idx),0*im[0]),ph)
                if map2 is not None:
                    ft_im2=self.comp_tf(self.backend.bk_complex(self.backend.bk_gather(map2[0],idx),0*im[0]),ph)
        else:
            if len(im.shape)==1: # nopol
                ft_im=self.comp_tf(self.backend.bk_complex(im,0*im),ph)
                if map2 is not None:
                    ft_im2=self.comp_tf(self.backend.bk_complex(map2,0*im),ph)
            else:
                ft_im=self.comp_tf(self.backend.bk_complex(im[0],0*im[0]),ph)
                if map2 is not None:
                    ft_im2=self.comp_tf(self.backend.bk_complex(map2[0],0*im[0]),ph)
                    
        lth=self.ring_th(nside)

        co_th=np.cos(lth)
        
        lmax=3*nside-1
        
        cl2=None
        cl2_L1=None

        
        if len(im.shape)==2: # nopol
            
            spin=2
        
            self.init_Ys(spin,nside)
            
            if nest:
                idx=hp.ring2nest(nside,np.arange(12*nside**2))
                l_Q=self.backend.bk_gather(im[1],idx)
                l_U=self.backend.bk_gather(im[2],idx)
                ft_im_Pp=self.comp_tf(self.backend.bk_complex(l_Q,l_U),ph)
                ft_im_Pm=self.comp_tf(self.backend.bk_complex(l_Q,-l_U),ph)
                if map2 is not None:
                    l_Q=self.backend.bk_gather(map2[1],idx)
                    l_U=self.backend.bk_gather(map2[2],idx)
                    ft_im2_Pp=self.comp_tf(self.backend.bk_complex(l_Q,l_U),ph)
                    ft_im2_Pm=self.comp_tf(self.backend.bk_complex(l_Q,-l_U),ph)
            else:
                ft_im_Pp=self.comp_tf(self.backend.bk_complex(im[1],im[2]),ph)
                ft_im_Pm=self.comp_tf(self.backend.bk_complex(im[1],-im[2]),ph)
                if map2 is not None:
                    ft_im2_Pp=self.comp_tf(self.backend.bk_complex(map2[1],map2[2]),ph)
                    ft_im2_Pm=self.comp_tf(self.backend.bk_complex(map2[1],-map2[2]),ph)

        for m in range(lmax+1):

            plm=self.compute_legendre_m(co_th,m,3*nside-1)/(12*nside**2)
            
            tmp=self.backend.bk_reduce_sum(plm*ft_im[:,m],1)
            
            if map2 is not None:
                tmp2=self.backend.bk_reduce_sum(plm*ft_im2[:,m],1)
            else:
                tmp2=tmp
                
            if len(im.shape)==2: # pol
                plmp=self.Yp[spin,nside][m]
                plmm=self.Ym[spin,nside][m]
            
                tmpp=self.backend.bk_reduce_sum(plmp*ft_im_Pp[:,m],1)
                tmpm=self.backend.bk_reduce_sum(plmm*ft_im_Pm[:,m],1)
                
                almE=-(tmpp+tmpm)/2.0
                almB=(tmpp-tmpm)/(2J)
                
                if map2 is not None:
                    tmpp2=self.backend.bk_reduce_sum(plmp*ft_im2_Pp[:,m],1)
                    tmpm2=self.backend.bk_reduce_sum(plmm*ft_im2_Pm[:,m],1)
                
                    almE2=-(tmpp2+tmpm2)/2.0
                    almB2=(tmpp2-tmpm2)/(2J)
                else:
                    almE2=almE
                    almB2=almB

                tmpTT=self.backend.bk_real((tmp*self.backend.bk_conjugate(tmp2)))
                tmpEE=self.backend.bk_real((almE*self.backend.bk_conjugate(almE2)))
                tmpBB=self.backend.bk_real((almB*self.backend.bk_conjugate(almB2)))
                tmpTE=self.backend.bk_real((tmp*self.backend.bk_conjugate(almE2)))
                tmpTB=-self.backend.bk_real((tmp*self.backend.bk_conjugate(almB2)))
                tmpEB=-self.backend.bk_real((almE*self.backend.bk_conjugate(almB2)))
                
                if map2 is not None:
                    tmpTE=(tmpTE+self.backend.bk_real((tmp2*self.backend.bk_conjugate(almE))))/2
                    tmpTB=(tmpTB-self.backend.bk_real((tmp2*self.backend.bk_conjugate(almB))))/2
                    tmpEB=(tmpEB-self.backend.bk_real((almE2*self.backend.bk_conjugate(almB))))/2
                

                if m==0:
                    l_cl=self.backend.bk_concat([tmpTT,tmpEE,tmpBB,tmpTE,tmpEB,tmpTB],0)
                else:
                    offset_tensor=self.backend.bk_zeros((m),dtype=self.backend.all_bk_type)
                    l_cl=self.backend.bk_concat([self.backend.bk_concat([offset_tensor,tmpTT],axis=0),
                                                 self.backend.bk_concat([offset_tensor,tmpEE],axis=0),
                                                 self.backend.bk_concat([offset_tensor,tmpBB],axis=0),
                                                 self.backend.bk_concat([offset_tensor,tmpTE],axis=0),
                                                 self.backend.bk_concat([offset_tensor,tmpEB],axis=0),
                                                 self.backend.bk_concat([offset_tensor,tmpTB],axis=0)],axis=0)
                    
                l_cl=self.backend.bk_reshape(l_cl,[6,lmax+1])
            else:
                tmp=self.backend.bk_real((tmp*self.backend.bk_conjugate(tmp2)))
                if m==0:
                    l_cl=tmp
                else:
                    offset_tensor=self.backend.bk_zeros((m),dtype=self.backend.all_bk_type)
                    l_cl=self.backend.bk_concat([offset_tensor,tmp],axis=0)
                    
            if cl2 is None:
                cl2=l_cl
                cl2_l1=self.backend.bk_L1(l_cl)
            else:
                cl2+=2*l_cl
                cl2_l1+=2*self.backend.bk_L1(l_cl)
                
        if len(im.shape)==1: # nopol
            cl2=cl2/(2*np.arange(cl2.shape[0])+1)
            cl2_l1=cl2_l1/(2*np.arange(cl2.shape[0])+1)
        else:
            cl2=cl2/np.expand_dims(2*np.arange(cl2.shape[1])+1,0)
            cl2_l1=cl2_l1/np.expand_dims(2*np.arange(cl2.shape[1])+1,0)
        return cl2,cl2_l1

    def map2alm(self,im,nest=False):
        nside=int(np.sqrt(im.shape[0]//12))
        th,ph=hp.pix2ang(nside,np.arange(12*nside*nside))
        if nest:
            idx=hp.ring2nest(nside,np.arange(12*nside**2))
            ft_im=self.comp_tf(self.backend.bk_complex(self.backend.bk_gather(im,idx),0*im),ph)
        else:
            ft_im=self.comp_tf(self.backend.bk_complex(im,0*im),ph)
            
        co_th=np.cos(self.ring_th(nside))

        lmax=3*nside-1
    
        alm=None
        for m in range(lmax+1):
            plm=self.compute_legendre_m(co_th,m,3*nside-1)/(12*nside**2)
            
            tmp=self.backend.bk_reduce_sum(plm*ft_im[:,m],1)
            if m==0:
                alm=tmp
            else:
                alm=self.backend.bk_concat([alm,tmp],axis=0)
            
        return alm

    def map2alm_spin(self,im_Q,im_U,spin=2,nest=False):
        
        if spin==0:
            return self.map2alm(im_Q,nest=nest),self.map2alm(im_U,nest=nest)

        
        nside=int(np.sqrt(im_Q.shape[0]//12))
        th,ph=hp.pix2ang(nside,np.arange(12*nside*nside))
        
        self.init_Ys(spin,nside)
        
        if nest:
            idx=hp.ring2nest(nside,np.arange(12*nside**2))
            l_Q=self.backend.bk_gather(im_Q,idx)
            l_U=self.backend.bk_gather(im_U,idx)
            ft_im_1=self.comp_tf(self.backend.bk_complex(l_Q,l_U),ph)
            ft_im_2=self.comp_tf(self.backend.bk_complex(l_Q,-l_U),ph)
        else:
            ft_im_1=self.comp_tf(self.backend.bk_complex(im_Q,im_U),ph)
            ft_im_2=self.comp_tf(self.backend.bk_complex(im_Q,-im_U),ph)

        #co_th=np.cos(self.ring_th[nside])
        #si_th=np.sin(self.ring_th[nside])

        lmax=3*nside-1
    
        alm=None
        for m in range(lmax+1):
            #not yet debug use spherical
            #plmp1,plmm1=self.compute_legendre_spin2_m(co_th,si_th,m,3*nside-1)
            #plmp1/=(12*nside**2)
            #plmm1/=(12*nside**2)
            
            plmp=self.Yp[spin,nside][m]
            plmm=self.Ym[spin,nside][m]
            
            tmpp=self.backend.bk_reduce_sum(plmp*ft_im_1[:,m],1)
            tmpm=self.backend.bk_reduce_sum(plmm*ft_im_2[:,m],1)
            if m==0:
                almE=-(tmpp+tmpm)/2.0
                almB=(tmpp-tmpm)/(2J)
            else:
                almE=self.backend.bk_concat([almE,-(tmpp+tmpm)/2],axis=0)
                almB=self.backend.bk_concat([almB,(tmpp-tmpm)/(2J)],axis=0)
            
        return almE,almB
