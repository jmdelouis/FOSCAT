import healpy as hp
import numpy as np
import time

class alm():

    def __init__(self,backend=None,lmax=24,
                 nside=None,limit_range=1E10):
        
        if backend is None:
            import foscat.scat_cov as sc
            self.sc=sc.funct()
            self.backend=self.sc.backend
        else:
            self.backend=backend.backend
            
        self._logtab={}
        self.lth={}
        self.lph={}
        self.matrix_shift_ph={}
        self.ratio_mm={}
        self.P_mm={}
        self.A={}
        self.B={}
        if nside is not None:
            self.maxlog=6*nside+1
            self.lmax=3*nside
        else:
            self.lmax=lmax
            self.maxlog=2*lmax+1
        
        for k in range(1,self.maxlog):
            self._logtab[k]=self.backend.bk_log(self.backend.bk_cast(k))
        self._logtab[0]=0.0

        if nside is not None:
            self.ring_th(nside)
            self.ring_ph(nside)
            self.shift_ph(nside)
            
        self._limit_range=1/limit_range
        self._log_limit_range=np.log(limit_range)
        

        self.Yp={}
        self.Ym={}

    def ring_th(self,nside):
        if nside not in self.lth:
            n=0
            ith=[]
            for k in range(nside-1):
                N=4*(k+1)
                ith.append(n)
                n+=N
                
            for k in range(2*nside+1):
                N=4*nside
                ith.append(n)
                n+=N
            for k in range(nside-1):
                N=4*(nside-1-k)
                ith.append(n)
                n+=N
                
            th,ph=hp.pix2ang(nside,ith)
            
            self.lth[nside]=th
        return self.lth[nside]
    
    def ring_ph(self,nside):
        if nside not in self.lph:
            n=0
            iph=[]
            for k in range(nside-1):
                N=4*(k+1)
                iph.append(n)
                n+=N
                
            for k in range(2*nside+1):
                N=4*nside
                iph.append(n)
                n+=N
            for k in range(nside-1):
                N=4*(nside-1-k)
                iph.append(n)
                n+=N
                
            th,ph=hp.pix2ang(nside,iph)
            
            self.lph[nside]=ph
        
    def shift_ph(self,nside):
        
        if nside not in self.matrix_shift_ph:
            self.ring_th(nside)
            self.ring_ph(nside)
            x=(-1J*np.arange(3*nside)).reshape(1,3*nside)
            self.matrix_shift_ph[nside]=self.backend.bk_cast(self.backend.bk_exp(x*self.lph[nside].reshape(4*nside-1,1)))

            self.lmax=3*nside-1
            
            ratio_mm={}
            
            for m in range(3*nside):
                val=np.zeros([self.lmax-m+1])
                aval=np.zeros([self.lmax-m+1])
                bval=np.zeros([self.lmax-m+1])
                
                if m>0:
                    val[0]=self.double_factorial_log(2*m - 1)-0.5*np.sum(np.log(1+np.arange(2*m)))
                else:
                    val[0]=self.double_factorial_log(2*m - 1)
                if m<self.lmax:
                    aval[1]=(2*m + 1)
                    val[1] = val[0]-0.5*self.log(2*m+1)

                    for l in range(m + 2, self.lmax+1):
                        aval[l-m]=(2*l - 1)/ (l - m)
                        bval[l-m]=(l + m - 1)/ (l - m)
                        val[l-m] = val[l-m-1] + 0.5*self.log(l-m) - 0.5*self.log(l+m)
                        
                self.A[nside,m]=self.backend.constant((aval))
                self.B[nside,m]=self.backend.constant((bval))
                self.ratio_mm[nside,m]=self.backend.constant(np.sqrt(4*np.pi)*np.expand_dims(np.exp(val),1))
            # Calcul de P_{mm}(x)
            P_mm=np.ones([3*nside,4*nside-1])
            x=np.cos(self.lth[nside])
            if m == 0:
                P_mm[m] = 1.0
            for m in range(3*nside-1):
                P_mm[m] = (0.5-m%2)*2 * (1 - x**2)**(m/2)
            self.P_mm[nside]=self.backend.constant(P_mm)
        
    def init_Ys(self,s,nside):

        if (s,nside) not in self.Yp:
            import quaternionic
            import spherical

            ell_max = 3*nside-1  # Use the largest ℓ value you expect to need
            wigner = spherical.Wigner(ell_max)

            #th,ph=hp.pix2ang(nside,np.arange(12*nside*nside))

            lth=self.ring_th(nside)

            R = quaternionic.array.from_spherical_coordinates(lth, 0*lth)
            self.Yp[s,nside]     = {}
            self.Ym[s,nside]     = {}
            iplus  = (wigner.sYlm( s, R)*(4*np.pi/(12*nside**2))).T.real
            imoins = (wigner.sYlm(-s, R)*(4*np.pi/(12*nside**2))).T.real
            
            for m in range(ell_max+1):
                idx=np.array([wigner.Yindex(k, m) for k in range(m,ell_max+1)])
                vnorm=1/np.expand_dims(np.sqrt(2*(np.arange(ell_max-m+1)+m)+1),1)
                self.Yp[s,nside][m] = iplus[idx]*vnorm
                self.Ym[s,nside][m] = imoins[idx]*vnorm
                
            del(iplus)
            del(imoins)
            del(wigner)
            
    def log(self,v):
        return np.log(v)
        if isinstance(v,np.ndarray):
            return np.array([self.backend.bk_log(self.backend.bk_cast(k)) for k in v])
        if v<self.maxlog:
            return self._logtab[v]
        else:
            self._logtab[v]=self.backend.bk_log(self.backend.bk_cast(v))
        return self._logtab[v]

    # Fonction pour calculer la double factorielle
    def double_factorial_log(self,n):
        if n <= 0:
            return 0.0
        result = 0.0
        for i in range(n, 0, -2):
            result += np.log(i)
        return result

    def recurrence_fn(self,states, inputs):
        """
        Fonction de récurrence pour tf.scan.
        states: un tuple (U_{n-1}, U_{n-2}) de forme [m]
        inputs: un tuple (a_n(x), b_n) où a_n(x) est de forme [m]
        """
        U_prev, U_prev2 = states
        a_n, b_n = inputs  # a_n est de forme [m], b_n est un scalaire
        U_n = a_n * U_prev - b_n * U_prev2
        return (U_n, U_prev)  # Avancer les états
# Calcul des P_{lm}(x) pour tout l inclus dans [m,lmax]
    def compute_legendre_m(self,x,m,lmax,nside):
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
            return result*np.exp(ratio)*np.sqrt(4*np.pi)
    
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
                
        return result*np.exp(ratio)*np.sqrt(4*np.pi)

    # Calcul des P_{lm}(x) pour tout l inclus dans [m,lmax]
    def compute_legendre_m_old2(self,x,m,lmax,nside):
        
        result={}
        
        # Si l == m, c'est directement P_{mm}
        result[0]  = self.P_mm[nside][m]
    
        if m == lmax:
            v=self.backend.bk_reshape(result[0]*self.ratio_mm[nside,m][0],[1,4*nside-1])
            return self.backend.bk_complex(v,0*v)
    
        # Étape 2 : Calcul de P_{l+1, m}(x)
        result[1] = x * self.A[nside,m][1] * result[0]
    
        # Étape 3 : Récurence pour l > m + 1
        for l in range(m + 2, lmax+1):
            result[l-m]  = self.A[nside,m][l-m] * x * result[l-m-1] - self.B[nside,m][l-m] *  result[l-m-2]
            """
            if np.max(abs(result[l-m]))>self._limit_range:
                result[l-m-1]*=  self._limit_range
                result[l-m]*=    self._limit_range
                ratio[l-m-1]+= self._log_limit_range
                ratio[l-m]+=   self._log_limit_range
            """
        result=self.backend.bk_reshape(self.backend.bk_concat([result[k] for k in range(lmax+1-m)],axis=0),[lmax+1-m,4*nside-1])

        return self.backend.bk_complex(result*self.ratio_mm[nside,m],0*result)

    
    def compute_legendre_m_old(self,x,m,lmax,nside):
        
        import tensorflow as tf
        result={}
        
        # Si l == m, c'est directement P_{mm}
        U_0  = self.P_mm[nside][m]
    
        if m == lmax:
            v=self.backend.bk_reshape(U_0*self.ratio_mm[nside,m][0],[1,4*nside-1])
            return self.backend.bk_complex(v,0*v)
    
        # Étape 2 : Calcul de P_{l+1, m}(x)
        U_1 = x * self.A[nside,m][1] * U_0
        if m == lmax-1:
            result = tf.concat([self.backend.bk_expand_dims(U_0,0),
                                self.backend.bk_expand_dims(U_1,0)],0)
            return self.backend.bk_complex(result*self.ratio_mm[nside,m],0*result)

        a_values = self.backend.bk_expand_dims(self.A[nside,m],1)*self.backend.bk_expand_dims(x,0)
        # Initialiser les états avec (U_1, U_0) pour chaque m
        initial_states = (U_1, U_0)
        inputs = (a_values[2:], self.B[nside,m][2:])
        # Appliquer tf.scan
        result = tf.scan(self.recurrence_fn, inputs, initializer=initial_states)
        # Le premier élément de result contient les U[n]
        result = tf.concat([self.backend.bk_expand_dims(U_0,0),
                            self.backend.bk_expand_dims(U_1,0),
                            result[0]], axis=0)
        """
        # Étape 3 : Récurence pour l > m + 1
        for l in range(m + 2, lmax+1):
            result[l-m]  = self.A[nside,m][l-m] * x * result[l-m-1] - self.B[nside,m][l-m] *  result[l-m-2]
            
            if np.max(abs(result[l-m]))>self._limit_range:
                result[l-m-1]*=  self._limit_range
                result[l-m]*=    self._limit_range
                ratio[l-m-1]+= self._log_limit_range
                ratio[l-m]+=   self._log_limit_range
        result=self.backend.bk_reshape(self.backend.bk_concat([result[k] for k in range(lmax+1-m)],axis=0),[lmax+1-m,4*nside-1])
         """
        
        return self.backend.bk_complex(result*self.ratio_mm[nside,m],0*result)

    
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

    def rfft2fft(self,val,axis=0):
        r=self.backend.bk_rfft(val)
        if axis==0:
            r_inv=self.backend.bk_reverse(self.backend.bk_conjugate(r[1:-1]),axis=axis)
        else:
            r_inv=self.backend.bk_reverse(self.backend.bk_conjugate(r[:,1:-1]),axis=axis)
        return self.backend.bk_concat([r,r_inv],axis=axis)
    
    def irfft2fft(self,val,N,axis=0):
        if axis==0:
            return self.backend.bk_irfft(val[0:N//2+1])
        else:
            return self.backend.bk_irfft(val[:,0:N//2+1])
    
    def comp_tf(self,im,nside,realfft=False):
        
        self.shift_ph(nside)
        n=0
        
        ft_im=[]
        for k in range(nside-1):
            N=4*(k+1)
            
            if realfft:
                tmp=self.rfft2fft(im[n:n+N])
            else:
                tmp=self.backend.bk_fft(im[n:n+N])

            l_n=tmp.shape[0]
            
            if l_n<3*nside+1:
                repeat_n=3*nside//l_n+1
                tmp=self.backend.bk_tile(tmp,repeat_n,axis=0)
                
            ft_im.append(tmp[0:3*nside])
            
            n+=N
        if nside>1:
            result=self.backend.bk_reshape(self.backend.bk_concat(ft_im,axis=0),[nside-1,3*nside])
        
        N=4*nside*(2*nside+1)    
        v=self.backend.bk_reshape(im[n:n+N],[2*nside+1,4*nside])
        if realfft:
            v_fft=self.rfft2fft(v,axis=1)[:,:3*nside]
        else:
            v_fft=self.backend.bk_fft(v)[:,:3*nside]

        n+=N
        if nside>1:
            result=self.backend.bk_concat([result,v_fft],axis=0)
        else:
            result=v_fft
        
        if nside>1:
            ft_im=[]
            for k in range(nside-1):
                N=4*(nside-1-k)

                if realfft:
                    tmp=self.rfft2fft(im[n:n+N])[0:l_n]
                else:
                    tmp=self.backend.bk_fft(im[n:n+N])[0:l_n]

                l_n=tmp.shape[0]

                if l_n<3*nside+1:
                    repeat_n=3*nside//l_n+1
                    tmp=self.backend.bk_tile(tmp,repeat_n,axis=0)

                ft_im.append(tmp[0:3*nside])
                n+=N

            lastresult=self.backend.bk_reshape(self.backend.bk_concat(ft_im,axis=0),[nside-1,3*nside])
            return self.backend.bk_concat([result,lastresult],axis=0)*self.matrix_shift_ph[nside]
        else:
            return result*self.matrix_shift_ph[nside]

    
    def icomp_tf(self,i_im,nside,realfft=False):
        
        self.shift_ph(nside)
        
        n=0
        im=[]
        ft_im=i_im*self.backend.bk_conjugate(self.matrix_shift_ph[nside])
        
        for k in range(nside-1):
            N=4*(k+1)
            
            if realfft:
                tmp=self.irfft2fft(ft_im[k],N)
            else:
                tmp=self.backend.bk_ifft(im[k],N)
                
            im.append(tmp[0:N])
            
            n+=N
            
        if nside>1:
            result=self.backend.bk_concat(im,axis=0)
        
        N=4*nside*(2*nside+1)    
        v=ft_im[nside-1:3*nside,0:2*nside+1]
        if realfft:
            v_fft=self.backend.bk_reshape(self.irfft2fft(v,N,axis=1),[4*nside*(2*nside+1)])
        else:
            v_fft=self.backend.bk_ifft(v)

        n+=N
        if nside>1:
            result=self.backend.bk_concat([result,v_fft],axis=0)
        else:
            result=v_fft
        
        if nside>1:
            im=[]
            for k in range(nside-1):
                N=4*(nside-1-k)
            
                if realfft:
                    tmp=self.irfft2fft(ft_im[k+3*nside],N)
                else:
                    tmp=self.backend.bk_ifft(im[k+3*nside],N)
                
                im.append(tmp[0:N])
            
                n+=N

            return self.backend.bk_concat([result]+im,axis=0)
        else:
            return result
        
    def anafast(self,im,map2=None,nest=False,spin=2):
        
        """The `anafast` function computes the L1 and L2 norm power spectra. 

        Currently, it is not optimized for single-pass computation due to the relatively inefficient computation of \(Y_{lm}\). 
        Nonetheless, it utilizes TensorFlow and can be integrated into gradient computations.

        Input:
        - `im`: a vector of size \([12 \times \text{Nside}^2]\) for scalar data, or of size \([2, 12 \times \text{Nside}^2]\) for Q,U polar data, 
        or of size \([3, 12 \times \text{Nside}^2]\) for I,Q,U polar data.
        - `map2` (optional): a vector of size \([12 \times \text{Nside}^2]\) for scalar data, or of size 
        \([3, 12 \times \text{Nside}^2]\) for polar data. If provided, cross power spectra will be computed.
        - `nest=True`: alters the ordering of the input maps.
        - `spin=2` for 1/2 spin data as Q and U. Spin=1 for seep fields

        Output:
        -A tensor of size \([l_{\text{max}} \times (l_{\text{max}}-1)]\) formatted as \([6, \ldots]\), 
        ordered as TT, EE, BB, TE, EB.TBanafast function computes L1 and L2 norm powerspctra.

        """
        i_im=self.backend.bk_cast(im)
        if map2 is not None:
            i_map2=self.backend.bk_cast(map2)
            
        doT=True
        if len(i_im.shape)==1: # nopol
            nside=int(np.sqrt(i_im.shape[0]//12))
        else:
            if i_im.shape[0]==2:
                doT=False
            nside=int(np.sqrt(i_im.shape[1]//12))
            
        self.shift_ph(nside)
    
        if doT: # nopol
            if len(i_im.shape)==2: # pol
                l_im=i_im[0]
                if map2 is not None:
                    l_map2=i_map2[0]
            else:
                l_im=i_im
                if map2 is not None:
                    l_map2=i_map2
                    
            if nest:
                idx=hp.ring2nest(nside,np.arange(12*nside**2))
                if len(i_im.shape)==1: # nopol
                    ft_im=self.comp_tf(self.backend.bk_gather(l_im,idx),nside,realfft=True)
                    if map2 is not None:
                        ft_im2=self.comp_tf(self.backend.bk_gather(l_map2,idx),nside,realfft=True)
            else:
                ft_im=self.comp_tf(l_im,nside,realfft=True)
                if map2 is not None:
                    ft_im2=self.comp_tf(l_map2,nside,realfft=True)

        lth=self.ring_th(nside)

        co_th=np.cos(lth)
        
        lmax=3*nside-1
        
        cl2=None
        cl2_L1=None
        dt2=0
        dt3=0
        dt4=0
        if len(i_im.shape)==2: # nopol
        
            self.init_Ys(spin,nside)
            
            if nest:
                idx=hp.ring2nest(nside,np.arange(12*nside**2))
                l_Q=self.backend.bk_gather(i_im[int(doT)],idx)
                l_U=self.backend.bk_gather(i_im[1+int(doT)],idx)
                ft_im_Pp=self.comp_tf(self.backend.bk_complex(l_Q,l_U),nside)
                ft_im_Pm=self.comp_tf(self.backend.bk_complex(l_Q,-l_U),nside)
                if map2 is not None:
                    l_Q=self.backend.bk_gather(i_map2[int(doT)],idx)
                    l_U=self.backend.bk_gather(i_map2[1+int(doT)],idx)
                    ft_im2_Pp=self.comp_tf(self.backend.bk_complex(l_Q,l_U),nside)
                    ft_im2_Pm=self.comp_tf(self.backend.bk_complex(l_Q,-l_U),nside)
            else:
                ft_im_Pp=self.comp_tf(self.backend.bk_complex(i_im[int(doT)],i_im[1+int(doT)]),nside)
                ft_im_Pm=self.comp_tf(self.backend.bk_complex(i_im[int(doT)],-i_im[1+int(doT)]),nside)
                if map2 is not None:
                    ft_im2_Pp=self.comp_tf(self.backend.bk_complex(i_map2[int(doT)],i_map2[1+int(doT)]),nside)
                    ft_im2_Pm=self.comp_tf(self.backend.bk_complex(i_map2[int(doT)],-i_map2[1+int(doT)]),nside)

        for m in range(lmax+1):

            plm=self.compute_legendre_m(co_th,m,3*nside-1,nside)/(12*nside**2)
            
            if doT:
                tmp=self.backend.bk_reduce_sum(plm*ft_im[:,m],1)

                if map2 is not None:
                    tmp2=self.backend.bk_reduce_sum(plm*ft_im2[:,m],1)
                else:
                    tmp2=tmp
                    
            if len(i_im.shape)==2: # pol
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
                    
                if doT:
                    tmpTT=self.backend.bk_real((tmp*self.backend.bk_conjugate(tmp2)))
                    tmpTE=self.backend.bk_real((tmp*self.backend.bk_conjugate(almE2)))
                    tmpTB=-self.backend.bk_real((tmp*self.backend.bk_conjugate(almB2)))
                
                tmpEE=self.backend.bk_real((almE*self.backend.bk_conjugate(almE2)))
                tmpBB=self.backend.bk_real((almB*self.backend.bk_conjugate(almB2)))
                tmpEB=-self.backend.bk_real((almE*self.backend.bk_conjugate(almB2)))
                
                if map2 is not None:
                    tmpEB=(tmpEB-self.backend.bk_real((almE2*self.backend.bk_conjugate(almB))))/2
                    
                    if doT:
                        tmpTE=(tmpTE+self.backend.bk_real((tmp2*self.backend.bk_conjugate(almE))))/2
                        tmpTB=(tmpTB-self.backend.bk_real((tmp2*self.backend.bk_conjugate(almB))))/2
                

                if m==0:
                    if doT:
                        l_cl=self.backend.bk_concat([tmpTT,tmpEE,tmpBB,tmpTE,tmpEB,tmpTB],0)
                    else:
                        l_cl=self.backend.bk_concat([tmpEE,tmpBB,tmpEB],0)
                else:
                    offset_tensor=self.backend.bk_zeros((m),dtype=self.backend.all_bk_type)
                    if doT:
                        l_cl=self.backend.bk_concat([self.backend.bk_concat([offset_tensor,tmpTT],axis=0),
                                                     self.backend.bk_concat([offset_tensor,tmpEE],axis=0),
                                                     self.backend.bk_concat([offset_tensor,tmpBB],axis=0),
                                                     self.backend.bk_concat([offset_tensor,tmpTE],axis=0),
                                                     self.backend.bk_concat([offset_tensor,tmpEB],axis=0),
                                                     self.backend.bk_concat([offset_tensor,tmpTB],axis=0)],axis=0)
                    else:
                        l_cl=self.backend.bk_concat([self.backend.bk_concat([offset_tensor,tmpEE],axis=0),
                                                     self.backend.bk_concat([offset_tensor,tmpBB],axis=0),
                                                     self.backend.bk_concat([offset_tensor,tmpEB],axis=0)],axis=0)
                    
                if doT:
                    l_cl=self.backend.bk_reshape(l_cl,[6,lmax+1])
                else:
                    l_cl=self.backend.bk_reshape(l_cl,[3,lmax+1])
            else:
                tmp=self.backend.bk_real((tmp*self.backend.bk_conjugate(tmp2)))
                if m==0:
                    l_cl=tmp
                else:
                    offset_tensor=self.backend.bk_zeros((m),dtype=self.backend.all_bk_type)
                    l_cl=self.backend.bk_concat([offset_tensor,tmp],axis=0)
                    
            if cl2 is None:
                cl2=l_cl
            else:
                cl2+=2*l_cl
        
        #cl2=cl2*(4*np.pi) #self.backend.bk_sqrt(self.backend.bk_cast(4*np.pi)) #(2*np.arange(cl2.shape[0])+1)))
                                         
        cl2_l1=self.backend.bk_L1(cl2)
                                         
        return cl2,cl2_l1

    def map2alm(self,im,nest=False):
        nside=int(np.sqrt(im.shape[0]//12))
        
        ph=self.shift_ph(nside)
        
        if nest:
            idx=hp.ring2nest(nside,np.arange(12*nside**2))
            ft_im=self.comp_tf(self.backend.bk_cast(self.backend.bk_gather(im,idx)),nside,realfft=True)
        else:
            ft_im=self.comp_tf(self.backend.bk_cast(im),nside,realfft=True)
            
        lth=self.ring_th(nside)

        co_th=np.cos(lth)

        lmax=3*nside-1
    
        alm=None
        for m in range(lmax+1):
            plm=self.compute_legendre_m(co_th,m,3*nside-1,nside)/(12*nside**2)
            
            tmp=self.backend.bk_reduce_sum(plm*ft_im[:,m],1)
            if m==0:
                alm=tmp
            else:
                alm=self.backend.bk_concat([alm,tmp],axis=0)
            
        return alm

    
    def alm2map(self,nside,alm):

        lth=self.ring_th(nside)

        co_th=np.cos(lth)
        
        ft_im=[]

        n=0

        lmax=3*nside-1
        
        for m in range(lmax+1):
            plm=self.compute_legendre_m(co_th,m,3*nside-1,nside)/(12*nside**2)

            print(alm[n:n+lmax-m+1].shape,plm.shape)
            ft_im.append(self.backend.bk_reduce_sum(self.backend.bk_reshape(alm[n:n+lmax-m+1],[lmax-m+1,1])*plm,0))

            n=n+lmax-m+1
            
        return self.backend.bk_reshape(self.backend.bk_concat(ft_im,0),[lmax+1,4*nside-1])
        
            
        if nest:
            idx=hp.ring2nest(nside,np.arange(12*nside**2))
            ft_im=self.comp_tf(self.backend.bk_cast(self.backend.bk_gather(im,idx)),nside,realfft=True)
        else:
            ft_im=self.comp_tf(self.backend.bk_cast(im),nside,realfft=True)
            

        lmax=3*nside-1
    
        alm=None
        for m in range(lmax+1):
            plm=self.compute_legendre_m(co_th,m,3*nside-1,nside)/(12*nside**2)
            
            tmp=self.backend.bk_reduce_sum(plm*ft_im[:,m],1)
            if m==0:
                alm=tmp
            else:
                alm=self.backend.bk_concat([alm,tmp],axis=0)
            
        return o_map

    def map2alm_spin(self,im_Q,im_U,spin=2,nest=False):
        
        if spin==0:
            return self.map2alm(im_Q,nest=nest),self.map2alm(im_U,nest=nest)

        nside=int(np.sqrt(im_Q.shape[0]//12))

        lth=self.ring_th(nside)
        
        co_th=np.cos(lth)
        
        if nest:
            idx=hp.ring2nest(nside,np.arange(12*nside**2))
            l_Q=self.backend.bk_gather(im_Q,idx)
            l_U=self.backend.bk_gather(im_U,idx)
            ft_im_1=self.comp_tf(self.backend.bk_complex(l_Q,l_U),nside)
            ft_im_2=self.comp_tf(self.backend.bk_complex(l_Q,-l_U),nside)
        else:
            ft_im_1=self.comp_tf(self.backend.bk_complex(im_Q,im_U),nside)
            ft_im_2=self.comp_tf(self.backend.bk_complex(im_Q,-im_U),nside)

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
