from scipy.interpolate import interp1d
import foscat.CircSpline as sc
import foscat.Spline1D as sc1d
import numpy as np
import healpy as hp

class heal_spline:
    def __init__(
            self,
            level,
            gamma=1,
    ):
        nside=2**level
        self.nside_store=2**(level//2)
        self.spline_tree={}
        self.gamma=gamma
        
        self.nside=nside
        #compute colatitude
        idx_th=np.zeros([4*nside],dtype='int')
        n=0
        d=0
        for k in range(nside):
            d+=4
            idx_th[k]=n
            n+=d
            
        for k in range(2*nside-1):
            idx_th[k+nside]=n
            n+=d
                
        for k in range(nside):
            idx_th[k+3*nside-1]=n
            n+=d
            d-=4
        idx_th[-1]=12*nside**2
    
        th0_val,ph0_val=hp.pix2ang(self.nside,idx_th[:-1])
        self.th0_val=th0_val    
        self.ph0_val=ph0_val    
        
        self.idx_th=idx_th
        
        #init spline table

        self.spline_lat=sc1d.Spline1D(4*self.nside-1,3)

        #convert colatitude in ring index
        self.f_interp_th = interp1d(np.concatenate([[0],(th0_val[:-1]+th0_val[1:])/2,[np.pi]],0),
                                    np.arange(4*self.nside)/(4*self.nside), 
                                   kind='cubic', fill_value='extrapolate')    

    
    def ang2weigths(self,th,ph,threshold=1E-2,nest=False): 
        th0=self.f_interp_th(th).flatten()
        
        idx_lat,w_th=self.spline_lat.eval(th0.flatten())
            
        www     = np.zeros([4,4,th0.shape[0]])
        all_idx = np.zeros([4,4,th0.shape[0]],dtype='int')
            
        iring_tab=np.unique(idx_lat)
        for iring in iring_tab:
            spline_table=sc.CircSpline(self.idx_th[iring+1]-self.idx_th[iring],3)
            for k in range(4):
                iii=np.where(idx_lat[k]==iring)[0]
                idx,w=spline_table.eval((ph[iii]-self.ph0_val[iring])/(2*np.pi))
                idx=idx+self.idx_th[iring]
                for m in range(4):
                    www[k,m,iii]=w[m]*w_th[k,iii]
                    all_idx[k,m,iii]=idx[m]

        www=www.reshape(16,www.shape[2])
        all_idx=all_idx.reshape(16,all_idx.shape[2])
            
        if nest:
            all_idx = hp.ring2nest(self.nside,all_idx)
            
        heal_idx,inv_idx = np.unique(all_idx,
                                    return_inverse=True)
        all_idx = inv_idx
            
        self.cell_ids = heal_idx
            
        hit=np.bincount(all_idx.flatten(),weights=www.flatten())
        www[hit[all_idx]<threshold]=0.0
        if self.gamma!=1:
            www=www**self.gamma
        www=www/np.sum(www,0)[None,:]
        return www,all_idx,heal_idx
        
    def P(self,x,www,all_idx):
        return np.sum(www*x[all_idx],0)
    
    #PT(y) must return a 1D NumPy array of shape (N,)
    def PT(self,y,www,all_idx,hit):
        value=np.bincount(all_idx.flatten(),weights=(www*y[None,:]).flatten())
        return value*hit

    # the data is of dimension M
    # the x is of dimension N=12*nside**2
    
    def conjugate_gradient_normal_equation(self,data, x0, www, all_idx, max_iter=100, tol=1e-8, verbose=True):
        """
        Solve (PᵗP)x = Pᵗy using explicit Conjugate Gradient without scipy.cg.
    
        Parameters:
        ----------
        P       : function(x) → forward operator (ℝⁿ → ℝᵐ)
        PT      : function(y) → adjoint operator (ℝᵐ → ℝⁿ)
        data    : array_like, observed data y ∈ ℝᵐ
        x0      : array_like, initial guess for x ∈ ℝⁿ
        max_iter: maximum number of iterations
        tol     : convergence tolerance on relative residual
        verbose : if True, print convergence info
    
        Returns:
        -------
        x       : estimated solution ∈ ℝⁿ
        """
        x = x0.copy()
        
        hit=np.bincount(all_idx.flatten(),weights=www.flatten())
        hit[hit>0]=1/hit[hit>0]
        
        # Compute b = Pᵗ y # This part could be distributed easily
        b = self.PT(data,www,all_idx,hit)
        
        # Compute initial residual r = b - A x = b - Pᵗ P x
        Ax = self.PT(self.P(x,www,all_idx),www,all_idx,hit)# This part could be distributed easily
        r = b - Ax
    
        # Initial direction
        p = r.copy()
        rs_old = np.dot(r, r)
    
        for i in range(max_iter):
            # Apply A p = Pᵗ P p
            Ap = self.PT(self.P(p,www,all_idx),www,all_idx,hit)# This part could be distributed easily
    
            alpha = rs_old / np.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
    
            rs_new = np.dot(r, r)
            
            if verbose and i%50==0:
                print(f"Iter {i:03d}: residual = {np.sqrt(rs_new):.3e}")
    
            if np.sqrt(rs_new) < tol:
                if verbose:
                    print(f"Converged. Iter {i:03d}: residual = {np.sqrt(rs_new):.3e}")
                break
    
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
    
        return x
        
    def Fit(self,X,th,ph,threshold=1E-2,nest=True, max_iter=100, tol=1e-8):
        
        www,all_idx,hidx=self.ang2weigths(th,ph,threshold=threshold,nest=nest)

        self.heal_idx=hidx
        self.spline=self.conjugate_gradient_normal_equation(X,
                                                            (self.heal_idx*0).astype('float'),
                                                            www,
                                                            all_idx
                                                            , max_iter=max_iter,
                                                            tol=tol
                                                            )
        scale=(self.nside//self.nside_store)**2
        h,ih=np.unique(hidx//scale,return_inverse=True)
        for k in range(h.shape[0]):
            spl=np.zeros([scale])
            spl[hidx[ih==k]-scale*h[k]]=self.spline[ih==k]
            self.spline_tree[h[k]]=spl

    def SetParam(self,nside,spline,heal_idx):
        
        self.heal_idx=heal_idx
        self.nside=nside
        self.spline=spline
        self.level=int(np.log2(nside))
        self.nside_store=2**(int(self.level//2))
        
        self.spline_tree={}
        
        scale=(self.nside//self.nside_store)**2
        h,ih=np.unique(heal_idx//scale,return_inverse=True)
        for k in range(h.shape[0]):
            spl=np.zeros([scale])
            spl[heal_idx[ih==k]-scale*h[k]]=self.spline[ih==k]
            self.spline_tree[h[k]]=spl
            
    def GetParam(self):
        return self.heal_idx,self.spline
    
    def Transform(self,th,ph,threshold=1E-2,nest=True):
        
        www,all_idx,hidx=self.ang2weigths(th,ph,threshold=threshold,nest=nest)

        x=np.zeros([hidx.shape[0]])
        scale=(self.nside//self.nside_store)**2
        h,ih=np.unique(hidx//scale,return_inverse=True)
        for k in range(h.shape[0]):
            if h[k] in self.spline_tree:
                spl=self.spline_tree[h[k]]
                x[ih==k]=spl[hidx[ih==k]-scale*h[k]]
        data=self.P(x,www,all_idx)
        return data

    def FitTransform(self,X,th,ph,threshold=1E-2,nest=True):

        self.Fit(X,th,ph)
        
        t,p=hp.pix2ang(self.nside,self.heal_idx,nest=True)

        return self.Transform(t,p)
