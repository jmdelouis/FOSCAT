import foscat.FoCUS as FOC
import numpy as np


def read(filename):
    thescat=scat(1,1,1)
    return thescat.read(filename)
    
class scat:
    def __init__(self,p00,s1,s2,cross=False):
        self.P00=p00
        self.S1=s1
        self.S2=s2
        self.cross=cross
        
    def get_S1(self):
        return(self.S1)
    
    def get_S2(self):
        return(self.S2)

    def get_P00(self):
        return(self.P00)

    def reset_P00(self):
        self.P00=0*self.P00

    def __add__(self,other):
        assert isinstance(other, float) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, scat)
        
        if isinstance(other, scat):
            return scat((self.P00+ other.P00), \
                        (self.S1+ other.S1), \
                        (self.S2+ other.S2))
        else:
            return scat((self.P00+ other), \
                        (self.S1+ other), \
                        (self.S2+ other))

    def __truediv__(self,other):
        assert isinstance(other, float) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, scat)
        
        if isinstance(other, scat):
            return scat((self.P00/ other.P00), \
                        (self.S1/ other.S1), \
                        (self.S2/ other.S2))
        else:
            return scat((self.P00/ other), \
                        (self.S1/ other), \
                        (self.S2/ other))
        

    def __rtruediv__(self,other):
        assert isinstance(other, float) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, scat)
        
        if isinstance(other, scat):
            return scat((other.P00/ self.P00), \
                        (other.S1 / self.S1), \
                        (other.S2 / self.S2))
        else:
            return scat((other/ self.P00), \
                        (other / self.S1), \
                        (other / self.S2))
        
    def __sub__(self,other):
        assert isinstance(other, float) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, scat)
        
        if isinstance(other, scat):
            return scat((self.P00- other.P00), \
                        (self.S1- other.S1), \
                        (self.S2- other.S2))
        else:
            return scat((self.P00- other), \
                        (self.S1- other), \
                        (self.S2- other))
        
    def __mul__(self,other):
        assert isinstance(other, float) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, scat)
        
        if isinstance(other, scat):
            return scat((self.P00* other.P00), \
                        (self.S1* other.S1), \
                        (self.S2* other.S2))
        else:
            return scat((self.P00* other), \
                        (self.S1* other), \
                        (self.S2* other))

    def plot(self,name=None,hold=True,color='blue',lw=1,legend=True):

        import matplotlib.pyplot as plt
        
        if name is None:
            name=''

        if hold:
            plt.figure(figsize=(8,8))
        
        plt.subplot(3,1,1)
        if legend:
            plt.plot(self.get_np(abs(self.S1)).flatten(),color=color,label=r'%s $S_1$'%(name),lw=lw)
        else:
            plt.plot(self.get_np(abs(self.S1)).flatten(),color=color,lw=lw)
        plt.yscale('log')
        plt.legend()
        plt.subplot(3,1,2)
        if legend:
            plt.plot(self.get_np(abs(self.P00)).flatten(),color=color,label=r'%s $P_{00}$'%(name),lw=lw)
        else:
            plt.plot(self.get_np(abs(self.P00)).flatten(),color=color,lw=lw)
        plt.yscale('log')
        plt.legend()
        plt.subplot(3,1,3)
        if legend:
            plt.plot(self.get_np(abs(self.S2)).flatten(),color=color,label=r'%s $S_2$'%(name),lw=lw)
        else:
            plt.plot(self.get_np(abs(self.S2)).flatten(),color=color,lw=lw)
        plt.yscale('log')
        plt.legend()
        
    def save(self,filename):
        np.save('%s_s1.npy'%(filename), self.get_S1().numpy())
        np.save('%s_s2.npy'%(filename), self.get_S2().numpy())
        np.save('%s_p0.npy'%(filename), self.get_P00().numpy())
        
    def read(self,filename):
        s1=np.load('%s_s1.npy'%(filename))
        s2=np.load('%s_s2.npy'%(filename))
        p0= np.load('%s_p0.npy'%(filename))
        return scat(p0,s1,s2)
    
    def get_np(self,x):
        if isinstance(x, np.ndarray):
            return x
        else:
            return x.numpy()

    def std(self):
        return np.sqrt(((abs(self.get_np(self.S1)).std())**2+(abs(self.get_np(self.S2)).std())**2+(abs(self.get_np(self.P00)).std())**2)/3)

    def mean(self):
        return abs(self.get_np(self.S1).mean()+self.get_np(self.S2).mean()+self.get_np(self.P00).mean())/3
        
        
    
class funct(FOC.FoCUS):
    
    def eval(self, image1, image2=None,mask=None,Auto=True):

        ### AUTO OR CROSS
        cross = False
        if image2 is not None:
            cross = True
            all_cross=Auto
        else:
            all_cross=False
            
        axis=1
        
        # determine jmax and nside corresponding to the input map
        im_shape = image1.shape
        if self.use_R_format and isinstance(image1,FOC.Rformat):
            if len(image1.shape)==4:
                nside=im_shape[2]-2*self.R_off
                npix = self.chans*nside*nside # Number of pixels
            else:
                nside=im_shape[1]-2*self.R_off
                npix = self.chans*nside*nside  # Number of pixels
        else:
            if len(image1.shape)==2:
                npix = int(im_shape[1])  # Number of pixels
            else:
                npix = int(im_shape[0])  # Number of pixels

            if self.chans==1:
                nside=im_shape[axis]
                npix=nside*nside
            else:
                nside=int(np.sqrt(npix//self.chans))
                
        jmax=int(np.log(nside)/np.log(2))-self.OSTEP

        ### LOCAL VARIABLES (IMAGES and MASK)
        # Check if image1 is [Npix] or [Nbatch,Npix]
        if len(image1.shape)==1 or (len(image1.shape)==2 and self.chans==1) or (len(image1.shape)==3 and isinstance(image1,FOC.Rformat)):
            # image1 is [Nbatch, Npix]
            I1 = self.bk_cast(self.bk_expand_dims(image1,0))  # Local image1 [Nbatch, Npix]
            if cross:
                I2 = self.bk_cast(self.bk_expand_dims(image2,0))  # Local image2 [Nbatch, Npix]
        else:
            I1=self.bk_cast(image1)
            if cross:
                I2=self.bk_cast(image2)
                
        # self.mask is [Nmask, Npix]
        if mask is None:
            if self.chans==1:
                vmask = self.bk_ones([1,nside,nside],dtype=self.all_type)
            else:
                vmask = self.bk_ones([1,npix],dtype=self.all_type)
            
            if self.use_R_format:
                vmask = self.to_R(vmask,axis=1,chans=self.chans)
        else:
            vmask = self.bk_cast( mask) # [Nmask, Npix]
            
            if self.use_R_format:
                vmask=self.to_R(vmask,axis=1,chans=self.chans)

        if self.use_R_format:
            I1=self.to_R(I1,axis=axis,chans=self.chans)
            if cross:
                I2=self.to_R(I2,axis=axis,chans=self.chans)

        if self.KERNELSZ>3:
            # if the kernel size is bigger than 3 increase the binning before smoothing
            l_image1=self.up_grade(I1,nside*2,axis=axis)
            vmask=self.up_grade(vmask,nside*2,axis=1)
            if cross:
                l_image2=self.up_grade(I2,nside*2,axis=axis)
        else:
            l_image1=I1
            if cross:
                l_image2=I2

        s1=None
        s2=None
        p00=None
        l2_image=None
        l2_image_imag=None

        for j1 in range(jmax):
            # Convol image along the axis defined by 'axis' using the wavelet defined at
            # the foscat initialisation
            #c_image_real is [....,Npix_j1,....,Norient]
            c_image1_real,c_image1_imag=self.convol(l_image1,axis=axis)
            if cross:
                c_image2_real,c_image2_imag=self.convol(l_image2,axis=axis)
            else:
                c_image2_real=c_image1_real
                c_image2_imag=c_image1_imag

            # Compute (a+ib)*(a+ib)* the last c_image column is the real and imaginary part
            conj_real=c_image1_real*c_image2_real+c_image1_imag*c_image2_imag
            if all_cross:
                conj_imag=c_image1_real*c_image2_imag-c_image1_imag*c_image2_real

            # Compute l_p00 [....,....,Nmask,1,Norient]  
            l_p00_real = self.bk_expand_dims(self.bk_masked_mean(conj_real,vmask,axis=axis),-2)
            if all_cross:
                l_p00_imag = self.bk_expand_dims(self.bk_masked_mean(conj_imag,vmask,axis=axis),-2)

            conj_real=self.bk_L1(conj_real)
            if all_cross:
                conj_imag=self.bk_L1(conj_imag)

            # Compute l_s1 [....,....,Nmask,1,Norient] 
            l_s1_real = self.bk_expand_dims(self.bk_masked_mean(conj_real,vmask,axis=axis),-2)
            if all_cross:
                l_s1_imag = self.bk_expand_dims(self.bk_masked_mean(conj_imag,vmask,axis=axis),-2)

            # Concat S1,P00 [....,....,Nmask,j1,Norient] 
            if s1 is None:
                if all_cross:
                    s1  = self.bk_complex(l_s1_real,l_s1_imag)
                    p00 = self.bk_complex(l_p00_real,l_p00_imag)
                else:
                    s1=l_s1_real
                    p00=l_p00_real
            else:
                if all_cross:
                    s1 =self.bk_concat([s1,self.bk_complex(l_s1_real,l_s1_imag)],axis=-2)
                    p00=self.bk_concat([p00,self.bk_complex(l_p00_real,l_p00_imag)],axis=-2)
                else:
                    s1=self.bk_concat([s1,l_s1_real],axis=-2)
                    p00=self.bk_concat([p00,l_p00_real],axis=-2)

            # Concat l2_image [....,Npix_j1,....,j1,Norient]
            if l2_image is None:
                l2_image=self.bk_expand_dims(self.update_R_border(conj_real,axis=axis),axis=-2)
            else:
                l2_image=self.bk_concat([self.bk_expand_dims(self.update_R_border(conj_real,axis=axis),axis=-2),l2_image],axis=-2)
            
            if all_cross:
                if l2_image_imag is None:
                    l2_image_imag=self.bk_expand_dims(self.update_R_border(conj_imag,axis=axis),axis=-2)
                else:
                    l2_image_imag=self.bk_concat([self.bk_expand_dims(self.update_R_border(conj_imag,axis=axis)
                                                                      ,axis=-2),l2_image_imag],axis=-2)

            # Convol l2_image [....,Npix_j1,....,j1,Norient,Norient]
            c2_image_real,c2_image_imag=self.convol(self.bk_relu(l2_image),axis=axis)
            if all_cross:
                c2_image_imag_real,c2_image_imag_imag=self.convol(self.bk_relu(l2_image_imag),axis=axis)

            conj2=self.bk_L1(c2_image_real*c2_image_real+c2_image_imag*c2_image_imag)
            if all_cross:
                conj2_imag=self.bk_L1(c2_image_imag_real*c2_image_imag_real+ \
                                      c2_image_imag_imag*c2_image_imag_imag)

            c2_image_real,c2_image_imag=self.convol(self.bk_relu(-l2_image),axis=axis)
            if cross:
                if all_cross:
                    c2_image_imag_real,c2_image_imag_imag=self.convol(self.bk_relu(-l2_image_imag),axis=axis)

                conj2=conj2-self.bk_L1(c2_image_real*c2_image_real+ \
                                       c2_image_imag*c2_image_imag)
                if all_cross:
                    conj2_imag=conj2_imag - self.bk_L1(c2_image_imag_real*c2_image_imag_real+ \
                                                       c2_image_imag_imag*c2_image_imag_imag)

            # Convol l_s2 [....,....,Nmask,j1,Norient,Norient]
            l_s2 = self.bk_masked_mean(conj2,vmask,axis=axis)

            if all_cross:
                l_imag_s2 = self.bk_masked_mean(conj2_imag,vmask,axis=axis)

            # Concat l_s2 [....,....,Nmask,j1*(j1+1)/2,Norient,Norient]
            if s2 is None:
                if all_cross:
                    s2=self.bk_complex(l_s2,l_imag_s2)
                else:
                    s2=l_s2
            else:
                if all_cross:
                    s2=self.bk_concat([s2,self.bk_complex(l_s2,l_imag_s2)],axis=-3)
                else:
                    s2=self.bk_concat([s2,l_s2],axis=-3)

            if j1!=jmax-1:
                # Rescale vmask [Nmask,Npix_j1//4]   
                vmask = self.smooth(vmask,axis=1)
                vmask = self.ud_grade_2(vmask,axis=1)

                # Rescale l2_image [....,Npix_j1//4,....,j1,Norient]   
                l2_image = self.smooth(l2_image,axis=axis)
                l2_image = self.ud_grade_2(l2_image,axis=axis)

                # Rescale l_image [....,Npix_j1//4,....]  
                l_image1 = self.smooth(l_image1,axis=axis)
                l_image1 = self.ud_grade_2(l_image1,axis=axis)
                if cross:
                    l_image2 = self.smooth(l_image2,axis=axis)
                    l_image2 = self.ud_grade_2(l_image2,axis=axis)
                    if all_cross:
                        l2_image_imag = self.smooth(l2_image_imag,axis=axis)
                        l2_image_imag = self.ud_grade_2(l2_image_imag,axis=axis)

        if len(image1.shape)==1 or (len(image1.shape)==3 and isinstance(image1,FOC.Rformat)):
            return(scat(p00[0],s1[0],s2[0],cross))

        return(scat(p00,s1,s2,cross))

    def square(self,x):
        # the abs make the complex value usable for reduce_sum or mean
        return scat(self.bk_abs(self.bk_square(x.P00)),
                    self.bk_abs(self.bk_square(x.S1)),
                    self.bk_abs(self.bk_square(x.S2)))

    def reduce_mean(self,x,axis=None):
        if axis is None:
            return  scat(self.bk_reduce_mean(x.P00),
                         self.bk_reduce_mean(x.S1),
                         self.bk_reduce_mean(x.S2))
        else:
            return scat(self.bk_reduce_mean(x.P00,axis=axis),
                        self.bk_reduce_mean(x.S1,axis=axis),
                        self.bk_reduce_mean(x.S2,axis=axis))

    def reduce_sum(self,x,axis=None):
        if axis is None:
            return  self.bk_reduce_sum(x.P00)+ \
                self.bk_reduce_sum(x.S1)+ \
                self.bk_reduce_sum(x.S2)
        else:
            return scat(self.bk_reduce_sum(x.P00,axis=axis),
                        self.bk_reduce_sum(x.S1,axis=axis),
                        self.bk_reduce_sum(x.S2,axis=axis))
            

    def log(self,x):
        return scat(self.bk_log(x.P00),
                    self.bk_log(x.S1),
                    self.bk_log(x.S2))
    def inv(self,x):
        return scat(1/(x.P00),1/(x.S1),1/(x.S2))

    def one(self):
        return scat(1.0,1.0,1.0)
        
