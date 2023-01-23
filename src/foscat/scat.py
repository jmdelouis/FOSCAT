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

    def __add__(self,val):
        return scat(self.P00 + val.P00, \
                    self.S1 + val.S1, \
                    self.S2 + val.S2,
                    cross=self.cross)

    def __div__(self,val1):
        return scat(self.P00 / val2.P00, \
                    self.S1 / val2.S1, \
                    self.S2 / val2.S2,
                    cross=self.cross)

    def __truediv__(self,val):
        return self.__div__(self,val)
        
    def __sub__(self,val):
        return scat(self.P00 - val.P00, \
                    self.S1 - val.S1, \
                    self.S2 - val.S2,
                    cross=self.cross)
        
    def __mul__(self,val):
        return scat(self.P00 * val.P00, \
                    self.S1 * val.S1, \
                    self.S2 * val.S2,
                    cross=self.cross)

    def plot(self,name=None,hold=True,color='blue',lw=1):

        import matplotlib.pyplot as plt
        
        if name is None:
            name=''

        if hold:
            plt.figure(figsize=(8,8))
        
        plt.subplot(3,1,1)
        plt.plot(self.get_np(abs(self.S1)).flatten(),color=color,label=r'%s $S_1$'%(name),lw=lw)
        plt.yscale('log')
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(self.get_np(abs(self.P00)).flatten(),color=color,label=r'%s $P_{00}$'%(name),lw=lw)
        plt.yscale('log')
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(self.get_np(abs(self.S2)).flatten(),color=color,label=r'%s $S_2$'%(name),lw=lw)
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
        return scat(s1,s2,p0)
    
    def get_np(self,x):
        if 'numpy.ndarray' in '%s'%(x.__class__):
            return x
        else:
            x.numpy()
        return(x)
        
    
class funct(FOC.FoCUS):

    def eval(self, image1, image2=None,mask=None):
        
        ### AUTO OR CROSS
        cross = False
        if image2 is not None:
            cross = True
        axis=1
        # determine jmax and nside corresponding to the input map
        im_shape = image1.shape
        if len(image1.shape)==2:
            npix = int(im_shape[1])  # Number of pixels
        else:
            npix = int(im_shape[0])  # Number of pixels
            
        nside=int(np.sqrt(npix//12))
        jmax=int(np.log(nside)/np.log(2))-self.OSTEP
        
        ### LOCAL VARIABLES (IMAGES and MASK)
        # Check if image1 is [Npix] or [Nbatch,Npix]
        if len(image1.shape)==1:
            # image1 is [Nbatch, Npix]
            I1 = self.bk_cast(image1[None, :])  # Local image1 [Nbatch, Npix]
            if cross:
                I2 = self.bk_cast(image2[None, :])  # Local image2 [Nbatch, Npix]
        else:
            I1=self.bk_cast(image1)
            if cross:
                I2=self.bk_cast(image2)
                
        # self.mask is [Nmask, Npix]
        if mask is None:
            vmask=self.bk_ones([1,npix],dtype=self.all_type)
        else:
            vmask = mask # [Nmask, Npix]

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
            if cross:
                conj_imag=c_image1_real*c_image2_imag-c_image1_imag*c_image2_real
            
            # Compute l_p00 [....,....,1,Norient]  
            l_p00_real = self.bk_expand_dims(self.bk_reduce_mean(conj_real,axis=axis),-2)
            if cross:
                l_p00_imag = self.bk_expand_dims(self.bk_reduce_mean(conj_imag,axis=axis),-2)
            
            conj_real=self.bk_L1(conj_real)
            if cross:
                conj_imag=self.bk_L1(conj_imag)

            # Compute l_s1 [....,....,1,Norient] 
            l_s1_real = self.bk_expand_dims(self.bk_reduce_mean(conj_real,axis=axis),-2)
            if cross:
                l_s1_imag = self.bk_expand_dims(self.bk_reduce_mean(conj_imag,axis=axis),-2)
            
            # Concat S1,P00 [....,....,j1,Norient] 
            if s1 is None:
                if cross:
                    s1  = self.bk_complex(l_s1_real,l_s1_imag)
                    p00 = self.bk_complex(l_p00_real,l_p00_imag)
                else:
                    s1=l_s1_real
                    p00=l_p00_real
            else:
                if cross:
                    s1 =self.bk_concat([s1,self.bk_complex(l_s1_real,l_s1_imag)],axis=-2)
                    p00=self.bk_concat([p00,self.bk_complex(l_p00_real,l_p00_imag)],axis=-2)
                else:
                    s1=self.bk_concat([s1,l_s1_real],axis=-2)
                    p00=self.bk_concat([p00,l_p00_real],axis=-2)

            # Concat l2_image [....,Npix_j1,....,j1,Norient]   
            if l2_image is None:
                l2_image=self.bk_expand_dims(conj_real,axis=-2)
            else:
                l2_image=self.bk_concat([self.bk_expand_dims(conj_real,axis=-2),l2_image],axis=-2)
                 
            if cross:
                if l2_image_imag is None:
                    l2_image_imag=self.bk_expand_dims(conj_imag,axis=-2)
                else:
                    l2_image_imag=self.bk_concat([self.bk_expand_dims(conj_imag,axis=-2),l2_image_imag],axis=-2)
            
            # Convol l2_image [....,Npix_j1,....,j1,Norient,Norient]
            c2_image_real,c2_image_imag=self.convol(l2_image,axis=axis)
            if cross:
                c2_image_imag_real,c2_image_imag_imag=self.convol(l2_image_imag,axis=axis)
            
            conj2=self.bk_sqrt(c2_image_real*c2_image_real+c2_image_imag*c2_image_imag)
            if cross:
                conj2_imag=self.bk_sqrt(c2_image_imag_real*c2_image_imag_real+c2_image_imag_imag*c2_image_imag_imag)
            
            # Convol l_s2 [....,....,j1,Norient,Norient]
            l_s2 = self.bk_reduce_mean(conj2,axis=axis)
            if cross:
                l_imag_s2 = self.bk_reduce_mean(conj2_imag,axis=axis)
            
            # Concat l_s2 [....,....,j1*(j1+1)/2,Norient,Norient]
            if s2 is None:
                if cross:
                    s2=self.bk_complex(l_s2,l_imag_s2)
                else:
                    s2=l_s2
            else:
                if cross:
                    s2=self.bk_concat([s2,self.bk_complex(l_s2,l_imag_s2)],axis=-3)
                else:
                    s2=self.bk_concat([s2,l_s2],axis=-3)

            # Rescale l2_image [....,Npix_j1//4,....,j1,Norient]   
            l2_image = self.smooth(l2_image,axis=axis)
            l2_image = self.ud_grade_2(l2_image,axis=axis)
                
            # Rescale l_image [....,Npix_j1//4,....]  
            l_image1 = self.smooth(l_image1,axis=axis)
            l_image1 = self.ud_grade_2(l_image1,axis=axis)
            if cross:
                l_image2 = self.smooth(l_image2,axis=axis)
                l_image2 = self.ud_grade_2(l_image2,axis=axis) 
                l2_image_imag = self.smooth(l2_image_imag,axis=axis)
                l2_image_imag = self.ud_grade_2(l2_image_imag,axis=axis)
        
        if len(image1.shape)==1:
            return(scat(p00[0],s1[0],s2[0],cross))
            
        return(scat(p00,s1,s2,cross))

    def square(self,x):
        # the abs make the complex value usable for reduce_sum or mean
        return scat(self.bk_abs(self.bk_square(x.P00)),
                    self.bk_abs(self.bk_square(x.S1)),
                    self.bk_abs(self.bk_square(x.S2)))

    def reduce_mean(self,x,axis=None):
        if axis is None:
            return  self.bk_reduce_mean(x.P00)+ \
                self.bk_reduce_mean(x.S1)+ \
                self.bk_reduce_mean(x.S2)
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
        
