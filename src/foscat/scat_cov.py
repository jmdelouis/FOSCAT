import foscat.FoCUS as FOC
import numpy as np

def read(filename):
    thescat=scat_cov(1,1,1)
    return thescat.read(filename)

class scat_cov:
    def __init__(self,p00,c01,c11,s1=None):
        self.P00=p00
        self.C01=c01
        self.C11=c11
        self.S1=s1

    def get_S1(self):
        return(self.S1)

    def get_P00(self):
        return(self.P00)

    def get_C01(self):
        return(self.C01)

    def get_C11(self):
        return(self.C11)

    def __add__(self,val):
        if self.S1 is None:
            if val.S1 is not None:
                print('Impossible to sum two scat_cov with S1 commonly defined or undefined')
            
            return scat_cov(self.P00 + val.P00, \
                            self.C01 + val.C01, \
                            self.C11 + val.C11)
        else:
            if val.S1 is None:
                print('Impossible to sum two scat_cov with S1 commonly defined or undefined')
            
            return scat_cov(self.P00 + val.P00, \
                            self.C01 + val.C01, \
                            self.C11 + val.C11, \
                            s1=self.S1 + val.S1)

    def __div__(self,val):
        if self.S1 is None:
            if val.S1 is not None:
                print('Impossible to sum two scat_cov with S1 commonly defined or undefined')
            
            return scat_cov(self.P00/val.P00, \
                            self.C01/val.C01, \
                            self.C11/val.C11)
        else:
            if val.S1 is None:
                print('Impossible to sum two scat_cov with S1 commonly defined or undefined')
            
            return scat_cov(self.P00/val.P00, \
                            self.C01/val.C01, \
                            self.C11/val.C11, \
                            s1=self.S1/val.S1)

    def __truediv__(self,val):
        return self.__div__(self,val)
        
    def __sub__(self,val):
        if self.S1 is None:
            if val.S1 is not None:
                print('Impossible to sum two scat_cov with S1 commonly defined or undefined')
            
            return scat_cov(self.P00 - val.P00, \
                            self.C01 - val.C01, \
                            self.C11 - val.C11)
        else:
            if val.S1 is None:
                print('Impossible to sum two scat_cov with S1 commonly defined or undefined')
            
            return scat_cov(self.P00 - val.P00, \
                            self.C01 - val.C01, \
                            self.C11 - val.C11, \
                            s1=self.S1 - val.S1)
        
    def __mul__(self,val):
        if self.S1 is None:
            if val.S1 is not None:
                print('Impossible to sum two scat_cov with S1 commonly defined or undefined')
            
            return scat_cov(self.P00 * val.P00, \
                            self.C01 * val.C01, \
                            self.C11 * val.C11)
        else:
            if val.S1 is None:
                print('Impossible to sum two scat_cov with S1 commonly defined or undefined')
            
            return scat_cov(self.P00 * val.P00, \
                            self.C01 * val.C01, \
                            self.C11 * val.C11, \
                            s1=self.S1 * val.S1)
    
    def plot(self,name=None,hold=True,color='blue',lw=1):

        import matplotlib.pyplot as plt
        
        if name is None:
            name=''

        if hold:
            plt.figure(figsize=(8,8))
        
        if self.S1 is not None:
            plt.subplot(2,2,1)
            plt.plot(self.get_np(abs(self.S1)).flatten(),color=color,label=r'%s $S_1$'%(name),lw=lw)
            plt.yscale('log')
            plt.legend()
        plt.subplot(2,2,2)
        plt.plot(self.get_np(abs(self.P00)).flatten(),color=color,label=r'%s $P_{00}$'%(name),lw=lw)
        plt.yscale('log')
        plt.legend()
        plt.subplot(2,2,3)
        plt.plot(self.get_np(abs(self.C01)).flatten(),color=color,label=r'%s $C_{01}$'%(name),lw=lw)
        plt.yscale('log')
        plt.legend()
        plt.subplot(2,2,4)
        plt.plot(self.get_np(abs(self.C11)).flatten(),color=color,label=r'%s $C_{11}$'%(name),lw=lw)
        plt.yscale('log')
        plt.legend()

    def get_np(self,x):
        if 'numpy.ndarray' in '%s'%(x.__class__):
            return x
        else:
            x.numpy()
        return(x)
    
    def save(self,filename):
        if self.S1 is not None:
            np.save('%s_s1.npy'%(filename), self.get_S1().numpy())
        np.save('%s_c01.npy'%(filename), self.get_C01().numpy())
        np.save('%s_c11.npy'%(filename), self.get_C11().numpy())
        np.save('%s_p0.npy'%(filename), self.get_P00().numpy())
        
    def read(self,filename):
        try:
            s1=np.load('%s_s1.npy'%(filename))
        except:
            s1=None
            
        c01=np.load('%s_c01.npy'%(filename))
        c11=np.load('%s_c11.npy'%(filename))
        p0= np.load('%s_p0.npy'%(filename))
        
        return scat_cov(p0,c01,c11,s1=s1)
        
class funct(FOC.FoCUS):

        
    def eval(self, image1, image2=None,mask=None):
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
        Returns
        -------
        S1, P00, C01, C11 normalized
        """
        ### AUTO OR CROSS
        cross = False
        if image2 is not None:
            cross = True
            
        ### PARAMETERS
        im_shape = image1.shape
        if len(image1.shape)==2:
            npix = int(im_shape[1])  # Number of pixels
            BATCH_SIZE = im_shape[0]
        else:
            npix = int(im_shape[0])  # Number of pixels
            BATCH_SIZE = 1
            
        n0 = int(np.sqrt(npix / 12))  # NSIDE
        J = int(np.log(np.sqrt(npix / 12)) / np.log(2)) # Number of j scales
        Jmax = J - self.OSTEP # Number of steps for the loop on scales
        # self.KERNELSZ is the number of pixel on one side (3, 5, 7...)
        kersize2 = self.KERNELSZ ** 2 # Kernel size square (9, 25, 49...)
        
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
            vmask=self.bk_ones([1,npix],dtype=self.all_bk_type)
        else:
            vmask = self.bk_cast(mask)  # [Nmask, Npix]
            
        if self.KERNELSZ>3:
            # if the kernel size is bigger than 3 increase the binning before smoothing
            I1=self.up_grade(I1,n0*2,axis=1)
            vmask=self.up_grade(vmask,n0*2,axis=1)
            if cross:
                I2=self.up_grade(I2,n0*2,axis=1)
        # Normalize the masks because they have different pixel numbers
        vmask /= self.bk_reduce_sum(vmask, axis=1)[:, None]  # [Nmask, Npix]

        ### COEFFS INITIALIZATION
        S1, P00, C01, C11 = None, None, None, None
        M1_dic, P1_dic = {}, {}  # M stands for Module
        if cross:
            M2_dic, P2_dic = {}, {}

        #### COMPUTE S1, P00, C01 and C11
        nside_j3 = n0   # NSIDE start (nside_j3 = n0 / 2^j3)
        npix_j3 = npix  # Pixel number at each iteration on j3 
        for j3 in range(Jmax):

            ### Make the convolution I1 * Psi_j3
            cconv1, sconv1, M1_square, M1 = self._compute_IconvPsi(I1, nside_j3, BATCH_SIZE, npix_j3, kersize2)
            # Store M1_j3 in a dictionary
            M1_dic[j3] = M1

            if cross:
                ### Make the convolution I2 * Psi_j3
                cconv2, sconv2, M2_square, M2 = self._compute_IconvPsi(I2, nside_j3, BATCH_SIZE, npix_j3, kersize2)
                # Store M2_j3 in a dictionary
                M2_dic[j3] = M2
            
            ####### S1 and P00
            ### P00_auto = < M1^2 >_pix
            #  M1_square [Nbatch,Npix,Norient3]
            #  vmask  [Nmask,Npix]
            
            p00 = self.bk_reduce_sum(vmask[None, :, :,None] * M1_square[:, None, :, :],
                                axis=2)  # [Nbatch, Nmask, Norient3]
            # We store it for normalisation of C01 and C11
            P1_dic[j3] = p00  # [Nbatch, Nmask, Norient3]
            
            if not cross:
                # We store P00_auto to be returned
                if P00 is None:
                    P00 = p00[:, :, None, :]  # Add a dimension for NP00
                else:
                    P00 = self.bk_concat([P00, p00[:, :, None, :]], axis=2)
                    
                #### S1_auto computation
                ### Image 1 : S1 = < M1 >_pix
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                s1 = self.bk_reduce_sum(vmask[None, :, :,None] * M1[:, None, :, :],
                                   axis=2)  # [Nbatch, Nmask, Norient3]  # [Nbatch, Nmask, Norient3]
                ### We store S1 for image1
                if S1 is None:
                    S1 = s1[:, :, None, :]  # Add a dimension for NS1
                else:
                    S1 = self.bk_concat([S1, s1[:, :, None, :]], axis=2)

            else:
                ### P00_auto = < M2^2 >_pix
                p00 = self.bk_reduce_sum(vmask[None, :, :, None] * M2_square[:, None, :, :],
                                    axis=2)  # [Nbatch, Nmask, Norient3]
                # We store it for normalisation
                P2_dic[j3] = p00  # [Nbatch, Nmask, Norient3]

                ### P00_cross = < (I1 * Psi_j3) (I2 * Psi_j3)^* >_pix
                # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
                p00_real = cconv1 * cconv2 + sconv1 * sconv2
                p00_imag = sconv1 * cconv2 - cconv1 * sconv2
                # Apply the mask [Nmask, Npix_j3] and average over pixels
                p00_real = self.bk_reduce_sum(vmask[None, :, :, None] * p00_real[:, None, :, :],
                                         axis=2)  # [Nbatch, Nmask, Norient3]
                p00_imag = self.bk_reduce_sum(vmask[None, :, :, None] * p00_imag[:, None, :, :],
                                         axis=2)  # [Nbatch, Nmask, Norient3]
                ### We store P00_cross
                if P00 is None:
                    P00 = self.bk_complex(p00_real[:, :, None, :], p00_imag[:, :, None, :])  # Add a dimension for NP00
                else:
                    P00 = self.bk_concat([P00, self.bk_complex(p00_real[:, :, None, :], \
                                                               p00_imag[:, :, None, :])], axis=2)

            # Initialize dictionaries for |I1*Psi_j| * Psi_j3
            cM1convPsi_dic = {}
            sM1convPsi_dic = {}
            if cross:
                # Initialize dictionaries for |I2*Psi_j| * Psi_j3
                cM2convPsi_dic = {}
                sM2convPsi_dic = {}

            ###### C01
            for j2 in range(0, j3):  # j2 <= j3
                ### C01_auto = < (I1 * Psi)_j3 x (|I1 * psi2| * Psi_j3)^* >_pix
                if not cross:
                    cc01, sc01 = self._compute_C01_auto(j2, cconv1, sconv1, vmask,
                                                        M1_dic, cM1convPsi_dic, sM1convPsi_dic,
                                                        nside_j3, BATCH_SIZE, npix_j3, kersize2)
                    ### Normalize C01 with P00 [Nbatch, Nmask, Norient]
                    cc01 /= (P1_dic[j2][:, :, :, None] *
                             P1_dic[j3][:, :, None, :]) ** 0.5  # [Nbatch, Nmask, Norient2, Norient3]
                    sc01 /= (P1_dic[j2][:, :, :, None] *
                             P1_dic[j3][:, :, None, :]) ** 0.5  # [Nbatch, Nmask, Norient2, Norient3]
                    ### Store C01
                    if C01 is None:
                        C01 = self.bk_concat([cc01[:, :, None, :, :], sc01[:, :, None, :, :]], axis=2)  # Add a dimension for NC01
                    else:
                        C01 = self.bk_concat([C01, cc01[:, :, None, :, :], sc01[:, :, None, :, :]], axis=2)  # Add a dimension for NC01

                    ### C01_cross = < (I1 * Psi)_j3 x (|I2 * psi2| * Psi_j3)^* >_pix
                    ### C01_cross_bis = < (I2 * Psi)_j3 x (|I1 * psi2| * Psi_j3)^* >_pix
                else:
                    cc01, sc01, cc01_bis, sc01_bis = self._compute_C01_cross(j2, cconv1, sconv1, cconv2, sconv2, vmask,
                                                                             M1_dic, M2_dic,
                                                                             cM1convPsi_dic, sM1convPsi_dic,
                                                                             cM2convPsi_dic, sM2convPsi_dic,
                                                                             nside_j3, BATCH_SIZE, npix_j3, kersize2)
                    ### Normalize C01 with P00 [Nbatch, Nmask, Norient]
                    cc01 /= (P2_dic[j2][:, :, :, None] *
                             P1_dic[j3][:, :, None, :]) ** 0.5  # [Nbatch, Nmask, Norient2, Norient3]
                    sc01 /= (P2_dic[j2][:, :, :, None] *
                             P1_dic[j3][:, :, None, :]) ** 0.5  # [Nbatch, Nmask, Norient2, Norient3]
                    cc01_bis /= (P1_dic[j2][:, :, :, None] *
                                 P2_dic[j3][:, :, None, :]) ** 0.5  # [Nbatch, Nmask, Norient2, Norient3]
                    sc01_bis /= (P1_dic[j2][:, :, :, None] *
                                 P2_dic[j3][:, :, None, :]) ** 0.5  # [Nbatch, Nmask, Norient2, Norient3]
                    ### Store C01
                    if C01 is None:
                        C01 = self.bk_concat([
                            self.bk_complex(cc01[:, :, None, :, :], sc01[:, :, None, :, :]),
                            self.bk_complex(cc01_bis[:, :, None, :, :], sc01_bis[:, :, None, :, :])],
                                        axis=2)  # Add a dimension for NC01
                    else:
                        C01 = self.bk_concat([C01,
                                              self.bk_complex(cc01[:, :, None, :, :], sc01[:, :, None, :, :]),
                                              self.bk_complex(cc01_bis[:, :, None, :, :], sc01_bis[:, :, None, :, :])],
                                             axis=2)  # Add a dimension for NC01

                ##### C11
                for j1 in range(0, j2):  # j1 <= j2
                    ### C11_auto = <(|I1 * psi1| * psi3)(|I1 * psi2| * psi3)^*>
                    if not cross:
                        cc11, sc11 = self._compute_C11_auto(j1, j2, vmask, cM1convPsi_dic, sM1convPsi_dic)
                        ### Normalize C11 with P00_j [Nbatch, Nmask, Norient_j]
                        cc11 /= (P1_dic[j1][:, :, None, :, None] *
                                 P1_dic[j2][:, :, None, None, :]) ** 0.5  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        sc11 /= (P1_dic[j1][:, :, None, :, None] *
                                 P1_dic[j2][:, :, None, None, :]) ** 0.5  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        # We store C11
                        if C11 is None:
                            C11 = self.bk_concat([cc11[:, :, None, :, :, :], sc11[:, :, None, :, :, :]],
                                            axis=2)  # Add a dimension for NC11
                        else:
                            C11 = self.bk_concat([C11, cc11[:, :, None, :, :, :], sc11[:, :, None, :, :, :]],
                                            axis=2)  # Add a dimension for NC11

                        ### C11_cross = <(|I1 * psi1| * psi3)(|I2 * psi2| * psi3)^*>
                    else:
                        cc11, sc11 = self._compute_C11_cross(j1, j2, vmask,
                                                             cM1convPsi_dic, sM1convPsi_dic,
                                                             cM2convPsi_dic, sM2convPsi_dic)
                        ### Normalize C11 with P00_j [Nbatch, Nmask, Norient_j]
                        cc11 /= (P1_dic[j1][:, :, None, :, None] *
                                 P2_dic[j2][:, :, None, None, :]) ** 0.5  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        sc11 /= (P1_dic[j1][:, :, None, :, None] *
                                 P2_dic[j2][:, :, None, None, :]) ** 0.5  # [Nbatch, Nmask, Norient3, Norient2, Norient1]
                        
                        # We store C11
                        if C11 is None:
                            C11 = self.bk_concat([cc11[:, :, None, :, :, :], sc11[:, :, None, :, :, :]],
                                            axis=2)  # Add a dimension for NC11
                        else:
                            C11 = self.bk_concat([C11, cc11[:, :, None, :, :, :], sc11[:, :, None, :, :, :]],
                                            axis=2)  # Add a dimension for NC11

            ###### Reshape for next iteration on j3
            ### Image I1, 
            #downscale the I1 [Nbatch,Npix_j3]
            I1_smooth = self.smooth(I1,axis=1)
            I1 = self.ud_grade_2(I1_smooth,axis=1)
            
            ### Image I2
            if cross:
                I2_smooth = self.smooth(I2,axis=1)
                I2 = self.ud_grade_2(I2_smooth,axis=1)
                
            ### Modules
            for j2 in range(0, j3+1):  # j2 <= j3
                ### Dictionary M1_dic[j2]
                M1_smooth = self.smooth(M1_dic[j2],axis=1) #[Nbatch, Npix_j3, Norient3]
                M1_dic[j2] = self.ud_grade_2(M1_smooth,axis=1) #[Nbatch, Npix_j3, Norient3]
                
                ### Dictionary M2_dic[j2]
                if cross:
                    M2_smooth = self.smooth(M2_dic[j2],axis=1) #[Nbatch, Npix_j3, Norient3]
                    M2_dic[j2] = self.ud_grade_2(M2_smooth,axis=1) #[Nbatch, Npix_j3, Norient3]
                    
            ### Mask
            vmask = self.ud_grade_2(vmask,axis=1)
            
            ### NSIDE_j3 and npix_j3
            nside_j3 = nside_j3 // 2
            npix_j3 = 12 * nside_j3**2

        ###### Normalize S1 and P00
        P00 = self.bk_log(P00)
        
        if not cross:
            S1 = self.bk_log(S1)
            return(scat_cov(P00, C01, C11,s1=S1))
        else:
            return(scat_cov(P00, C01, C11))

    def _compute_IconvPsi(self, I, nside_j3, BATCH_SIZE, npix_j3, kersize2):
        """
        Make the convolution I * Psi_j3
        Returns
        -------
        Use convol function
        """
        cconv,sconv=self.convol(I,axis=1)
        
        # Module square |I * Psi_j3|^2
        M_square = cconv * cconv + sconv * sconv  # [Nbatch, Npix_j3, Norient3]
        # Module |I * Psi_j3|
        M = self.bk_sqrt(M_square)  # [Nbatch, Norient3, Npix_j3]
        return cconv, sconv, M_square, M

    def _compute_C01_auto(self, j2, cconv1, sconv1, vmask, M1_dic, cM1convPsi_dic, sM1convPsi_dic,
                          nside_j3, BATCH_SIZE, npix_j3, kersize2):
        ### Compute |I1 * psi2| * Psi_j3 = M1_j2 * Psi_j3
        # Warning: M1_dic[j2] is already at j3 resolution [Nbatch, Npix_j3, Norient3]
        # self.widx2[nside_j3] is [Npix_j3 x kersize2]
        
        cM1convPsi,sM1convPsi=self.convol(M1_dic[j2],axis=1)
        
        # Store it so we can use it in C11 computation
        cM1convPsi_dic[j2] = cM1convPsi  # [Nbatch, Npix_j3,Norient3, Norient2]
        sM1convPsi_dic[j2] = sM1convPsi  # [Nbatch, Npix_j3,Norient3, Norient2]

        ### Compute the product (I1 * Psi)_j3 x (M1_j2 * Psi_j3)^*
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        # cconv1, sconv1 are [Nbatch, Npix_j3, Norient3]
        cc01 = cconv1[:, :, :, None] * cM1convPsi + \
               sconv1[:, :, :, None] * sM1convPsi  # Real [Nbatch, Npix_j3, Norient3, Norient2]
        sc01 = sconv1[:, :, :, None] * cM1convPsi - \
               cconv1[:, :, :, None] * sM1convPsi  # Imag [Nbatch, Npix_j3, Norient3, Norient2]

        ### Sum over pixels after applying the mask [Nmask, Npix_j3]
        cc01 = self.bk_reduce_sum(vmask[None, :, :,None, None] *
                             cc01[:, None, :, :, :], axis=2)  # Real [Nbatch, Nmask, Norient3, Norient2]
        sc01 = self.bk_reduce_sum(vmask[None, :, :, None, None] *
                             sc01[:, None, :, :, :], axis=2)  # Imag [Nbatch, Nmask, Norient3, Norient2]
        return cc01, sc01

    def _compute_C01_cross(self, j2, cconv1, sconv1, cconv2, sconv2,
                           vmask, M1_dic, M2_dic,
                           cM1convPsi_dic, sM1convPsi_dic,
                           cM2convPsi_dic, sM2convPsi_dic,
                           nside_j3, BATCH_SIZE, npix_j3, kersize2):
        """
        Compute the C01 cross coefficients
        C01_cross = < (I2 * Psi)_j3 x (|I1 * psi2| * Psi_j3)^* >_pix
        C01_cross_bis = < (I1 * Psi)_j3 x (|I2 * psi2| * Psi_j3)^* >_pix
        Parameters
        ----------
        Returns
        -------
        cc01, sc01, cc01_bis, sc01_bis: real and imag parts of C01 cross coeff
        """

        ####### C01_cross
        ### Compute |I1 * psi2| * Psi_j3 = M1_j2 * Psi_j3
        # Warning: M1_dic[j2] is already at j3 resolution [Nbatch, Norient3, Npix_j3]
        # self.widx2[nside_j3] is [Npix_j3 x kersize2]
        cM1convPsi,sM1convPsi=self.convol(M1_dic[j2],axis=1)
        
        # Store it so we can use it in C11 computation
        cM1convPsi_dic[j2] = cM1convPsi  # [Nbatch, Npix_j3, Norient3, Norient2]
        sM1convPsi_dic[j2] = sM1convPsi  # [Nbatch, Npix_j3, Norient3, Norient2]
        ### Compute the product (I2 * Psi)_j3 x (M1_j2 * Psi_j3)^*
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        # cconv1, sconv1 are [Nbatch, Npix_j3, Norient3]
        # cM1convPsi, sM1convPsi are [Nbatch, Npix_j3, Norient3,Norient2]
        
        cc01 = cconv2[:, :, :, None] * cM1convPsi + \
               sconv2[:, :, :, None] * sM1convPsi  # Real [Nbatch, Npix_j3, Norient3, Norient2]
        sc01 = sconv2[:, :, :, None] * cM1convPsi - \
               cconv2[:, :, :, None] * sM1convPsi  # Imag [Nbatch, Npix_j3, Norient3, Norient2]
        
        ### Sum over pixels after applying the mask [Nmask, Npix_j3]
        cc01 = self.bk_reduce_sum(vmask[None, :, :, None, None] *
                             cc01[:, None,:, :, :], axis=2)  # Real [Nbatch, Nmask, Norient3, Norient2]
        sc01 = self.bk_reduce_sum(vmask[None, :, :, None, None] *
                             sc01[:, None,:, :, :], axis=2)  # Imag [Nbatch, Nmask, Norient3, Norient2]

        ####### C01_cross_bis
        ### Compute |I2 * psi2| * Psi_j3 = M2_j2 * Psi_j3
        # Warning: M2_dic[j2] is already at j3 resolution [Nbatch, Norient3, Npix_j3]
        # self.widx2[nside_j3] is [Npix_j3 x kersize2]
        cM2convPsi,sM2convPsi=self.convol(M2_dic[j2],axis=1)
        
        # Store it so we can use it in C11 computation
        cM2convPsi_dic[j2] = cM2convPsi  # [Nbatch, Npix_j3, Norient3, Norient2]
        sM2convPsi_dic[j2] = sM2convPsi  # [Nbatch, Npix_j3, Norient3, Norient2]
        ### Compute the product (I1 * Psi)_j3 x (M2_j2 * Psi_j3)^*
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        # cconv1, sconv1 are [Nbatch, Npix_j3, Norient3]
        cc01_bis = cconv1[:, :, :, None] * cM2convPsi + \
                       sconv1[:, :, :, None] * sM2convPsi  # Real [Nbatch, Npix_j3, Norient3, Norient2]
        sc01_bis = sconv1[:, :, :, None] * cM2convPsi - \
                       cconv1[:, :, :, None] * sM2convPsi  # Imag [Nbatch, Npix_j3, Norient3, Norient2]

        ### Sum over pixels after applying the mask [Nmask, Npix_j3]
        cc01_bis = self.bk_reduce_sum(vmask[None, :, :, None, None] *
                                 cc01_bis[:, None, :, :, :], axis=2)  # Real [Nbatch, Nmask, Norient3, Norient2]
        sc01_bis = self.bk_reduce_sum(vmask[None, :, :, None, None] *
                                 sc01_bis[:, None, :, :, :], axis=2)  # Imag [Nbatch, Nmask, Norient3, Norient2]
        return cc01, sc01, cc01_bis, sc01_bis

    def _compute_C11_auto(self, j1, j2, vmask, cM1convPsi_dic, sM1convPsi_dic):
        ### Compute the product (|I1 * psi1| * psi3)(|I1 * psi2| * psi3)
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        # cM1convPsi_dic[j] is [Nbatch, Npix_j3, Norient, Norient3]
        cc11 = cM1convPsi_dic[j1][:, :, :,None, :] * cM1convPsi_dic[j2][:,:, :, :, None] + \
               sM1convPsi_dic[j1][:, :, :,None, :] * sM1convPsi_dic[j2][:,:, :, :, None]  # Real [Nbatch, Npix_j3, Norient3, Norient2, Norient1]
        sc11 = sM1convPsi_dic[j1][:, :, :,None, :] * cM1convPsi_dic[j2][:,:, :, :, None] - \
               cM1convPsi_dic[j1][:, :, :,None, :] * sM1convPsi_dic[j2][:,:, :, :, None]  # Imag [Nbatch, Npix_j3, Norient3, Norient2, Norient1]
        ### Sum over pixels and apply the mask
        cc11 = self.bk_reduce_sum(vmask[None, :, : ,None, None, None] *
                             cc11[:, None,:, :, :, :],
                             axis=2)  # Real [Nbatch, Nmask, Norient1, Norient2, Norient3]
        sc11 = self.bk_reduce_sum(vmask[None, :, : ,None, None, None] *
                             sc11[:, None,:, :, :, :],
                             axis=2)  # Imag [Nbatch, Nmask, Norient1, Norient2, Norient3]
        return cc11, sc11

    def _compute_C11_cross(self, j1, j2, vmask, cM1convPsi_dic, sM1convPsi_dic, cM2convPsi_dic, sM2convPsi_dic):

        ### Compute the product (|I1 * psi1| * psi3)(|I2 * psi2| * psi3)
        # z_1 x z_2^* = (a1a2 + b1b2) + i(b1a2 - a1b2)
        # cM1convPsi_dic[j] is [Nbatch, Norient, Norient3, Npix_j3]
        cc11 = cM1convPsi_dic[j1][:, :, :, None, :] * cM2convPsi_dic[j2][:, :, :, :, None] + \
               sM1convPsi_dic[j1][:, :, :, None, :] * sM2convPsi_dic[j2][:, :, :, :, None]  # Real [Nbatch, Npix_j3, Norient3, Norient2, Norient1]
        sc11 = sM1convPsi_dic[j1][:, :, :, None, :] * cM2convPsi_dic[j2][:, :, :, :, None] - \
               cM1convPsi_dic[j1][:, :, :, None, :] * sM2convPsi_dic[j2][:, :, :, :, None]  # Imag [Nbatch, Npix_j3, Norient3, Norient2, Norient1]
        ### Sum over pixels and apply the mask
        cc11 = self.bk_reduce_sum(vmask[None, :, :, None, None, None] *
                             cc11[:, None, :, :, :, :],
                             axis=2)  # Real [Nbatch, Nmask, Norient3, Norient2, Norient1]
        sc11 = self.bk_reduce_sum(vmask[None, :, :, None, None, None] *
                             sc11[:, None, :, :, :, :],
                             axis=2)  # Imag [Nbatch, Nmask, Norient3, Norient2, Norient1]
        return cc11, sc11

    def square(self,x):
        if x.S1 is None:
            return scat_cov(self.bk_abs(self.bk_square(x.P00)),
                            self.bk_abs(self.bk_square(x.C01)),
                            self.bk_abs(self.bk_square(x.C11)))
        else:
            return scat_cov(self.bk_abs(self.bk_square(x.P00)),
                            self.bk_abs(self.bk_square(x.C01)),
                            self.bk_abs(self.bk_square(x.C11)),
                            s1=self.bk_abs(self.bk_square(x.S1)))

    def reduce_mean(self,x):
        
        if x.S1 is None:
            result=self.bk_reduce_mean(x.P00) + \
                    self.bk_reduce_mean(x.C01) + \
                    self.bk_reduce_mean(x.C11)
        else:
            result=self.bk_reduce_mean(x.P00) + \
                    self.bk_reduce_mean(x.S1) + \
                    self.bk_reduce_mean(x.C01) + \
                    self.bk_reduce_mean(x.C11)
        return(result)

    def reduce_sum(self,x):
        
        if x.S1 is None:
            result=self.bk_reduce_sum(x.P00) + \
                    self.bk_reduce_sum(x.C01) + \
                    self.bk_reduce_sum(x.C11)
        else:
            result=self.bk_reduce_sum(x.P00) + \
                    self.bk_reduce_sum(x.S1) + \
                    self.bk_reduce_sum(x.C01) + \
                    self.bk_reduce_sum(x.C11)
        return(result)
    
    def log(self,x):
        
        if x.S1 is None:
            result= self.bk_log(x.P00) + \
                    self.bk_log(x.C01) + \
                    self.bk_log(x.C11)
        else:
            result= self.bk_log(x.P00) + \
                    self.bk_log(x.S1) + \
                    self.bk_log(x.C01) + \
                    self.bk_log(x.C11)
        return(result)

