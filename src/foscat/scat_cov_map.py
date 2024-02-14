import foscat.scat_cov as scat
import healpy as hp

import foscat.scat_cov as scat

class scat_cov_map(scat.funct):
    def __init__(self,P00,S0,C01,C11,S1=None,C10=None,backend=None):

        self.P00=P00
        self.S0=S0
        self.S1=S1
        self.C01=C01
        self.C10=C10
        self.C11=C11
        self.backend=backend
        self.bk_type='SCAT_COV_MAP2D'
    
    def fill(self,im,nullval=hp.UNSEEN):
        return self.fill_healpy(im,nullval=nullval)

    
class funct(scat.funct):
    def __init__(self, *args, **kwargs):
        super(funct, self).__init__(return_data=True, *args, **kwargs)

    def eval(self, image1, image2=None, mask=None, norm=None, Auto=True, calc_var=False):
        r=super(funct, self).eval(image1, image2=image2, mask=mask, norm=norm, Auto=Auto, calc_var=calc_var)
        return scat_cov_map(r.P00,r.S0,r.C01,r.C11,S1=r.S1,C10=r.C10,backend=r.backend)

    def scat_coeffs_apply(self, scat, method, no_order_1=False,no_order_2=False,no_order_3=False):
        for j in scat.P00:
            if no_order_1==False:
                scat.P00[j] = method(scat.P00[j])
                if scat.S1 is not None:
                    scat.S1[j] = method(scat.S1[j])
            
            if no_order_2==False:
                for n1 in scat.C01[j]:
                    scat.C01[j][n1] = method(scat.C01[j][n1])

                if scat.C10 is not None:
                    for n1 in scat.C10[j]:
                        scat.C10[j][n1] = method(scat.C10[j][n1])

            if no_order_3==False:
                for n1 in scat.C11[j]:
                    for n2 in scat.C11[j][n1]:
                        scat.C11[j][n1][n2] = method(scat.C11[j][n1][n2])

    def scat_ud_grade_2(self,scat,no_order_1=False,no_order_2=False,no_order_3=False):
        self.scat_coeffs_apply(scat,lambda x: self.ud_grade_2(x, axis=1),
                               no_order_1=no_order_1,
                               no_order_2=no_order_2,
                               no_order_3=no_order_3)
       
       
    def iso_mean(self,scat,no_order_1=False,no_order_2=False,no_order_3=False):
       self.scat_coeffs_apply(scat,lambda x: self.backend.iso_mean(x),
                               no_order_1=no_order_1,
                               no_order_2=no_order_2,
                               no_order_3=no_order_3)
       
    def fft_ang(self,scat,nharm=1,imaginary=False,no_order_1=False,no_order_2=False,no_order_3=False):
       self.scat_coeffs_apply(scat,lambda x: self.backend.fft_ang(x,nharm=nharm,imaginary=imaginary),
                               no_order_1=no_order_1,
                               no_order_2=no_order_2,
                               no_order_3=no_order_3)
