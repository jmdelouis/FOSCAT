import foscat.scat_cov as scat
import healpy as hp
class scat_cov_map:
    def __init__(self,p00,s0,s1,s2,s2l,j1,j2,cross=False,backend=None):

        the_scat=scat(P00, C01, C11, s1=S1, c10=C10,backend=self.backend)
        the_scat.set_bk_type('SCAT_COV_MAP')
        return the_scat
    
    def fill(self,im,nullval=hp.UNSEEN):
        return self.fill_healpy(im,nullval=nullval)
    
    def scat_coeffs_apply(self, method):
        for j in self.P00:
            #self.P00[j] = np.array([ scat.ud_grade_2(r) for r in self.P00[j] ])
            self.P00[j] = method(self.P00[j])
            self.S1[j] = method(self.S1[j])
            
            for n1 in self.C01[j]:
                self.C01[j][n1] = method(self.C01[j][n1])

            for n1 in self.C11[j]:
                for n2 in self.C11[j][n1]:
                    self.C11[j][n1][n2] = method(self.C11[j][n1][n2])

   def ud_grade_2(self, scat):
       self.scat_coeffs_apply(lambda x: scat.ud_grade_2(x, axis=1))
       
class funct(scat.funct):
    def __init__(self, *args, **kwargs):
        # Impose return_data=True pour la classe scat
        super(funct, self).__init__(return_data=True, *args, **kwargs)
