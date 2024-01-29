import foscat.scat_cov as scat

class scat_cov2D:
    def __init__(self,p00,s0,s1,s2,s2l,j1,j2,cross=False,backend=None):

        the_scat=scat(P00, C01, C11, s1=S1, c10=C10,backend=self.backend)
        the_scat.set_bk_type('SCAT_COV2D')
        return the_scat
    
    def fill(self,im,nullval=0):
        return self.fill_2d(im,nullval=nullval)

class funct(scat.funct):
    def __init__(self, *args, **kwargs):
        # Impose que use_2D=True pour la classe scat
        super(funct, self).__init__(use_2D=True, *args, **kwargs)
