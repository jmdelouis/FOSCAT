import foscat.scat_cov as scat


class scat_cov2D:
    def __init__(self, s0, p00, c01, c11, s1=None, c10=None, backend=None):

        the_scat = scat(s0, p00, c01, c11, s1=s1, c10=c10, backend=self.backend)
        the_scat.set_bk_type("SCAT_COV2D")
        return the_scat

    def fill(self, im, nullval=0):
        return self.fill_2d(im, nullval=nullval)


class funct(scat.funct):
    def __init__(self, *args, **kwargs):
        # Impose que use_2D=True pour la classe scat
        super().__init__(use_2D=True, *args, **kwargs)
