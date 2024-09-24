import foscat.scat as scat


class scat2D:
    def __init__(self, p00, s0, s1, s2, s2l, j1, j2, cross=False, backend=None):
        the_scat = scat(p00, s0, s1, s2, s2l, j1, j2, cross=cross, backend=backend)
        the_scat.set_bk_type("SCAT2D")
        return the_scat

    def fill(self, im, nullval=0):
        return self.fill_2d(im, nullval=nullval)


class funct(scat.funct):
    def __init__(self, *args, **kwargs):
        # Impose que use_2D=True pour la classe scat
        super().__init__(use_2D=True, *args, **kwargs)
