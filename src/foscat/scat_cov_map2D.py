import foscat.scat_cov as scat


class scat_cov_map(scat.funct):
    def __init__(self, S2, S0, S3, S4, S1=None, S3P=None, backend=None):

        self.S2 = S2
        self.S0 = S0
        self.S1 = S1
        self.S3 = S3
        self.S3P = S3P
        self.S4 = S4
        self.backend = backend
        self.bk_type = "SCAT_COV_MAP2D"

    def fill(self, im, nullval=0):
        return self.backend.fill_2d(im, nullval=nullval)


class funct(scat.funct):
    def __init__(self, *args, **kwargs):
        # Impose que use_2D=True pour la classe scat
        super().__init__(use_2D=True, return_data=True, *args, **kwargs)

    def eval(
        self, image1, image2=None, mask=None, norm=None, Auto=True, calc_var=False
    ):
        r = super().eval(
            image1, image2=image2, mask=mask, norm=norm, Auto=Auto, calc_var=calc_var
        )
        return scat_cov_map(
            r.S2, r.S0, r.S3, r.S4, S1=r.S1, S3P=r.S3P, backend=r.backend
        )

    def scat_coeffs_apply(
        self, scat, method, no_order_1=False, no_order_2=False, no_order_3=False
    ):
        for j in scat.S2:
            if not no_order_1:
                scat.S2[j] = method(scat.S2[j])
                if scat.S1 is not None:
                    scat.S1[j] = method(scat.S1[j])

            if not no_order_2:
                for n1 in scat.S3[j]:
                    scat.S3[j][n1] = method(scat.S3[j][n1])

                if scat.S3P is not None:
                    for n1 in scat.S3P[j]:
                        scat.S3P[j][n1] = method(scat.S3P[j][n1])

            if not no_order_3:
                for n1 in scat.S4[j]:
                    for n2 in scat.S4[j][n1]:
                        scat.S4[j][n1][n2] = method(scat.S4[j][n1][n2])

    def scat_ud_grade_2(
        self, scat, no_order_1=False, no_order_2=False, no_order_3=False
    ):
        self.scat_coeffs_apply(
            scat,
            lambda x: self.ud_grade_2(x, axis=1),
            no_order_1=no_order_1,
            no_order_2=no_order_2,
            no_order_3=no_order_3,
        )

    def iso_mean(self, scat, no_order_1=False, no_order_2=False, no_order_3=False):
        self.scat_coeffs_apply(
            scat,
            lambda x: self.backend.iso_mean(x, use_2D=True),
            no_order_1=no_order_1,
            no_order_2=no_order_2,
            no_order_3=no_order_3,
        )

    def fft_ang(
        self,
        scat,
        nharm=1,
        imaginary=False,
        no_order_1=False,
        no_order_2=False,
        no_order_3=False,
    ):
        self.scat_coeffs_apply(
            scat,
            lambda x: self.backend.fft_ang(
                x, use_2D=True, nharm=nharm, imaginary=imaginary
            ),
            no_order_1=no_order_1,
            no_order_2=no_order_2,
            no_order_3=no_order_3,
        )
