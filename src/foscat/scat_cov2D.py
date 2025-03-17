import numpy as np

import foscat.scat_cov as scat


class scat_cov2D:
    def __init__(self, s0, s2, s3, s4, s1=None, s3p=None, backend=None):

        the_scat = scat(s0, s2, s3, s4, s1=s1, s3p=s3p, backend=self.backend)
        the_scat.set_bk_type("SCAT_COV2D")
        return the_scat

    def fill(self, im, nullval=0):
        return self.fill_2d(im, nullval=nullval)


class funct(scat.funct):
    def __init__(self, *args, **kwargs):
        # Impose que use_2D=True pour la classe scat
        super().__init__(use_2D=True, KERNELSZ=5, *args, **kwargs)

    def spectrum(self, image):
        """
        Compute the 1D power spectrum of a 2D image by averaging the 2D power spectrum
        over concentric frequency rings (radial averaging), using np.bincount for efficiency.

        Parameters:
        - image : ndarray (2D), input image

        Returns:
        - freq : radial frequencies
        - spectrum_1d : corresponding 1D power spectrum
        """
        import numpy as np

        # Compute the 2D Fourier Transform and shift the zero frequency to the center
        F = np.fft.fftshift(np.fft.fft2(image))
        power_spectrum = np.abs(F) ** 2

        # Create coordinate grids and compute the radial distance from the center
        y, x = np.indices(power_spectrum.shape)
        center = np.array(power_spectrum.shape) // 2
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)

        # Use np.bincount for fast summation and counting
        sum_power = np.bincount(r.ravel(), weights=power_spectrum.ravel())
        counts = np.bincount(r.ravel())

        # Compute the mean power for each radial bin
        spectrum_1d = sum_power / counts

        return spectrum_1d

    def plot_results(
        self,
        in_image,
        out_image,
        vmin=None,
        vmax=None,
        cmap="coolwarm",
        spec_range=None,
    ):
        import matplotlib.pyplot as plt

        if len(out_image.shape) > 2:
            nimage = out_image.shape[0]
            ndraw = np.min([3, nimage])
            plt.figure(figsize=(16, 12))
            plt.subplot(2, ndraw + 1, 1)
            plt.title("Original field")
            plt.imshow(in_image, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            plt.xticks([])
            plt.yticks([])
            for k in range(ndraw):
                plt.subplot(2, ndraw + 1, 2 + k)
                plt.title("Modeled field #%d" % (k))
                plt.imshow(
                    out_image[k], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower"
                )
                plt.xticks([])
                plt.yticks([])
            plt.subplot(2, 2, 3)
            plt.title("Histogram")
            for k in range(nimage):
                if k == 0:
                    plt.hist(
                        out_image[k].flatten(),
                        bins=100,
                        label="modeled",
                        color="b",
                        histtype="step",
                        log=True,
                        alpha=0.5,
                    )
                else:
                    plt.hist(
                        out_image[k].flatten(),
                        bins=100,
                        color="b",
                        histtype="step",
                        log=True,
                        alpha=0.5,
                    )
            plt.hist(
                in_image.flatten(),
                bins=100,
                label="original",
                color="r",
                histtype="step",
                log=True,
            )
            plt.legend(frameon=0)
            plt.subplot(2, 2, 4)
            plt.title("Powerspectra")
            for k in range(nimage):
                if k == 0:
                    plt.plot(
                        self.spectrum(out_image[k]),
                        color="b",
                        label="modeled",
                        alpha=0.5,
                    )
                else:
                    plt.plot(self.spectrum(out_image[k]), color="b", alpha=0.5)
            plt.plot(self.spectrum(in_image), color="r", label="original")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend(frameon=0)
            if spec_range is not None:
                plt.ylim(spec_range[0], spec_range[1])
        else:
            plt.figure(figsize=(16, 3))
            plt.subplot(1, 4, 1)
            plt.title("Original field")
            plt.imshow(in_image, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 4, 2)
            plt.title("Modeled field")
            plt.imshow(out_image, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 4, 3)
            plt.title("Histogram")
            plt.hist(
                in_image.flatten(),
                bins=100,
                label="original",
                color="r",
                histtype="step",
                log=True,
            )
            plt.hist(
                out_image.flatten(),
                bins=100,
                label="modeled",
                color="b",
                histtype="step",
                log=True,
            )
            plt.legend(frameon=0)
            plt.subplot(1, 4, 4)
            plt.title("Powerspectra")
            plt.plot(self.spectrum(in_image), color="b", label="original")
            plt.plot(self.spectrum(out_image), color="r", label="modeled")
            plt.xscale("log")
            plt.yscale("log")
            if spec_range is not None:
                plt.ylim(spec_range[0], spec_range[1])
            plt.legend(frameon=0)

    def plot_results(self, in_image, out_image, vmin=None, vmax=None, cmap="coolwarm"):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(16, 3))
        plt.subplot(1, 4, 1)
        plt.title("Original field")
        plt.imshow(in_image, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 4, 2)
        plt.title("Modeled field")
        plt.imshow(out_image, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 4, 3)
        plt.title("Histogram")
        plt.hist(
            in_image.flatten(),
            bins=100,
            label="original",
            color="r",
            histtype="step",
            log=True,
        )
        plt.hist(
            out_image.flatten(),
            bins=100,
            label="modeled",
            color="b",
            histtype="step",
            log=True,
        )
        plt.legend(frameon=0)
        plt.subplot(1, 4, 4)
        plt.title("Powerspectra")
        plt.plot(self.spectrum(in_image), color="b", label="original")
        plt.plot(self.spectrum(out_image), color="r", label="modeled")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(frameon=0)
