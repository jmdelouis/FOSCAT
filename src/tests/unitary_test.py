# =============================================================================
# create directory to store all test
# =============================================================================
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import foscat.Synthesis as synthe

try:
    os.system("rm -rf UNITARY")
except:
    print("UNIATRY directory does not exist, create it")

os.system("mkdir UNITARY")


scratch_path = "UNITARY"


# =================================================================================
# Function to reduce the data used in the FoCUS algorithm
# =================================================================================
def dodown(a, nside):
    nin = int(np.sqrt(a.shape[0] // 12))
    if nin == nside:
        return a
    return np.mean(a.reshape(12 * nside * nside, (nin // nside) ** 2), 1)


# =============================================================================
# Unitary test for scat
# =============================================================================


tabtype = ["float32", "float64"]

nside = 16
data = "Venus_256.npy"


def lossX(x, scat_operator, args):

    ref = args[0]
    im = args[1]
    mask = args[2]
    docross = args[3]

    if docross:
        print("Loss ", im, x)
        learn = scat_operator.eval(im, image2=x, Auto=False, mask=mask)
    else:
        learn = scat_operator.eval(x, mask=mask)

    loss = scat_operator.reduce_sum(scat_operator.square(ref - learn))

    return loss


for itest in range(len(tabtype)):
    for k in range(6):
        if k == 2:
            import foscat.scat as sc

            ishape = [12 * 16 * 16]
            ims = np.random.randn(12 * 16 * 16)
            mask = (np.random.rand(3, ims.shape[0]) > 0.5).astype("float")
        elif k == 1:
            import foscat.scat_cov as sc

            ishape = [12 * 16 * 16]
            ims = np.random.randn(12 * 16 * 16)
            mask = (np.random.rand(3, ims.shape[0]) > 0.5).astype("float")
        elif k == 0:
            import foscat.scat2D as sc

            ishape = [12, 16]
            ims = np.random.randn(12, 16)
            mask = (np.random.rand(3, ims.shape[0], ims.shape[1]) > 0.5).astype("float")
        elif k == 3:
            import foscat.scat_cov2D as sc

            ishape = [12, 16]
            ims = np.random.randn(12, 16)
            mask = (np.random.rand(3, ims.shape[0], ims.shape[1]) > 0.5).astype("float")
        elif k == 4:
            import foscat.scat1D as sc

            ishape = [12 * 16]
            ims = np.random.randn(12 * 16)
            mask = (np.random.rand(3, ims.shape[0]) > 0.5).astype("float")
        elif k == 5:
            import foscat.scat_cov1D as sc

            ishape = [12 * 16]
            ims = np.random.randn(12 * 16)
            mask = (np.random.rand(3, ims.shape[0]) > 0.5).astype("float")

        scat_op = sc.funct(
            NORIENT=4,  # define the number of wavelet orientation
            KERNELSZ=5,  # KERNELSZ,  # define the kernel size
            OSTEP=0,  # get very large scale (nside=1)
            LAMBDA=1.2,
            TEMPLATE_PATH=scratch_path,
            slope=1.0,
            gpupos=0,
            all_type=tabtype[itest],
        )

        print("Start Test Synthesis no cross 0 : Type=%s ...." % (tabtype[itest]))
        ref = scat_op.eval(ims, mask=mask)
        loss1 = synthe.Loss(lossX, scat_op, ref, ims, mask, False)
        sy = synthe.Synthesis([loss1])

        if len(ims.shape) == 1:
            imap = np.random.randn(ims.shape[0])
        else:
            imap = np.random.randn(ims.shape[0], ims.shape[1])

        omap = sy.run(imap, NUM_EPOCHS=10)

        print(
            "Test Synthesis no cross 0 : Type=%s %s DONE"
            % (tabtype[itest], sc.__name__)
        )

        print("Start Test Synthesis cross :Type=%s ...." % (tabtype[itest]))

        ref = scat_op.eval(ims, image2=ims, mask=mask, Auto=False)
        loss1 = synthe.Loss(lossX, scat_op, ref, ims, mask, True)
        sy = synthe.Synthesis([loss1])

        if len(ims.shape) == 1:
            imap = np.random.randn(ims.shape[0])
        else:
            imap = np.random.randn(ims.shape[0], ims.shape[1])

        omap = sy.run(imap, NUM_EPOCHS=10)

        print("Test Synthesis cross : Type=%s %s DONE" % (tabtype[itest], sc.__name__))

        print("Start Test :  Type=%s ...." % (tabtype[itest]))

        if len(ims.shape) == 1:
            im = np.random.randn(ims.shape[0])
        else:
            im = np.random.randn(ims.shape[0], ims.shape[1])

        a = scat_op.convol(im)

        r1 = scat_op.eval(im)
        r2 = scat_op.eval(im, image2=im)
        r3 = scat_op.eval(im, mask=mask)
        r = r1 * r2
        r = r1 / r2
        r = r1 - r2
        r = 3 * r2
        r = 3 / r2
        r = 3 - r2
        r = r1 * 3
        r = r1 / 3
        r = r1 - 3

        if len(ims.shape) == 1:
            im = np.random.randn(ims.shape[0]) + complex(0, 1) * np.random.randn(
                ims.shape[0]
            )
        else:
            im = np.random.randn(ims.shape[0], ims.shape[1]) + complex(
                0, 1
            ) * np.random.randn(ims.shape[0], ims.shape[1])

        scat_op = sc.funct(
            NORIENT=4,  # define the number of wavelet orientation
            KERNELSZ=5,  # KERNELSZ,  # define the kernel size
            OSTEP=-1,  # get very large scalSe (nside=1)
            LAMBDA=1.2,
            TEMPLATE_PATH=scratch_path,
            slope=1.0,
            gpupos=0,
            all_type=tabtype[itest],
        )

        print(
            "Start Test COMPLEX Synthesis no cross : Type=%s %s ...."
            % (tabtype[itest], sc.__name__)
        )

        ref = scat_op.eval(im, mask=mask)
        loss1 = synthe.Loss(lossX, scat_op, ref, im, mask, False)
        sy = synthe.Synthesis([loss1])

        if len(ims.shape) == 1:
            imap = np.random.randn(ims.shape[0]) + complex(0, 1) * np.random.randn(
                ims.shape[0]
            )
        else:
            imap = np.random.randn(ims.shape[0], ims.shape[1]) + complex(
                0, 1
            ) * np.random.randn(ims.shape[0], ims.shape[1])

        omap = sy.run(imap, NUM_EPOCHS=10)

        print(
            "Test Synthesis COMPLEX no cross : Type=%s %s DONE"
            % (tabtype[itest], sc.__name__)
        )

        print("Start Test Synthesis cross :Type=%s ...." % (tabtype[itest]))

        ref = scat_op.eval(im, image2=im, mask=mask, Auto=False)
        loss1 = synthe.Loss(lossX, scat_op, ref, im, mask, True)
        sy = synthe.Synthesis([loss1])

        if len(ims.shape) == 1:
            imap = np.random.randn(ims.shape[0]) + complex(0, 1) * np.random.randn(
                ims.shape[0]
            )
        else:
            imap = np.random.randn(ims.shape[0], ims.shape[1]) + complex(
                0, 1
            ) * np.random.randn(ims.shape[0], ims.shape[1])

        """
        omap=sy.run(imap,NUM_EPOCHS = 10) # LBFGS does not know how to manage complex minimisation
        """
        print("Test Synthesis cross : Type=%s %s DONE" % (tabtype[itest], sc.__name__))

        a = scat_op.convol(im)

        r1 = scat_op.eval(im)
        r2 = scat_op.eval(im, image2=im)
        r3 = scat_op.eval(im, mask=mask)
        r = r1 * r2
        r = r1 / r2
        r = r1 - r2
        r = r1 + r2
        r = 3 * r2
        r = 3 / r2
        r = 3 - r2
        r = 3 + r2
        r = r1 * 3
        r = r1 / 3
        r = r1 - 3
        r = r1 + 3

        print("Test : Type=%s OK" % (tabtype[itest]))
