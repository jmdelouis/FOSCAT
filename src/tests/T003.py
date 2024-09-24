import numpy as np

# test iso_mean function
nside = 32
im = np.random.randn(12 * nside**2)

# test scat
import foscat.scat as sc

op = sc.funct()

a = op.eval(im)
b = a.fft_ang()

idx = np.arange(4, dtype="int")

test = 0.0
# print(b.S1.numpy()[0,4],np.array([np.sum(a.S1.numpy()[0,4]),a.S1.numpy()[0,4,0]-a.S1.numpy()[0,4,2]]))
x = a.S2.numpy()[0, 4]
xl = a.S2L.numpy()[0, 4]
# print(b.S2.numpy()[0,4].flatten(),np.array([np.sum(x),
#                                  np.sum(x[:,0])-np.sum(x[:,2]),
#                                  np.sum(x[0,:])-np.sum(x[2,:]),
#                                  x[0,0]+x[2,2]-x[2,0]-x[0,2]]))
test = (
    np.sum(
        b.S1.numpy()[0, 4]
        - np.array(
            [np.sum(a.S1.numpy()[0, 4]), a.S1.numpy()[0, 4, 0] - a.S1.numpy()[0, 4, 2]]
        )
    )
    + np.sum(
        b.S2.numpy()[0, 4].flatten()
        - np.array(
            [
                np.sum(x),
                np.sum(x[:, 0]) - np.sum(x[:, 2]),
                np.sum(x[0, :]) - np.sum(x[2, :]),
                x[0, 0] + x[2, 2] - x[2, 0] - x[0, 2],
            ]
        )
    )
    + np.sum(
        b.P00.numpy()[0, 4]
        - np.array(
            [
                np.sum(a.P00.numpy()[0, 4]),
                a.P00.numpy()[0, 4, 0] - a.P00.numpy()[0, 4, 2],
            ]
        )
    )
    + np.sum(
        b.S2L.numpy()[0, 4].flatten()
        - np.array(
            [
                np.sum(xl),
                np.sum(xl[:, 0]) - np.sum(xl[:, 2]),
                np.sum(xl[0, :]) - np.sum(xl[2, :]),
                xl[0, 0] + xl[2, 2] - xl[2, 0] - xl[0, 2],
            ]
        )
    )
)
print(test)

test = 0.0
# test scat_cov
import foscat.scat_cov as sc

op = sc.funct()

a = op.eval(im)
b = a.fft_ang()
x = a.C01.numpy()[0, 0, 4]
print(a.C11.shape, b.C11.shape)
xl = a.C11.numpy()[0, 0, 4]
# print(b.S2.numpy()[0,4].flatten(),np.array([np.sum(x),
#                                  np.sum(x[:,0])-np.sum(x[:,2]),
#                                  np.sum(x[0,:])-np.sum(x[2,:]),
#                                  x[0,0]+x[2,2]-x[2,0]-x[0,2]]))

test = (
    np.sum(
        b.S1.numpy()[0, 0, 4]
        - np.array(
            [
                np.sum(a.S1.numpy()[0, 0, 4]),
                a.S1.numpy()[0, 0, 4, 0] - a.S1.numpy()[0, 0, 4, 2],
            ]
        )
    )
    + np.sum(
        b.C01.numpy()[0, 0, 4].flatten()
        - np.array(
            [
                np.sum(x),
                np.sum(x[:, 0]) - np.sum(x[:, 2]),
                np.sum(x[0, :]) - np.sum(x[2, :]),
                x[0, 0] + x[2, 2] - x[2, 0] - x[0, 2],
            ]
        )
    )
    + np.sum(
        b.P00.numpy()[0, 0, 4]
        - np.array(
            [
                np.sum(a.P00.numpy()[0, 0, 4]),
                a.P00.numpy()[0, 0, 4, 0] - a.P00.numpy()[0, 0, 4, 2],
            ]
        )
    )
)


print(test)
