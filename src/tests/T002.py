import numpy as np

# test iso_mean function
nside = 32
im = np.random.randn(12 * nside**2)

# test scat
import foscat.scat as sc

op = sc.funct()

a = op.eval(im)
b = a.iso_mean()
c = a.iso_std()

idx = np.arange(4, dtype="int")

test = 0.0
test2 = 0.0
for k in range(4):
    test = (
        test + b.S2.numpy()[0, 4, k] - np.mean(a.S2.numpy()[0, 4, idx, (idx + k) % 4])
    )
    test2 = (
        test2 + c.S2.numpy()[0, 4, k] - np.std(a.S2.numpy()[0, 4, idx, (idx + k) % 4])
    )
print(test, test2)


# test scat_cov
import foscat.scat_cov as sc

op = sc.funct()

a = op.eval(im)
b = a.iso_mean()
c = a.iso_std()

idx = np.arange(4, dtype="int")

test = 0.0
tes2 = 0.0
test3 = 0.0
test4 = 0.0
for k in range(4):
    test = (
        test
        + b.C01.numpy()[0, 0, 4, k]
        - np.mean(a.C01.numpy()[0, 0, 4, idx, (idx + k) % 4])
    )
    x = a.C01.numpy()[0, 0, 4, idx, (idx + k) % 4]
    test2 = test2 + c.C01.numpy()[0, 0, 4, k] - np.std(x.real) - 1j * np.std(x.imag)
    test3 = (
        test3
        + b.C11.numpy()[0, 0, 4, 0, k]
        - np.mean(a.C11.numpy()[0, 0, 4, 0, idx, (idx + k) % 4])
    )
    x = a.C11.numpy()[0, 0, 4, 0, idx, (idx + k) % 4]
    test4 = test2 + c.C11.numpy()[0, 0, 4, 0, k] - np.std(x.real) - 1j * np.std(x.imag)

print(test, test2, test3, test4)
