import os
import sys

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

nitt = 5
scratch_path = "foscat_data"
outname = "TEST"


def dodown(a, nout):
    nin = int(np.sqrt(a.shape[0] // 12))
    if nin == nout:
        return a
    return np.mean(a.reshape(12 * nout * nout, (nin // nout) ** 2), 1)


plt.figure(figsize=(12, 12))
# reference
i1 = np.load(scratch_path + "/i%s1_%d.npy" % (outname, 0))
i2 = np.load(scratch_path + "/i%s2_%d.npy" % (outname, 0))
# mesure
c1 = np.load(scratch_path + "/c%s1_%d.npy" % (outname, 0))
c2 = np.load(scratch_path + "/c%s2_%d.npy" % (outname, 0))

plt.subplot(1, 2, 1)
plt.plot(abs(1 - i1.flatten() / i1.flatten()), color="blue")
plt.plot(abs(1 - c1.flatten() / i1.flatten()), color="orange")
plt.yscale("log")
plt.ylabel(r"$S_1$")
plt.xlabel("j1")
plt.subplot(1, 2, 2)
plt.plot(abs(1 - i2.flatten() / i2.flatten()), color="blue")
plt.plot(abs(1 - c2.flatten() / i2.flatten()), color="orange")
plt.yscale("log")
plt.ylabel(r"$S_2$")
plt.xlabel("j1,j2")

for itt in range(nitt):
    # foscat map
    o1 = np.load(scratch_path + "/o%s1_%d.npy" % (outname, itt))
    o2 = np.load(scratch_path + "/o%s2_%d.npy" % (outname, itt))

    plt.subplot(1, 2, 1)
    plt.plot(abs(1 - o1.flatten() / i1.flatten()), color="red", lw=1)
    plt.subplot(1, 2, 2)
    plt.plot(abs(1 - o2.flatten() / i2.flatten()), color="red", lw=1)

plt.subplot(1, 2, 1)
plt.plot(abs(1 - o1.flatten() / i1.flatten()), color="red", lw=4)
plt.subplot(1, 2, 2)
plt.plot(abs(1 - o2.flatten() / i2.flatten()), color="red", lw=4)

plt.subplot(1, 2, 1)
plt.plot(abs(1 - i1.flatten() / i1.flatten()), color="blue")
plt.plot(abs(1 - c1.flatten() / i1.flatten()), color="orange")
plt.yscale("log")
plt.ylabel(r"$S_1$")
plt.xlabel("j1")
plt.subplot(1, 2, 2)
plt.plot(abs(1 - i2.flatten() / i2.flatten()), color="blue")
plt.plot(abs(1 - c2.flatten() / i2.flatten()), color="orange")
plt.yscale("log")
plt.ylabel(r"$S_2$")
plt.xlabel("j1,j2")

d = np.load(scratch_path + "/test%sref.npy" % (outname))
d1 = np.load(scratch_path + "/test%sinput1.npy" % (outname))
d2 = np.load(scratch_path + "/test%sinput2.npy" % (outname))
di = np.load(scratch_path + "/test%sinput.npy" % (outname))
s = np.load(scratch_path + "/test%sresult_%d.npy" % (outname, nitt - 1)).flatten()

amp = 4

plt.figure(figsize=(12, 6))
hp.mollview(
    d,
    cmap="jet",
    nest=True,
    hold=False,
    sub=(2, 3, 1),
    min=-amp,
    max=amp,
    title="Model",
    norm="hist",
)
hp.mollview(
    di,
    cmap="jet",
    nest=True,
    hold=False,
    sub=(2, 3, 2),
    min=-amp,
    max=amp,
    title="Noisy",
    norm="hist",
)
hp.mollview(
    s,
    cmap="jet",
    nest=True,
    hold=False,
    sub=(2, 3, 3),
    min=-amp,
    max=amp,
    title="Cleanned",
    norm="hist",
)
hp.mollview(
    di - d,
    cmap="jet",
    nest=True,
    hold=False,
    sub=(2, 3, 4),
    min=-amp / 4,
    max=amp / 4,
    title="Noisy-Model",
)
hp.mollview(
    di - s,
    cmap="jet",
    nest=True,
    hold=False,
    sub=(2, 3, 5),
    min=-amp / 4,
    max=amp / 4,
    title="Noisy-Cleanned",
)
hp.mollview(
    s - d,
    cmap="jet",
    nest=True,
    hold=False,
    sub=(2, 3, 6),
    min=-amp / 4,
    max=amp / 4,
    title="Cleanned-Model",
)

nin = 512
nout = int(np.sqrt(d.shape[0] // 12))
idx = hp.ring2nest(nout, np.arange(12 * nout**2))

mask = dodown(np.load(scratch_path + "/MASK_GAL080_%d.npy" % (nin)), nout)
clr = hp.anafast((mask * d - np.median(mask * d))[idx])
cli = hp.anafast((mask * di - np.median(mask * di))[idx])
cln = hp.anafast((mask * (d - di) - np.median(mask * (d - di)))[idx])

plt.figure(figsize=(12, 12))

for itt in range(nitt):
    s = np.load(scratch_path + "/test%sresult_%d.npy" % (outname, itt)).flatten()

    clo = hp.anafast((mask * s - np.median(mask * s))[idx])
    cld = hp.anafast((mask * (d - s) - np.median(mask * (d - s)))[idx])

    plt.plot(clo, color="orange", label="s %d" % (itt), lw=1)
    plt.plot(cld, color="red", label="r-s %d" % (itt), lw=1)

plt.plot(clo, color="orange", label="s %d" % (itt), lw=4)
plt.plot(cld, color="red", label="r-s %d" % (itt), lw=4)

plt.plot(clr, color="black", label="r")
plt.plot(cli, color="blue", label="d")
plt.plot(cln, color="grey", label="r-d")

plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.show()
