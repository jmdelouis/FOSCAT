import numpy as np
import matplotlib.pyplot as plt

outname = "test2D"

scratch_path = "foscat_data"
r1 = np.load(scratch_path + "/%s_r1.npy" % (outname))
r2 = np.load(scratch_path + "/%s_r2.npy" % (outname))
o1 = np.load(scratch_path + "/%s_o1.npy" % (outname))
o2 = np.load(scratch_path + "/%s_o2.npy" % (outname))

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(r1.flatten(), color="blue", label="input")
plt.plot(o1.flatten(), color="red", label="synthesised")
plt.yscale("log")
plt.ylabel(r"$S_1$")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(r2.flatten(), color="blue", label="input")
plt.plot(o2.flatten(), color="red", label="synthesised")
plt.yscale("log")
plt.ylabel(r"$S_2$")

ref = np.load(scratch_path + "/%s_ref.npy" % (outname))
omap = np.load(scratch_path + "/%s_result.npy" % (outname))

amp = ref.max()
nx = ref.shape[0]
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(ref, cmap="jet", origin="lower", vmin=-amp, vmax=amp)
plt.title("Input")
plt.subplot(1, 2, 2)
plt.imshow(omap, cmap="jet", origin="lower", vmin=-amp, vmax=amp)
plt.title("FoSCAT")
plt.show()
