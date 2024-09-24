import numpy as np
import os, sys
import matplotlib.pyplot as plt
import foscat.FoCUS as FOC


def conjugate(mat, vec, nitt=10):
    x = np.zeros([vec.shape[0]])
    r = vec - np.dot(mat, x)
    p = r
    for itt in range(nitt):
        Ap = np.dot(mat, p)
        delta = np.sum(r * r)
        if itt % 10 == 9:
            print(itt, delta)
        alpha = delta / np.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        beta = np.sum(r * r) / delta
        p = r + beta * p
    return x


# =================================================================================
# INITIALIZE foscat class
# =================================================================================
#  Change the temporary path
# =================================================================================
scratch_path = "foscat_data"
outname = "test2D"

# =================================================================================
# This test could be run till XSIZE=512 but could last for Hours if not GPUs are
# available. XSIZE=128 takes few minutes with NVIDIA T4.
# =================================================================================
XSIZE = 64

# =================================================================================
# ==   Make the statistic invariant by rotation => Higher dimension reduction    ==
# ==   avg_ang = True                                                            ==
# =================================================================================
avg_ang = False

fc = FOC.FoCUS(
    healpix=False, NORIENT=4, KERNELSZ=3, OSTEP=0, TEMPLATE_PATH=scratch_path, slope=2
)

# =================================================================================
#  READ data and get data if necessary:
# =================================================================================
#  Here the input data is a vorticity map of 512x512:
# This image shows a simulated snapshot of ocean turbulence in the North Atlantic Ocean in March 2012,
# from a groundbreaking super-high-resolution global ocean simulation (approximately 1.2 miles,
# or 2 kilometers, horizontal resolution) developed at JPL.
# (http://wwwcvs.mitgcm.org/viewvc/MITgcm/MITgcm_contrib/llc_hires/llc_4320/).
# =================================================================================
try:
    d = np.load(scratch_path + "/Vorticity.npy")
except:
    import imageio as iio

    os.system(
        "wget -O "
        + scratch_path
        + "/PIA22256.tif https://photojournal.jpl.nasa.gov/tiff/PIA22256.tif"
    )

    im = iio.imread(scratch_path + "/PIA22256.tif")
    im = im[1000:1512, 2000:2512, 0] / 255.0 - im[1000:1512, 2000:2512, 2] / 255.0
    np.save(scratch_path + "/Vorticity.npy", im)
    os.system("rm " + scratch_path + "/PIA22256.tif")
    d = np.load(scratch_path + "/Vorticity.npy")


d = d[0:XSIZE, 0:XSIZE]
nx = d.shape[0]
# define the level of noise of the simulation
ampnoise = 1.0

# =================================================================================
# Synthesise data with the same cross statistic than the input data
# =================================================================================

# convert data in tensor for Foscat
idata = fc.convimage(d)

# define the mask where the statistic are used
x = np.repeat((np.arange(XSIZE) - XSIZE / 2) / XSIZE, XSIZE).reshape(XSIZE, XSIZE)
mask = np.exp(-32 * (x**4 + (x.T) ** 4))
mask[:, :] = 1.0
# mask[32,33]=1.0
fc.add_mask(mask.reshape(1, nx, nx))

# Initialize the learning and initialize the tensor to be synthesized
randfield = np.random.randn(nx, nx)
# randfield[:,:]=0.0
# randfield[32:34,32]=1.0
ldata = fc.init_synthese(randfield)

# Build the loss:
# here Loss += (d x d - s x s - tb[0]).tw[0]
fc.add_loss_2d(idata, idata, ldata, ldata, avg_ang=avg_ang)

# initiliaze the synthesise process
loss = fc.init_optim()

tw1 = {}
tw2 = {}
tb1 = {}
tb2 = {}
# compute the weights and the bias for each loss
modd = d.reshape(1, nx, nx)
r1, r2 = fc.calc_stat(modd, modd, avg_ang=avg_ang)

tw1[0] = 1.0 + 0.0 * r1[0]
tw2[0] = 1.0 + 0.0 * r2[0]
tb1[0] = 0.0 * r1[0]
tb2[0] = 0.0 * r2[0]

# save the reference statistics
np.save(scratch_path + "/%s_r1.npy" % (outname), r1)
np.save(scratch_path + "/%s_r2.npy" % (outname), r2)
feed_dict = {}
feed_dict[fc.tw1[0]] = tw1[0]
feed_dict[fc.tw2[0]] = tw2[0]
feed_dict[fc.tb1[0]] = tb1[0]
feed_dict[fc.tb2[0]] = tb2[0]

sr1, sr2 = fc.cwst1_comp(idata, idata)
s1, s2 = fc.cwst1_comp(ldata, ldata)

r1, r2 = fc.sess.run([sr1, sr2], feed_dict=feed_dict)
rr1n, rr2n = fc.sess.run([s1, s2], feed_dict=feed_dict)

print("initialize grad 1")
ngrd1 = 5
g2 = {}
for i in range(4):
    for j in range(5):
        g2[i + 4 * j] = fc.opti.compute_gradients(s1[j][0, i], var_list=[fc.param[0]])

print("initialize grad 2")
ngrad = 14
g22 = {}
for i in range(16):
    for j in range(ngrad):
        g22[i + 16 * j] = fc.opti.compute_gradients(
            s2[j][0, i // 4, i % 4], var_list=[fc.param[0]]
        )
print("initialize grad Done")

xidx = {}
yidx = {}
for i in range(5):
    xidx[i] = (np.arange(64 * 64, dtype="int") // 64) // (2**i)
    yidx[i] = (np.arange(64 * 64, dtype="int") % 64) // (2**i)

for itt in range(10000):

    rr1, rr2 = fc.sess.run([s1, s2], feed_dict=feed_dict)

    gg1 = fc.sess.run(g2, feed_dict=feed_dict)
    gg2 = fc.sess.run(g22, feed_dict=feed_dict)

    diff = np.zeros([1, 64, 64, 1])

    for i in range(5):
        for j in range(4):
            tmp = (rr1[i][0, j] - r1[i][0, j]) * (
                gg1[i * 4 + j][0][0].reshape(64, 64) / (64 * 64)
            )
            diff[0, :, :, 0] += tmp / 64  # (tmp[xidx[i],yidx[i]].reshape(64,64)/64

    for i in range(ngrad):
        for j in range(16):
            tmp = (
                (rr2[i][0, j // 4, j % 4] - r2[i][0, j // 4, j % 4])
                * (gg2[i * 16 + j][0][0].reshape(64, 64))
                / (64 * 64)
            )
            diff[0, :, :, 0] += tmp / 2048  # (tmp[xidx[i],yidx[i]].reshape(64,64))/64

    par = fc.get_param()
    fc.set_value(par - (diff).flatten(), 0)
    if itt % 100 == 0:
        var1 = 0
        for i in range(5):
            var1 += ((r1[i] - rr1[i]).std()) ** 2
        var2 = 0
        for i in range(ngrad):
            var2 += ((r2[i] - rr2[i]).std()) ** 2
        print(itt, np.sqrt(var1), np.sqrt(var2))
        """
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(rr1n.flatten(),color='yellow',lw=8)
        plt.plot(r1.flatten(),color='black')
        plt.plot(rr1.flatten(),color='blue')
        plt.yscale('log')
        plt.subplot(1,2,2)
        plt.plot(rr2n.flatten(),color='yellow',lw=8)
        plt.plot(r2.flatten(),color='black')
        plt.plot(rr2.flatten(),color='blue')
        plt.yscale('log')
        plt.show()
        """
imap = fc.get_map()
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(d.reshape(64, 64), cmap="jet")
plt.subplot(1, 2, 2)
plt.imshow(imap.reshape(64, 64), cmap="jet")
plt.show()
exit(0)

exit(0)
gr1 = fc.sess.run(g1, feed_dict=feed_dict)
gc1 = fc.sess.run(grad1, feed_dict=feed_dict)
gr2 = fc.sess.run(g2, feed_dict=feed_dict)
gr22 = fc.sess.run(g22, feed_dict=feed_dict)
gc2 = fc.sess.run(grad2, feed_dict=feed_dict)
gc3 = fc.sess.run(grad3, feed_dict=feed_dict)

c, s = fc.get_ww()
np.save("coef.npy", c)
np.save("coef1.npy", gc1[0])
np.save("coef2.npy", gc2[0])
np.save("coef3.npy", gc3[0])

c = c.reshape(3, 3, 4)


def convol(val, c):
    val2 = np.zeros([val.shape[0] + 2, val.shape[1] + 2])
    val2[1:-1, 1:-1] = val
    res = 0 * val
    for i in range(3):
        for j in range(3):
            res = res + c[i, j] * val2[i : -2 + i, j : -2 + j]
    return res


# rrr=convol(np.ones([64,64]),c[:,:,0])
# print(rrr[0:3,0:3])

print(s2[0])
print(gc2[0].shape)

print(c.shape)
for i in range(4):
    print("=====  GRAD S1 CALC")
    print(gc1[0][0, 30:35, 30:35, i])
    print("=====  GRAD S1 TENS")
    print((gr2[i][0][0].reshape(64, 64))[30:35, 30:35])

for i in range(4):
    print("===== GRAD S2 CALC", i // 4, i % 4)
    print(gc2[0][0, 30:35, 30:35, i])
    print("....")
    print((gr22[i][0][0].reshape(64, 64))[30:35, 30:35])

exit(0)
plt.figure(figsize=(16, 6))
"""
plt.subplot(2,1,1)
plt.plot(gr1[0][0],color='blue',lw=8)
plt.plot(gc1[0][0,:,:,1].flatten(),color='red')
"""
for i in range(4):
    plt.subplot(2, 4, 1 + i)
    plt.imshow(gr22[i][0][0].reshape(64, 64), cmap="jet")
    plt.subplot(2, 4, 5 + i)
    plt.imshow(-gc2[0][0, :, :, 4 * i].reshape(64, 64), cmap="jet")
# plt.plot(gr2[0][0],color='blue',lw=8)
# plt.plot(gc2[0][0,:,:,0].flatten(),color='red')
plt.show()
exit(0)

gg = fc.sess.run(grad, feed_dict=feed_dict)
g2 = fc.sess.run(grad2, feed_dict=feed_dict)

imap = fc.get_map()
imm = imap.reshape(1, 64, 64, 1)
or1, or2 = fc.calc_stat(imm, imm, avg_ang=avg_ang)
imm[0, 10, 10, 0] += 1e-6
rr1, rr2 = fc.calc_stat(imm, imm, avg_ang=avg_ang)
"""
plt.figure()
for i in range(len(gg)):
    for k in range(4):
        plt.subplot(5,4,1+4*i+k)
        plt.imshow(gg[i][0,:,:,k],cmap='jet')
"""
gg2 = fc.sess.run(fc.grd1, feed_dict=feed_dict)
gg3 = fc.sess.run(fc.grd2, feed_dict=feed_dict)

plt.figure()
plt.plot(gg[0][0, :, :, 1].flatten(), color="blue", lw=8)
plt.plot(gg2[1][0][0].flatten(), color="red")

print(len(gg2))

plt.figure()
for i in range(4):
    plt.subplot(2, 2, 1 + i)
    plt.imshow(gg3[i][0][0].reshape(64, 64), cmap="jet")

plt.figure()
print(g2[0].shape)
print(len(g2))
for i in range(4):
    plt.subplot(2, 2, 1 + i)
    plt.imshow(g2[0][0, :, :, i], cmap="jet")
plt.show()

exit(0)

a = np.zeros([20])
b = np.zeros([20])
c = np.zeros([20])
"""
plt.figure()
for i in range(len(gg2)):
    plt.subplot(5,4,1+i)
    plt.imshow(gg2[i][0][0].reshape(64,64),cmap='jet')
    print((rr1[0,0,0,i]-or1[0,0,0,i])*1E6,gg[i//4][0,10//(2**(i//4)),10//(2**(i//4)),i%4],gg2[i][0][0][10+10*64])
    a[i]=(rr1[0,0,0,i]-or1[0,0,0,i])*1E6
    b[i]=gg[i//4][0,10//(2**(i//4)),10//(2**(i//4)),i%4]
    c[i]=gg2[i][0][0][10+10*64]

plt.figure()
plt.plot(abs(b),color='red',lw=6)
plt.plot(abs(c),color='orange',lw=4)
plt.plot(abs(a),color='blue')
plt.yscale('log')
plt.show()
"""

xidx = {}
yidx = {}
for i in range(5):
    xidx[i] = (np.arange(64 * 64, dtype="int") // 64) // (2**i)
    yidx[i] = (np.arange(64 * 64, dtype="int") % 64) // (2**i)

for itt in range(1000):
    imap = fc.get_map()
    imm = imap.reshape(1, 64, 64, 1)
    rr1, rr2 = fc.calc_stat(imm, imm, avg_ang=avg_ang)

    gg = fc.sess.run(grad, feed_dict=feed_dict)
    # gg2=fc.sess.run(fc.grd1,feed_dict=feed_dict)

    diff = np.zeros([1, 64, 64, 1])
    # diff2=np.zeros([1,64,64,1])
    """
    plt.figure(figsize=(16,8))
    """
    for i in range(5):
        tmp = np.sum(
            (
                (
                    rr1[0, 0, 0, i * 4 : (i + 1) * 4] - r1[0, 0, 0, i * 4 : (i + 1) * 4]
                ).reshape(1, 1, 4)
            )
            * (gg[i][0, :, :, :] / (64 * 64)),
            2,
        )
        diff[0, :, :, 0] += (tmp[xidx[i], yidx[i]].reshape(64, 64)) / 64
        """
        if i//4==0:
            tmp=(gg[i//4][0,:,:,i%4]/(64*64)).reshape(1,64,64,1)
        else:
            rtmp=(gg[i//4][0,:,:,i%4]/(64*64)).reshape(64//(2**(i//4)),64//(2**(i//4)))

            tmp=np.zeros([64,64])
            xidx=np.arange(((64//(2**(i//4)))**2),dtype='int')//(64//(2**(i//4)))
            yidx=np.arange(((64//(2**(i//4)))**2),dtype='int')%(64//(2**(i//4)))
            for k in range(2**(i//4)):
                for l in range(2**(i//4)):
                    tmp[k+xidx*(2**(i//4)),l+yidx*(2**(i//4))]=rtmp[xidx,yidx]

        #diff+=((rr1[0,0,0,i]-r1[0,0,0,i])*gg2[i][0][0]/(64*64*64)).reshape(1,64,64,1)
        diff+=((rr1[0,0,0,i]-r1[0,0,0,i])*tmp/64.0).reshape(1,64,64,1)
        """
        """
        imm=imap.reshape(1,64,64,1)+((rr1[0,0,0,i]-r1[0,0,0,i])*gg2[i][0][0]/(16*64*64)).reshape(1,64,64,1)
        r1d,r2d=fc.calc_stat(imm,imm,avg_ang=avg_ang)

        imm=imap.reshape(1,64,64,1)+((rr1[0,0,0,i]-r1[0,0,0,i])*tmp/16).reshape(1,64,64,1)
        r1d2,r2d2=fc.calc_stat(imm,imm,avg_ang=avg_ang)

        plt.subplot(5,4,1+i)
        plt.plot(r1.flatten(),color='black')
        plt.plot(rr1.flatten(),color='blue')
        plt.plot(r1d.flatten(),color='orange')
        plt.plot(r1d2.flatten(),color='gray')
        plt.yscale('log')
        """

    # imm=imap.reshape(1,64,64,1)+diff/2
    # r1d,r2d=fc.calc_stat(imm,imm,avg_ang=avg_ang)

    # imm=imap.reshape(1,64,64,1)+diff2/8
    # r1d2,r2d2=fc.calc_stat(imm,imm,avg_ang=avg_ang)

    par = fc.get_param()
    fc.set_value(par - (diff).flatten(), 0)
    if itt % 100 == 0:
        print(itt, (r1 - rr1).std())
    """
    plt.figure()
    plt.plot(rr1n.flatten(),color='yellow',lw=8)
    plt.plot(r1.flatten(),color='black')
    plt.plot(rr1.flatten(),color='blue')
    plt.plot(r1d.flatten(),color='orange',lw=4)
    plt.plot(r1d2.flatten(),color='gray')
    plt.yscale('log')


    plt.show()
    """

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(d.reshape(64, 64), cmap="jet")
plt.subplot(1, 2, 2)
plt.imshow(imap.reshape(64, 64), cmap="jet")
plt.show()
exit(0)
for itt in range(10):
    gg2 = fc.sess.run(fc.grd1, feed_dict=feed_dict)
    """
    plt.figure()
    for i in range(len(gg2)):
        plt.subplot(5,4,1+i)
        plt.imshow(gg2[i][0][0].reshape(64,64),cmap='jet')
    """
    imap = fc.get_map()

    imm = imap.reshape(1, 64, 64, 1)
    rr1, rr2 = fc.calc_stat(imm, imm, avg_ang=avg_ang)
    if itt == 0:
        rrr1 = rr1

    mat = np.zeros([4096, 4096])
    vec = np.zeros([4096])
    for i in range(20):
        imm = gg2[i][0][0].reshape(64 * 64)
        mat += np.dot(imm.reshape(4096, 1), imm.reshape(1, 4096))
        vec += (
            (rr1[0, 0, 0, i] - r1[0, 0, 0, i])
            * imm
            / (20 * 64 * 64)
            * (2 ** (i // 4) / 16)
        )

    x = vec  # conjugate(mat,vec)

    imm = (imap + x.reshape(64, 64)).reshape(1, 64, 64, 1)
    r1d, r2d = fc.calc_stat(imm, imm, avg_ang=avg_ang)

    plt.figure()
    plt.plot(r1.flatten(), color="black")
    plt.plot(rrr1.flatten(), color="blue")
    plt.plot(r1d.flatten(), color="orange")
    plt.yscale("log")
    plt.show()

    print(((rr1 - r1) ** 2).sum())
    fc.set_value(-x, 0)

exit(0)
imm = (randfield + gg[0][0, :, :, 1]).reshape(1, 64, 64, 1)
r1d1, r2d1 = fc.calc_stat(imm, imm, avg_ang=avg_ang)

plt.plot(r1d1.flatten(), color="red")
plt.yscale("log")
plt.show()
exit(0)
# Run the learning
fc.learn(tw1, tw2, tb1, tb2, NUM_EPOCHS=1000, DECAY_RATE=1.0)

# get the output map
omap = fc.get_map()

modd = omap.reshape(1, nx, nx)
o1, o2 = fc.calc_stat(modd, modd, avg_ang=avg_ang)
# save the statistics on the synthesised data
np.save(scratch_path + "/%s_o1.npy" % (outname), o1)
np.save(scratch_path + "/%s_o2.npy" % (outname), o2)

np.save(scratch_path + "/%s_ref.npy" % (outname), d)
np.save(scratch_path + "/%s_start.npy" % (outname), randfield)
np.save(scratch_path + "/%s_result.npy" % (outname), omap)
