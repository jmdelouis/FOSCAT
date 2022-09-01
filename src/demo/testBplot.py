import numpy as np
import matplotlib.pyplot as plt

nitt=4

plt.figure(figsize=(12,8))
for itt in range(nitt):
    #reference
    i1=np.load('../data/iB1_%d.npy'%(itt))
    i2=np.load('../data/iB2_%d.npy'%(itt))
    
    #Focus map
    o1=np.load('../data/oB1_%d.npy'%(itt))
    o2=np.load('../data/oB2_%d.npy'%(itt))

    #Noise baise
    n1=np.load('../data/nB1_%d.npy'%(itt))
    n2=np.load('../data/nB2_%d.npy'%(itt))
    #ref biais
    b1=np.load('../data/bB1_%d.npy'%(itt))
    b2=np.load('../data/bB2_%d.npy'%(itt))
    
    print(i1.shape)
    print(o1.shape)
    print(n1.shape)
    print(b1.shape)
    plt.subplot(nitt,2,1+itt*2)
    #plt.plot(b1[0,0,0,:],color='black')
    #plt.plot(b1[0,0,0,:]-bias,color='Grey')
    plt.plot(i1.flatten(),color='blue')
    plt.plot(o1.flatten(),color='red')
    plt.yscale('log')
    plt.subplot(nitt,2,2+itt*2)
    plt.plot(i2.flatten(),color='blue')
    plt.plot(o2.flatten(),color='red')
    plt.yscale('log')

tt=np.arange(3,dtype='int')#*(nitt)
plt.figure(figsize=(16,10))
for itt in range(3):
    ref=np.load('../data/testBref_%d.npy'%(tt[itt]))
    imap=np.load('../data/testBinput_%d.npy'%(tt[itt]))
    omap=np.load('../data/testBresult_%d.npy'%(tt[itt]))

    plt.subplot(3,5,1+itt*5)
    plt.imshow(ref,origin='lower',vmin=0,vmax=1.0)
    plt.title('Input model Itt %d'%(tt[itt]))
    plt.subplot(3,5,2+itt*5)
    plt.imshow(imap,origin='lower',vmin=0,vmax=1.0)
    plt.title('FoCUS input Itt %d'%(tt[itt]))
    plt.subplot(3,5,3+itt*5)
    plt.imshow(omap,origin='lower',vmin=0,vmax=1.0)
    plt.title('FoCUS result Itt %d'%(tt[itt]))
    plt.subplot(3,5,4+itt*5)
    plt.imshow(omap-ref,origin='lower',vmin=-0.5,vmax=0.5)
    plt.title('Residu result Itt %d'%(tt[itt]))
    plt.subplot(3,5,5+itt*5)
    plt.imshow(imap-ref,origin='lower',vmin=-0.5,vmax=0.5)
    plt.title('Noise Itt %d'%(tt[itt]))
plt.show()
