import numpy as np
import matplotlib.pyplot as plt


nitt = 1

for itt in range(nitt):
  r1=np.load('../data/oB1_%d.npy'%(itt))
  r2=np.load('../data/oB2_%d.npy'%(itt))
  norient=r2.shape[2]
  #nstep=r1.shape[3]//(2*norient)
  plt.figure(figsize=(12,8))
  plt.subplot(1,2,1)
  plt.plot(np.mean(r1,0).flatten())
  plt.yscale('log')
  plt.title('fsl_oB1')
  plt.subplot(1,2,2)
  for i in range(norient):
      plt.plot(np.mean(r2,0)[0,i])
  plt.yscale('log')
  plt.title('fsl_oB2')

  ref=np.load('../data/testBref_%d.npy'%(itt))
  imap=np.load('../data/testBinput_%d.npy'%(itt))
  omap=np.load('../data/testBresult_%d.npy'%(itt))

  amp1=ref.max()/10
  amp2=ref.max()
  plt.figure(figsize=(12,8))
  plt.subplot(2,3,1)
  plt.imshow(ref,cmap='jet',origin='lower',vmin=-amp1,vmax=amp2)
  plt.title('Input model')
  plt.subplot(2,3,2)
  plt.imshow(imap,cmap='jet',origin='lower',vmin=-amp1,vmax=amp2)
  plt.title('noisy data')
  plt.subplot(2,3,3)
  plt.imshow(omap,cmap='jet',origin='lower',vmin=-amp1,vmax=amp2)
  plt.title('FoCUS result')
  plt.subplot(2,3,4)
  plt.imshow(ref-imap,cmap='jet',origin='lower',vmin=-amp1,vmax=amp1)
  plt.title('Input - ref')
  plt.subplot(2,3,5)
  plt.imshow(ref-omap,cmap='jet',origin='lower',vmin=-amp1,vmax=amp1)
  plt.title('FoCUS - ref'+str(itt))


plt.show()
