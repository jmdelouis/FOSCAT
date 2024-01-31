import numpy as np
# test iso_mean function 
nside=32
im=np.random.randn(12*nside**2)

# test scat
import foscat.scat as sc
op=sc.funct()

a=op.eval(im)
b=a.fft_ang(nharm=1)
c=a.fft_ang(nharm=1,mean=True)

idx=np.arange(4,dtype='int')

test=0.0
print(b.S2.numpy()[0,4,:]-np.mean(a.S2.numpy()[0,4,idx,idx]))
print(test)


# test scat_cov
import foscat.scat_cov as sc
op=sc.funct()

a=op.eval(im)
b=a.fft_ang(nharm=1)
c=a.fft_ang(nharm=1,mean=True)

idx=np.arange(4,dtype='int')

test=0.0
print(b.C01.numpy()[0,0,4,:]-np.mean(a.C01.numpy()[0,0,4,idx,idx]))
