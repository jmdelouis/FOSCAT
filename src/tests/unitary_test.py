import numpy as np
import matplotlib.pyplot as plt

#=============================================================================
# create directory to store all test
#=============================================================================
import os, sys

try:
    os.system('rm -rf UNITARY')
except:
    print('UNIATRY directory does not exist, create it')

os.system('mkdir UNITARY')

scratch_path='UNITARY'

#=============================================================================
# Unitary test for scat
#=============================================================================
import foscat.scat as sc

r_format=[False,True,False,True]
tabtype=['float32','float64','float32','float64']

for itest in range(len(tabtype)):
    
    print('Start Test : R_format=%s Type=%s ....'%(r_format[itest],tabtype[itest]))
    
    scat_op=sc.funct(NORIENT=4,          # define the number of wavelet orientation
                     KERNELSZ=5,  #KERNELSZ,  # define the kernel size
                     OSTEP=-1,           # get very large scale (nside=1)
                     LAMBDA=1.2,
                     TEMPLATE_PATH=scratch_path,
                     slope=1.0,
                     gpupos=0,
                     use_R_format=r_format[itest],
                     all_type=tabtype[itest])

    im=np.random.randn(12*16*16)

    a=scat_op.convol(im)

    r1=scat_op.eval(im)
    r2=scat_op.eval(im,image2=im)
    r3=scat_op.eval(im,mask=np.zeros([3,12*16*16]))
    r=r1*r2
    r=r1/r2
    r=r1-r2
    r=3*r2
    r=3/r2
    r=3-r2
    r=r1*3
    r=r1/3
    r=r1-3

    
    im=np.random.randn(12*16*16)+np.complex(0,1)*np.random.randn(12*16*16)

    a=scat_op.convol(im)

    r1=scat_op.eval(im)
    r2=scat_op.eval(im,image2=im)
    r3=scat_op.eval(im,mask=np.zeros([3,12*16*16]))
    r=r1*r2
    r=r1/r2
    r=r1-r2
    r=r1+r2
    r=3*r2
    r=3/r2
    r=3-r2
    r=3+r2
    r=r1*3
    r=r1/3
    r=r1-3
    r=r1+3

    print('Test : R_format=%s Type=%s OK'%(r_format[itest],tabtype[itest]))


#=============================================================================
# Unitary test for scat
#=============================================================================
import foscat.scat_cov as sc

r_format=[False,True,False,True]
tabtype=['float32','float64','float32','float64']

for itest in range(len(tabtype)):
    
    print('Start Test : R_format=%s Type=%s ....'%(r_format[itest],tabtype[itest]))
    
    scat_op=sc.funct(NORIENT=4,          # define the number of wavelet orientation
                     KERNELSZ=5,  #KERNELSZ,  # define the kernel size
                     OSTEP=-1,           # get very large scale (nside=1)
                     LAMBDA=1.2,
                     TEMPLATE_PATH=scratch_path,
                     slope=1.0,
                     gpupos=0,
                     use_R_format=r_format[itest],
                     all_type=tabtype[itest])

    im=np.random.randn(12*16*16)

    a=scat_op.convol(im)

    r1=scat_op.eval(im)
    r2=scat_op.eval(im,image2=im)
    r3=scat_op.eval(im,mask=np.zeros([3,12*16*16]))
    r=r1*r2
    r=r1/r2
    r=r1-r2
    r=3*r2
    r=3/r2
    r=3-r2
    r=r1*3
    r=r1/3
    r=r1-3

    
    im=np.random.randn(12*16*16)+np.complex(0,1)*np.random.randn(12*16*16)

    a=scat_op.convol(im)

    r1=scat_op.eval(im)
    r2=scat_op.eval(im,image2=im)
    r3=scat_op.eval(im,mask=np.zeros([3,12*16*16]))
    r=r1*r2
    r=r1/r2
    r=r1-r2
    r=r1+r2
    r=3*r2
    r=3/r2
    r=3-r2
    r=3+r2
    r=r1*3
    r=r1/3
    r=r1-3
    r=r1+3

    print('Test : R_format=%s Type=%s OK'%(r_format[itest],tabtype[itest]))
