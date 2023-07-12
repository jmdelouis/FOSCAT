import numpy as np
import matplotlib.pyplot as plt
import foscat.Synthesis as synthe

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

#=================================================================================
# Function to reduce the data used in the FoCUS algorithm 
#=================================================================================
def dodown(a,nside):
    nin=int(np.sqrt(a.shape[0]//12))
    if nin==nside:
        return(a)
    return(np.mean(a.reshape(12*nside*nside,(nin//nside)**2),1))
    
#=============================================================================
# Unitary test for scat
#=============================================================================
import foscat.scat as sc

r_format=[False,True,False,True]
tabtype=['float32','float64','float32','float64']

nside=16
data='Venus_256.npy'
ims=dodown(np.load(data),nside)
mask=(np.random.rand(3,ims.shape[0])>0.5).astype('float')


def lossX(x,scat_operator,args):
        
    ref = args[0]
    im  = args[1]
    mask = args[2]
    docross = args[3]

    if docross:
        learn=scat_operator.eval(im,image2=x,Auto=False,mask=mask)
    else:
        learn=scat_operator.eval(x,mask=mask)
            
    loss=scat_operator.reduce_sum(scat_operator.square(ref-learn))        

    return(loss)


for itest in range(len(tabtype)):
    
    scat_op=sc.funct(NORIENT=4,          # define the number of wavelet orientation
                     KERNELSZ=5,  #KERNELSZ,  # define the kernel size
                     OSTEP=-1,           # get very large scale (nside=1)
                     LAMBDA=1.2,
                     TEMPLATE_PATH=scratch_path,
                     slope=1.0,
                     gpupos=0,
                     use_R_format=r_format[itest],
                     all_type=tabtype[itest])
    
    print('Start Test Synthesis no cross 0 : R_format=%s Type=%s ....'%(r_format[itest],tabtype[itest]))
    ref=scat_op.eval(ims,mask=mask)
    loss1=synthe.Loss(lossX,scat_op,ref,ims,mask,False)
    sy = synthe.Synthesis([loss1])
    
    imap=np.random.randn(ims.shape[0])
    
    omap=sy.run(imap,
                DECAY_RATE=0.9995,
                NUM_EPOCHS = 10,
                LEARNING_RATE = 0.03,
                EPSILON = 1E-15)
    
    print('Test Synthesis no cross 0 : R_format=%s Type=%s DONE'%(r_format[itest],tabtype[itest]))
    
    print('Start Test Synthesis cross : R_format=%s Type=%s ....'%(r_format[itest],tabtype[itest]))
    
    ref=scat_op.eval(ims,image2=ims,mask=mask,Auto=False)
    loss1=synthe.Loss(lossX,scat_op,ref,ims,mask,True)
    sy = synthe.Synthesis([loss1])
    
    print(ims.shape,mask.shape,imap.shape)
    
    imap=np.random.randn(ims.shape[0])
    omap=sy.run(imap,
                DECAY_RATE=0.9995,
                NUM_EPOCHS = 10,
                LEARNING_RATE = 0.03,
                EPSILON = 1E-15)
    
    print('Test Synthesis cross : R_format=%s Type=%s DONE'%(r_format[itest],tabtype[itest]))
    
    print('Start Test : R_format=%s Type=%s ....'%(r_format[itest],tabtype[itest]))
    

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

    
    im=np.random.randn(12*16*16)+complex(0,1)*np.random.randn(12*16*16)

    scat_op=sc.funct(NORIENT=4,          # define the number of wavelet orientation
                     KERNELSZ=5,  #KERNELSZ,  # define the kernel size
                     OSTEP=-1,           # get very large scale (nside=1)
                     LAMBDA=1.2,
                     TEMPLATE_PATH=scratch_path,
                     slope=1.0,
                     gpupos=0,
                     use_R_format=r_format[itest],
                     all_type=tabtype[itest])
    
    print('Start Test COMPLEX Synthesis no cross : R_format=%s Type=%s ....'%(r_format[itest],tabtype[itest]))
    
    ref=scat_op.eval(im,mask=mask)
    loss1=synthe.Loss(lossX,scat_op,ref,im,mask,False)
    sy = synthe.Synthesis([loss1])
    
    imap=np.random.randn(im.shape[0])+complex(0,1)*np.random.randn(im.shape[0])
    
    omap=sy.run(imap,
                do_lbfgs=False, # LBFGS does not know how to manage complex minimisation
                DECAY_RATE=0.9995,
                NUM_EPOCHS = 10,
                LEARNING_RATE = 0.03,
                EPSILON = 1E-15)
    
    print('Test Synthesis COMPLEX no cross : R_format=%s Type=%s DONE'%(r_format[itest],tabtype[itest]))
    
    print('Start Test Synthesis cross : R_format=%s Type=%s ....'%(r_format[itest],tabtype[itest]))
    
    ref=scat_op.eval(im,image2=im,mask=mask,Auto=False)
    loss1=synthe.Loss(lossX,scat_op,ref,im,mask,True)
    sy = synthe.Synthesis([loss1])
    
    imap=np.random.randn(im.shape[0])+complex(0,1)*np.random.randn(im.shape[0])
    
    omap=sy.run(imap,
                do_lbfgs=False, # LBFGS does not know how to manage complex minimisation
                DECAY_RATE=0.9995,
                NUM_EPOCHS = 10,
                LEARNING_RATE = 0.03,
                EPSILON = 1E-15)
    
    print('Test Synthesis cross : R_format=%s Type=%s DONE'%(r_format[itest],tabtype[itest]))
    
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

for itest in range(len(tabtype)):
    
    scat_op=sc.funct(NORIENT=4,          # define the number of wavelet orientation
                     KERNELSZ=5,  #KERNELSZ,  # define the kernel size
                     OSTEP=-1,           # get very large scale (nside=1)
                     LAMBDA=1.2,
                     TEMPLATE_PATH=scratch_path,
                     slope=1.0,
                     gpupos=0,
                     use_R_format=r_format[itest],
                     all_type=tabtype[itest])
    
    print('Start Test Synthesis no cross 1 : R_format=%s Type=%s ....'%(r_format[itest],tabtype[itest]))
    
    ref=scat_op.eval(ims,mask=mask)
    loss1=synthe.Loss(lossX,scat_op,ref,ims,mask,False)
    sy = synthe.Synthesis([loss1])
    
    imap=np.random.randn(ims.shape[0])
    omap=sy.run(imap,
                DECAY_RATE=0.9995,
                NUM_EPOCHS = 10,
                LEARNING_RATE = 0.03,
                EPSILON = 1E-15)
    print('Test Synthesis no cross 1 : R_format=%s Type=%s DONE'%(r_format[itest],tabtype[itest]))
    
    print('Start Test Synthesis cross 1 : R_format=%s Type=%s ....'%(r_format[itest],tabtype[itest]))
    
    ref=scat_op.eval(ims,image2=ims,mask=mask,Auto=False)
    loss1=synthe.Loss(lossX,scat_op,ref,ims,mask,True)
    sy = synthe.Synthesis([loss1])
    imap=np.random.randn(ims.shape[0])
    omap=sy.run(imap,
                DECAY_RATE=0.9995,
                NUM_EPOCHS = 10,
                LEARNING_RATE = 0.03,
                EPSILON = 1E-15)
    
    print('Test Synthesis cross 1 : R_format=%s Type=%s DONE'%(r_format[itest],tabtype[itest]))
    
    print('Start Test : R_format=%s Type=%s ....'%(r_format[itest],tabtype[itest]))
    

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

    
    im=np.random.randn(12*16*16)+complex(0,1)*np.random.randn(12*16*16)

    print('Start Test Synthesis no cross 2 : R_format=%s Type=%s ....'%(r_format[itest],tabtype[itest]))
    
    ref=scat_op.eval(im,mask=mask)
    loss1=synthe.Loss(lossX,scat_op,ref,im,mask,False)
    sy = synthe.Synthesis([loss1])
    
    imap=np.random.randn(im.shape[0])+complex(0,1)*np.random.randn(im.shape[0])
    
    omap=sy.run(imap,
                do_lbfgs=False,
                DECAY_RATE=0.9995,
                NUM_EPOCHS = 10,
                LEARNING_RATE = 0.03,
                EPSILON = 1E-15)
    
    print('Test Synthesis no cross 2 : R_format=%s Type=%s DONE'%(r_format[itest],tabtype[itest]))
    
    print('Start Test Synthesis cross : R_format=%s Type=%s ....'%(r_format[itest],tabtype[itest]))
    
    ref=scat_op.eval(im,image2=im,mask=mask,Auto=False)
    loss1=synthe.Loss(lossX,scat_op,ref,im,mask,True)
    sy = synthe.Synthesis([loss1])
    
    imap=np.random.randn(im.shape[0])+complex(0,1)*np.random.randn(im.shape[0])
    
    omap=sy.run(imap,
                do_lbfgs=False,
                DECAY_RATE=0.9995,
                NUM_EPOCHS = 10,
                LEARNING_RATE = 0.03,
                EPSILON = 1E-15)
    
    print('Test Synthesis cross : R_format=%s Type=%s DONE'%(r_format[itest],tabtype[itest]))
    
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
