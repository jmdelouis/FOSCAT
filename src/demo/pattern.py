import numpy as np
import os, sys
import matplotlib.pyplot as plt


bolo=['857-1','857-2','857-3','857-4']
Signal = ["/home1/scratch/jmdeloui/DATA4SROLL4/%s_REP7_2"%(i) for i in bolo]
Ptg0 = ["/home1/scratch/jmdeloui/DATA4SROLL4/%s_REP6_ptg"%(i) for i in bolo]
Ptg1 = ["/home1/scratch/jmdeloui/DATA4SROLL4/%s_REP6_ptg_TUPLE_1"%(i) for i in bolo]
Ptg2 = ["/home1/scratch/jmdeloui/DATA4SROLL4/%s_REP6_ptg_TUPLE_2"%(i) for i in bolo]
Hit = ["/home1/scratch/jmdeloui/DATA4SROLL4/%s_REP6_hit"%(i) for i in bolo]
phase = ["/home1/scratch/jmdeloui/DATA4SROLL4/%s_REP6_phregul"%(i) for i in bolo]
rgcnn = ["/export/home1/jmdeloui/DATA4SROLL4/%s_REP6_rgadutot.int32.bin"%(i) for i in bolo]

nring = 26051-240


#=================================================================================
# FUNCTIONS
#=================================================================================
def normalise(x,y,z,lenght):
  for i in range(lenght):
    r = norm(x[i],y[i],z[i])
    x[i][r!=0] /=r[r!=0]
    y[i][r!=0] /=r[r!=0]
    z[i][r!=0] /= r[r!=0]

#---------------------------------------------------------------------------------
def norm(A,B,C):
  return np.sqrt((A*A)+(B*B)+(C*C))
#---------------------------------------------------------------------------------
def survey(X,Y,Z,beginring,endring,rgidx,psi,signal):
  rgidx = rgidx.astype('int32')
  nbring = rgidx[endring]- rgidx[beginring]

  X_survey = np.zeros([4,nbring,256])
  Y_survey = np.zeros([4,nbring,256])
  Z_survey = np.zeros([4,nbring,256])
  psi_survey = np.zeros([4,nbring,256])
  sig_survey = np.zeros([4,nbring,256])

  for i in range(4):
    X_survey[i] = X[i][rgidx[beginring]:rgidx[endring],:]
    Y_survey[i] = Y[i][rgidx[beginring]:rgidx[endring],:]
    Z_survey[i] = Z[i][rgidx[beginring]:rgidx[endring],:]
    psi_survey[i] = psi[i][rgidx[beginring]:rgidx[endring],:]
    sig_survey[i] = signal[i][rgidx[beginring]:rgidx[endring],:]

  return X_survey,Y_survey,Z_survey,psi_survey,sig_survey
#---------------------------------------------------------------------------------
def calcul_idx(x,y,z,x_stot,y_stot,z_stot,psi,psi_stot,pi):
  tab_idx = np.zeros([4,x.shape[1],256],dtype = 'int')
  # Calcul distance angulaire
  for i in range(4):
    for j in range(x[i].shape[0]):
      for k in range(256):
        if pi==1 :
          adist = np.arctan2(np.sin(psi[i][j,k])- np.sin(psi_stot[i]),np.cos(psi[i][j,k])-np.cos(psi_stot[i])+np.pi)
        else:
          adist = np.arctan2(np.sin(psi[i][j,k])- np.sin(psi_stot[i]),np.cos(psi[i][j,k])-np.cos(psi_stot[i]))

        dist = np.sqrt(((x[i][j,k]-x_stot[i])**2)+((y[i][j,k]-y_stot[i])**2)+((z[i][j,k]-z_stot[i])**2))+adist
        tab_idx[i,j,k] = np.argmin(dist)

  return tab_idx
#---------------------------------------------------------------------------------
def create_img256_bolo(data,nbolo):
  ## create 256*256 imagette from diff for debug first duplicate data
  tmp = data[0]
 
  for i in range(nbolo-1):
    tmp = np.append(tmp,data[0],axis=0)
 
  lim = 256-tmp.shape[0]
  nul_tab = np.zeros([256])

  for i in range(lim):
    tmp =  np.vstack([tmp,nul_tab])
    
  return tmp
#---------------------------------------------------------------------------------
def create_img(data,xsize,ysize,nbolo):
  #For debug we only use first bolometer
  tmp = data[0][:xsize]
 
  lim = xsize-tmp.shape[0]


  if(lim >0):  
    nul_tab = np.zeros([256])
    for i in range(lim):
      tmp =  np.vstack([tmp,nul_tab])
    
  return tmp
#---------------------------------------------------------------------------------
def create_imagette(xsize,ysize):
  ## Create imagette 256*256
  #print("Load data")
  
  X =np.zeros([4,256,256])
  Y =np.zeros([4,256,256])
  Z =np.zeros([4,256,256])
  psi = np.zeros([4,256,256])
  signal = np.zeros([4,256,256])
  rgidx = np.zeros([4,26061])

  for i in range(4):
    rgidx[i] = np.fromfile(rgcnn[i],dtype='int32')
    tmp =int(0)
    for j in range(len(rgidx[i])):
      if rgidx[i][j] == 0:
        rgidx[i][j] = tmp
      tmp = rgidx[i][j]

    X[i]=np.load('../data/%s_FSL_X.npy'%(bolo[i])).flatten().reshape(256,256)
    Y[i]=np.load('../data/%s_FSL_Y.npy'%(bolo[i])).flatten().reshape(256,256)
    Z[i]=np.load('../data/%s_FSL_Z.npy'%(bolo[i])).flatten().reshape(256,256)
    psi[i]=np.load('../data/%s_FSL_PSI.npy'%(bolo[i])).flatten().reshape(256,256)
    signal[i] = np.load('../data/%s_FSL_MAP.npy'%(bolo[i])).flatten().reshape(256,256)
    X[i][np.isnan(X[i])==True]=np.median(X[i][np.isnan(X[i])==False])
    Y[i][np.isnan(Y[i])==True]=np.median(Y[i][np.isnan(Y[i])==False])
    Z[i][np.isnan(Z[i])==True]=np.median(Z[i][np.isnan(Z[i])==False])
    psi[i][np.isnan(psi[i])==True]=np.median(psi[i][np.isnan(psi[i])==False])
    signal[i][np.isnan(signal[i])==True]=np.median(signal[i][np.isnan(signal[i])==False])

   
  print("Calcul survey ")
  x_stot,y_stot,z_stot,psi_stot,sig_stot = survey(X,Y,Z,240,21720,rgidx[0],psi,signal)
  x_s1,y_s1,z_s1,psi_s1,sig_s1 = survey(X,Y,Z,240,5721,rgidx[0],psi,signal)
  x_s2,y_s2,z_s2,psi_s2,sig_s2 = survey(X,Y,Z,5720,11195,rgidx[0],psi,signal)
  x_s3,y_s3,z_s3,psi_s3,sig_s3 = survey(X,Y,Z,11194,16692,rgidx[0],psi,signal)
  x_s4,y_s4,z_s4,psi_s4,sig_s4 = survey(X,Y,Z,16691,21721,rgidx[0],psi,signal)
  x_s5,y_s5,z_s5,psi_s5,sig_s5 = survey(X,Y,Z,21720,26051,rgidx[0],psi,signal)

  print("Normalisation ")
  ## normalisation
  normalisation = True
  if(normalisation):
    #normalise rgidx
    normalise(x_stot,y_stot,z_stot,len(x_stot))
    normalise(x_s1,y_s1,z_s1,len(x_s1))
    normalise(x_s2,y_s2,z_s2,len(x_s2))
    normalise(x_s3,y_s3,z_s3,len(x_s3))
    normalise(x_s4,y_s4,z_s4,len(x_s4))
    normalise(x_s5,y_s5,z_s5,len(x_s5))


  #Create data to process 
  tab_idxs5 = np.load('../data/tab_idx_survey5.npy')
  tab_idxs2 = np.load('../data/tab_idx_survey2.npy')
  tab_idxs4 = np.load('../data/tab_idx_survey4.npy')

  #diffs2_s1 = np.zeros([4,55,256])
  #diffs2_s1 = np.zeros([4,sig_s2.shape[1],256])
  #diffs4_s3 = np.zeros([4,sig_s4.shape[1],256])

  diffs2_s1 = np.zeros([4,sig_s2.shape[1],256])
  diffs4_s3 = np.zeros([4,sig_s4.shape[1],256])

  for i in range(4):
    tmp_sig = sig_stot[i].flatten()
    # 1 - calcul difference s1-s2 and plot result
    diffs2_s1[i] = sig_s2[i] - tmp_sig[tab_idxs2[i]]
    # 2 - calcul difference s3-s4 and plot result
    diffs4_s3[i] = sig_s4[i] - tmp_sig[tab_idxs4[i]]
 

  nbolo = 4

  d1 = create_img(diffs2_s1,xsize,ysize,nbolo)
  d2 = create_img(diffs4_s3,xsize,xsize,nbolo)

  return d1,d2
#---------------------------------------------------------------------------------
