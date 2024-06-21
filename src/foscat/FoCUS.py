import numpy as np
import healpy as hp
import os, sys
import foscat.backend as bk
from scipy.interpolate import griddata


TMPFILE_VERSION='V3_0'

class FoCUS:
    def __init__(self,
                 NORIENT=4,
                 LAMBDA=1.2,
                 KERNELSZ=3,
                 slope=1.0,
                 all_type='float64',
                 nstep_max=16,
                 padding='SAME',
                 gpupos=0,
                 mask_thres=None,
                 mask_norm=False,
                 OSTEP=0,
                 isMPI=False,
                 TEMPLATE_PATH='data',
                 BACKEND='tensorflow',
                 use_2D=False,
                 return_data=False,
                 JmaxDelta=0,
                 DODIV=False,
                 InitWave=None,
                 silent=False,
                 mpi_size=1,
                 mpi_rank=0):

        self.__version__ = '3.0.36'
        # P00 coeff for normalization for scat_cov
        self.TMPFILE_VERSION=TMPFILE_VERSION
        self.P1_dic = None
        self.P2_dic = None
        self.isMPI=isMPI
        self.mask_thres = mask_thres
        self.mask_norm = mask_norm
        self.InitWave=InitWave

        self.mpi_size=mpi_size
        self.mpi_rank=mpi_rank
        self.return_data=return_data
        self.silent=silent

        if not self.silent:
            print('================================================')
            print('          START FOSCAT CONFIGURATION')
            print('================================================')
            sys.stdout.flush()

        self.TEMPLATE_PATH=TEMPLATE_PATH
        if os.path.exists(self.TEMPLATE_PATH)==False:
            if not self.silent:
                print('The directory %s to store temporary information for FoCUS does not exist: Try to create it'%(self.TEMPLATE_PATH))
            try:
                os.system('mkdir -p %s'%(self.TEMPLATE_PATH))
                if not self.silent:
                    print('The directory %s is created')
            except:
                if not self.silent:
                    print('Impossible to create the directory %s'%(self.TEMPLATE_PATH))
                return None
                
        self.number_of_loss=0

        self.history=np.zeros([10])
        self.nlog=0
        self.padding=padding
            
        if OSTEP!=0:
            if not self.silent:
                print('OPTION option is deprecated after version 2.0.6. Please use Jmax option')
            JmaxDelta=OSTEP
        else:
            OSTEP=JmaxDelta
            
        if JmaxDelta<-1:
            if not self.silent:
                print('Warning : Jmax can not be smaller than -1')
            return None
            
        self.OSTEP=JmaxDelta
        self.use_2D=use_2D
        
        if isMPI:
            from mpi4py import MPI

            self.comm= MPI.COMM_WORLD
            if all_type=='float32':
                self.MPI_ALL_TYPE=MPI.FLOAT
            else:
                self.MPI_ALL_TYPE=MPI.DOUBLE
        else:
            self.MPI_ALL_TYPE=None
            
        self.all_type=all_type
        self.BACKEND=BACKEND
        self.backend=bk.foscat_backend(BACKEND,
                                       all_type=all_type,
                                       mpi_rank=mpi_rank,
                                       gpupos=gpupos,
                                       silent=self.silent)

        self.all_bk_type=self.backend.all_bk_type
        self.all_cbk_type=self.backend.all_cbk_type
        self.gpulist=self.backend.gpulist
        self.ngpu=self.backend.ngpu
        self.rank=mpi_rank
        
        self.gpupos=(gpupos+mpi_rank)%self.backend.ngpu

        if not self.silent:
            print('============================================================')
            print('==                                                        ==')
            print('==                                                        ==')
            print('==     RUN ON GPU Rank %d : %s                          =='%(mpi_rank,self.gpulist[self.gpupos%self.ngpu]))
            print('==                                                        ==')
            print('==                                                        ==')
            print('============================================================')
            sys.stdout.flush()

        l_NORIENT=NORIENT
        if DODIV:
            l_NORIENT=NORIENT+2
            
        self.NORIENT=l_NORIENT
        self.LAMBDA=LAMBDA
        self.slope=slope
        
        self.R_off=(KERNELSZ-1)//2
        if (self.R_off//2)*2<self.R_off:
            self.R_off+=1
            
        self.ww_Real  = {} 
        self.ww_Imag  = {} 
        self.ww_CNN_Transpose  = {}
        self.ww_CNN  = {}
        self.X_CNN  = {}
        self.Y_CNN  = {}
        self.Z_CNN  = {}
        
        wwc=np.zeros([KERNELSZ**2,l_NORIENT]).astype(all_type)
        wws=np.zeros([KERNELSZ**2,l_NORIENT]).astype(all_type)

        x=np.repeat(np.arange(KERNELSZ)-KERNELSZ//2,KERNELSZ).reshape(KERNELSZ,KERNELSZ)
        y=x.T

        if NORIENT==1:
            xx=(3/float(KERNELSZ))*LAMBDA*x
            yy=(3/float(KERNELSZ))*LAMBDA*y

            if KERNELSZ==5:
                #w_smooth=np.exp(-2*((3.0/float(KERNELSZ)*xx)**2+(3.0/float(KERNELSZ)*yy)**2))
                w_smooth=np.exp(-(xx**2+yy**2))
                tmp=np.exp(-2*(xx**2+yy**2))-0.25*np.exp(-0.5*(xx**2+yy**2))
            else:
                w_smooth=np.exp(-0.5*(xx**2+yy**2))
                tmp=np.exp(-2*(xx**2+yy**2))-0.25*np.exp(-0.5*(xx**2+yy**2))

            wwc[:,0]=tmp.flatten()-tmp.mean()
            tmp=0*w_smooth
            wws[:,0]=tmp.flatten()
            sigma=np.sqrt((wwc[:,0]**2).mean())
            wwc[:,0]/=sigma
            wws[:,0]/=sigma

            w_smooth=w_smooth.flatten()
        else:
            for i in range(NORIENT):
                a=i/float(NORIENT)*np.pi
                xx=(3/float(KERNELSZ))*LAMBDA*(x*np.cos(a)+y*np.sin(a))
                yy=(3/float(KERNELSZ))*LAMBDA*(x*np.sin(a)-y*np.cos(a))

                if KERNELSZ==5:
                    #w_smooth=np.exp(-2*((3.0/float(KERNELSZ)*xx)**2+(3.0/float(KERNELSZ)*yy)**2))
                    w_smooth=np.exp(-(xx**2+yy**2))
                else:
                    w_smooth=np.exp(-0.5*(xx**2+yy**2))
                tmp1=np.cos(yy*np.pi)*w_smooth
                tmp2=np.sin(yy*np.pi)*w_smooth

                wwc[:,i]=tmp1.flatten()-tmp1.mean()
                wws[:,i]=tmp2.flatten()-tmp2.mean()
                sigma=np.sqrt((wwc[:,i]**2).mean())
                wwc[:,i]/=sigma
                wws[:,i]/=sigma
                    
                if DODIV and i==0:
                    r=(xx**2+yy**2)
                    theta=np.arctan2(yy,xx)
                    theta[KERNELSZ//2,KERNELSZ//2]=0.0
                    tmp1=r*np.cos(2*theta)*w_smooth
                    tmp2=r*np.sin(2*theta)*w_smooth
                        
                    wwc[:,NORIENT]=tmp1.flatten()-tmp1.mean()
                    wws[:,NORIENT]=tmp2.flatten()-tmp2.mean()
                    sigma=np.sqrt((wwc[:,NORIENT]**2).mean())
                        
                    wwc[:,NORIENT]/=sigma
                    wws[:,NORIENT]/=sigma
                    tmp1=r*np.cos(2*theta+np.pi)
                    tmp2=r*np.sin(2*theta+np.pi)
                        
                    wwc[:,NORIENT+1]=tmp1.flatten()-tmp1.mean()
                    wws[:,NORIENT+1]=tmp2.flatten()-tmp2.mean()
                    sigma=np.sqrt((wwc[:,NORIENT+1]**2).mean())
                    wwc[:,NORIENT+1]/=sigma
                    wws[:,NORIENT+1]/=sigma
                        

                w_smooth=w_smooth.flatten()

        self.KERNELSZ=KERNELSZ

        self.Idx_Neighbours={}
        
        if not self.use_2D:
            self.w_smooth = {}
            for i in range(nstep_max):
                lout=(2**i)
                self.ww_Real[lout]=None

            for i in range(1,6):
                lout=(2**i)
                if not self.silent:
                    print('Init Wave ',lout)
                
                if self.InitWave is None:
                    wr,wi,ws,widx=self.init_index(lout)
                else:
                    wr,wi,ws,widx=self.InitWave(self,lout)
                    
                self.Idx_Neighbours[lout]=1 #self.backend.constant(widx)
                self.ww_Real[lout]=wr
                self.ww_Imag[lout]=wi
                self.w_smooth[lout]=ws
        else:
            self.w_smooth=slope*(w_smooth/w_smooth.sum()).astype(self.all_type)
            self.ww_RealT={}
            self.ww_ImagT={}
            self.ww_SmoothT={}

            self.ww_SmoothT[1] = self.backend.constant(self.w_smooth.reshape(KERNELSZ,KERNELSZ,1,1))
            www=np.zeros([KERNELSZ,KERNELSZ,NORIENT,NORIENT],dtype=self.all_type)
            for k in range(NORIENT):
                www[:,:,k,k]=self.w_smooth.reshape(KERNELSZ,KERNELSZ)
            self.ww_SmoothT[NORIENT] = self.backend.constant(www.reshape(KERNELSZ,KERNELSZ,NORIENT,NORIENT))
            self.ww_RealT[1]=self.backend.constant(self.backend.bk_reshape(wwc.astype(self.all_type),[KERNELSZ,KERNELSZ,1,NORIENT]))
            self.ww_ImagT[1]=self.backend.constant(self.backend.bk_reshape(wws.astype(self.all_type),[KERNELSZ,KERNELSZ,1,NORIENT]))
            def doorientw(x):
                y=np.zeros([KERNELSZ,KERNELSZ,NORIENT,NORIENT*NORIENT],dtype=self.all_type)
                for k in range(NORIENT):
                    y[:,:,k,k*NORIENT:k*NORIENT+NORIENT]=x.reshape(KERNELSZ,KERNELSZ,NORIENT)
                return y
            self.ww_RealT[NORIENT]=self.backend.constant(doorientw(wwc.astype(self.all_type)))
            self.ww_ImagT[NORIENT]=self.backend.constant(doorientw(wws.astype(self.all_type)))
        self.pix_interp_val={}
        self.weight_interp_val={}
        self.ring2nest={}
        self.nest2R={}
        self.nest2R1={}
        self.nest2R2={}
        self.nest2R3={}
        self.nest2R4={}
        self.inv_nest2R={}
        self.remove_border={}
            
        self.ampnorm={}
        
        for i in range(nstep_max):
            lout=(2**i)
            self.pix_interp_val[lout]={}
            self.weight_interp_val[lout]={}
            for j in range(nstep_max):
                lout2=(2**j)
                self.pix_interp_val[lout][lout2]=None
                self.weight_interp_val[lout][lout2]=None
            self.ring2nest[lout]=None
            self.Idx_Neighbours[lout]=None
            self.nest2R[lout]=None
            self.nest2R1[lout]=None
            self.nest2R2[lout]=None
            self.nest2R3[lout]=None
            self.nest2R4[lout]=None
            self.inv_nest2R[lout]=None
            self.remove_border[lout]=None
            self.ww_CNN_Transpose[lout]=None
            self.ww_CNN[lout]=None
            self.X_CNN[lout]=None
            self.Y_CNN[lout]=None
            self.Z_CNN[lout]=None

        self.loss={}

    def get_type(self):
        return self.all_type

    def get_mpi_type(self):
        return self.MPI_ALL_TYPE
    
    # ---------------------------------------------−---------
    # --       COMPUTE 3X3 INDEX FOR HEALPIX WORK          --
    # ---------------------------------------------−---------
    def conv_to_FoCUS(self,x,axis=0):
        if self.use_2D and isinstance(x,np.ndarray):
            return(self.to_R(x,axis,chans=self.chans))
        return x

    def diffang(self,a,b):
        return np.arctan2(np.sin(a)-np.sin(b),np.cos(a)-np.cos(b))
        
    def corr_idx_wXX(self,x,y):
        idx=np.where(x==-1)[0]
        res=x
        res[idx]=y[idx]
        return(res)

    # ---------------------------------------------−---------
    # make the CNN working : index reporjection of the kernel on healpix
    
    def calc_indices_convol(self,nside,kernel,rotation=None):
        to,po=hp.pix2ang(nside,np.arange(12*nside*nside),nest=True)
        x,y,z=hp.pix2vec(nside,np.arange(12*nside*nside),nest=True)

        idx=np.argsort((x-1.0)**2+y**2+z**2)[0:kernel]
        x0,y0,z0=hp.pix2vec(nside,idx[0],nest=True)
        t0,p0=hp.pix2ang(nside,idx[0],nest=True)
        
        idx=np.argsort((x-x0)**2+(y-y0)**2+(z-z0)**2)[0:kernel]
        im=np.ones([12*nside**2])*-1
        im[idx]=np.arange(len(idx))
        
        xc,yc,zc=hp.pix2vec(nside,idx,nest=True)
        
        xc-=x0
        yc-=y0
        zc-=z0

        vec=np.concatenate([np.expand_dims(x,-1),
                            np.expand_dims(y,-1),
                            np.expand_dims(z,-1)],1)

        indices=np.zeros([12*nside**2*250,2],dtype='int')
        weights=np.zeros([12*nside**2*250])
        nn=0
        for k in range(12*nside*nside):
            if k%(nside*nside)==nside*nside-1:
                print('Nside=%d KenelSZ=%d %.2f%%'%(nside,kernel,k/(12*nside**2)*100))
            if nside<4:
                idx2=np.arange(12*nside**2)
            else:
                idx2=hp.query_disc(nside, vec[k], np.pi/nside, inclusive=True,nest=True)
            t2,p2=hp.pix2ang(nside,idx2,nest=True)
            if rotation is None:
                rot=[po[k]/np.pi*180.0,(t0-to[k])/np.pi*180.0]
            else:
                rot=[po[k]/np.pi*180.0,(t0-to[k])/np.pi*180.0,rotation[k]]
                
            r=hp.Rotator(rot=rot)
            t2,p2=r(t2,p2)
            
            ii,ww=hp.get_interp_weights(nside,t2,p2,nest=True)
            
            ii=im[ii]

            for l in range(4):
                iii=np.where(ii[l]!=-1)[0]
                if len(iii)>0:
                    indices[nn:nn+len(iii),1]=idx2[iii]
                    indices[nn:nn+len(iii),0]=k*kernel+ii[l,iii]
                    weights[nn:nn+len(iii)]=ww[l,iii]
                    nn+=len(iii)

        indices=indices[0:nn]
        weights=weights[0:nn]
        if k%(nside*nside)==nside*nside-1:
            print('Nside=%d KenelSZ=%d Total Number of value=%d Ratio of the matrix %.2g%%'%(nside,
                                                                                         kernel,
                                                                                         nn,
                                                                                         100*nn/(kernel*12*nside**2*12*nside**2)))
        return indices,weights,xc,yc,zc

    # ---------------------------------------------−---------
    def calc_orientation(self,im): # im is [Ndata,12*Nside**2]
        nside=int(np.sqrt(im.shape[1]//12))
        l_kernel=self.KERNELSZ*self.KERNELSZ
        norient=32
        w=np.zeros([l_kernel,1,2*norient])
        ca=np.cos(np.arange(norient)/norient*np.pi)
        sa=np.sin(np.arange(norient)/norient*np.pi)
        stat=np.zeros([12*nside**2,norient])
        
        if self.ww_CNN[nside] is None:
            self.init_CNN_index(nside,transpose=False)
                
        y=self.Y_CNN[nside]
        z=self.Z_CNN[nside]
        
        for k in range(norient):
            w[:,0,k]=(np.exp(-0.5*nside**2*((y)**2+(z)**2))*np.cos(nside*(y*ca[k]+z*sa[k])*np.pi/2))
            w[:,0,k+norient]=(np.exp(-0.5*nside**2*((y)**2+(z)**2))*np.sin(nside*(y*ca[k]+z*sa[k])*np.pi/2))
            w[:,0,k]=w[:,0,k]-np.mean(w[:,0,k])
            w[:,0,k+norient]=w[:,0,k]-np.mean(w[:,0,k+norient])

        for k in range(im.shape[0]):
            tmp=im[k].reshape(12*nside**2,1)
            im2=self.healpix_layer(tmp,w)
            stat=stat+im2[:,0:norient]**2+im2[:,norient:]**2

        rotation=(np.argmax(stat,1)).astype('float')/32.*180.0
        
        indices,weights,x,y,z=self.calc_indices_convol(nside,9,rotation=rotation)

        return indices,weights
        
    def init_CNN_index(self,nside,transpose=False):
        l_kernel=int(self.KERNELSZ*self.KERNELSZ)
        try:
            indices=np.load('%s/FOSCAT_%s_I%d_%d_%d_CNNV3.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,l_kernel,self.NORIENT,nside))
            weights=np.load('%s/FOSCAT_%s_W%d_%d_%d_CNNV3.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,l_kernel,self.NORIENT,nside))
            xc=np.load('%s/FOSCAT_%s_X%d_%d_%d_CNNV3.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,l_kernel,self.NORIENT,nside))
            yc=np.load('%s/FOSCAT_%s_Y%d_%d_%d_CNNV3.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,l_kernel,self.NORIENT,nside))
            zc=np.load('%s/FOSCAT_%s_Z%d_%d_%d_CNNV3.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,l_kernel,self.NORIENT,nside))
        except:
            indices,weights,xc,yc,zc=self.calc_indices_convol(nside,l_kernel)
            np.save('%s/FOSCAT_%s_I%d_%d_%d_CNNV3.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,l_kernel,self.NORIENT,nside),indices)
            np.save('%s/FOSCAT_%s_W%d_%d_%d_CNNV3.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,l_kernel,self.NORIENT,nside),weights)
            np.save('%s/FOSCAT_%s_X%d_%d_%d_CNNV3.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,l_kernel,self.NORIENT,nside),xc)
            np.save('%s/FOSCAT_%s_Y%d_%d_%d_CNNV3.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,l_kernel,self.NORIENT,nside),yc)
            np.save('%s/FOSCAT_%s_Z%d_%d_%d_CNNV3.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,l_kernel,self.NORIENT,nside),zc)
            if not self.silent:
                print('Write %s/FOSCAT_%s_W%d_%d_%d_CNNV2.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,l_kernel,self.NORIENT,nside))

        self.X_CNN[nside]=xc
        self.Y_CNN[nside]=yc
        self.Z_CNN[nside]=zc
        self.ww_CNN[nside]=self.backend.bk_SparseTensor(indices,
                                                        weights,[12*nside*nside*l_kernel,
                                                                 12*nside*nside])
            
    # ---------------------------------------------−---------
    def healpix_layer_coord(self,im,axis=0):
        nside=int(np.sqrt(im.shape[axis]//12))
        l_kernel=self.KERNELSZ*self.KERNELSZ
        if self.ww_CNN[nside] is None:
            self.init_CNN_index(nside)
        return self.X_CNN[nside],self.Y_CNN[nside],self.Z_CNN[nside]

    # ---------------------------------------------−---------
    def healpix_layer_transpose(self,im,ww,indices=None,weights=None,axis=0):
        nside=int(np.sqrt(im.shape[axis]//12))
        l_kernel=self.KERNELSZ*self.KERNELSZ
            
        if im.shape[1+axis]!=ww.shape[1]:
            if not self.silent:
                print('Weights channels should be equal to the input image channels')
            return -1
        if axis==1:
            results=[]
        
            for k in range(im.shape[0]):
                
                tmp=self.healpix_layer(im[k],ww,indices=indices,weights=weights,axis=0)
                tmp=self.backend.bk_reshape(self.up_grade(tmp,2*nside),[12*4*nside**2,ww.shape[2]])
                
                results.append(tmp)
            
            return self.backend.bk_stack(results,axis=0)
        else:
            tmp=self.healpix_layer(im,ww,indices=indices,weights=weights,axis=axis)
        
            return self.up_grade(tmp,2*nside)
    
    # ---------------------------------------------−---------
    # ---------------------------------------------−---------
    def healpix_layer(self,im,ww,indices=None,weights=None,axis=0):
        nside=int(np.sqrt(im.shape[axis]//12))
        l_kernel=self.KERNELSZ*self.KERNELSZ

        if im.shape[1+axis]!=ww.shape[1]:
            if not self.silent:
                print('Weights channels should be equal to the input image channels')
            return -1

        if indices is None:
            if self.ww_CNN[nside] is None:
                self.init_CNN_index(nside,transpose=False)
            mat=self.ww_CNN[nside]
        else:
            if weights is None:
                print('healpix_layer : If indices is not none weights should be specify')
                return 0
            
            mat=self.backend.bk_SparseTensor(indices,weights,[12*nside*nside*l_kernel,12*nside*nside])
            
        if axis==1:
            results=[]
        
            for k in range(im.shape[0]):
                
                tmp=self.backend.bk_sparse_dense_matmul(mat,im[k])
                
                density=self.backend.bk_reshape(tmp,[12*nside*nside,l_kernel*im.shape[1+axis]])
                
                density=self.backend.bk_matmul(density,self.backend.bk_reshape(ww,[l_kernel*im.shape[1+axis],ww.shape[2]]))
                
                results.append(self.backend.bk_reshape(density,[12*nside**2,ww.shape[2]]))
            
            return self.backend.bk_stack(results,axis=0)
        else:
            tmp=self.backend.bk_sparse_dense_matmul(mat,im)
            
            density=self.backend.bk_reshape(tmp,[12*nside*nside,l_kernel*im.shape[1]])
            
            return self.backend.bk_matmul(density,self.backend.bk_reshape(ww,[l_kernel*im.shape[1],ww.shape[2]]))
    # ---------------------------------------------−---------
    
    # ---------------------------------------------−---------
    def get_rank(self):
        return(self.rank)
    # ---------------------------------------------−---------
    def get_size(self):
        return(self.size)
    
    # ---------------------------------------------−---------
    def barrier(self):
        if self.isMPI:
            self.comm.Barrier()
        
    # ---------------------------------------------−---------
    def toring(self,image,axis=0):
        lout=int(np.sqrt(image.shape[axis]//12))
        
        if self.ring2nest[lout] is None:
            self.ring2nest[lout]=hp.ring2nest(lout,np.arange(12*lout**2))
            
        return image.numpy()[self.ring2nest[lout]]

    #--------------------------------------------------------
    def ud_grade(self,im,j,axis=0):
        rim=im
        for k in range(j):
            rim=self.smooth(rim,axis=axis)
            rim=self.ud_grade_2(rim,axis=axis)
        return rim
    
    #--------------------------------------------------------
    def ud_grade_2(self,im,axis=0):
        
        if self.use_2D:
            ishape=list(im.shape)
            if len(ishape)<axis+2:
                if not self.silent:
                    print('Use of 2D scat with data that has less than 2D')
                return None
                
            npix=im.shape[axis]
            npiy=im.shape[axis+1]
            odata=1
            if len(ishape)>axis+2:
                for k in range(axis+2,len(ishape)):
                    odata=odata*ishape[k]
                    
            ndata=1
            for k in range(axis):
                ndata=ndata*ishape[k]

            tim=self.backend.bk_reshape(self.backend.bk_cast(im),[ndata,npix,npiy,odata])
            tim=self.backend.bk_reshape(tim[:,0:2*(npix//2),0:2*(npiy//2),:],[ndata,npix//2,2,npiy//2,2,odata])

            res=self.backend.bk_reduce_mean(self.backend.bk_reduce_mean(tim,4),2)
        
            if axis==0:
                if len(ishape)==2:
                    return self.backend.bk_reshape(res,[npix//2,npiy//2])
                else:
                    return self.backend.bk_reshape(res,[npix//2,npiy//2]+ishape[axis+2:])
            else:
                if len(ishape)==axis+2:
                    return self.backend.bk_reshape(res,ishape[0:axis]+[npix//2,npiy//2])
                else:
                    return self.backend.bk_reshape(res,ishape[0:axis]+[npix//2,npiy//2]+ishape[axis+2:])
                
            return self.backend.bk_reshape(res,[npix//2,npiy//2])
            
        else:
            shape=list(im.shape)
            
            lout=int(np.sqrt(shape[axis]//12))
            if im.__class__==np.zeros([0]).__class__:
                oshape=np.zeros([len(shape)+1],dtype='int')
                if axis>0:
                    oshape[0:axis]=shape[0:axis]
                oshape[axis]=12*lout*lout//4
                oshape[axis+1]=4
                if len(shape)>axis:
                    oshape[axis+2:]=shape[axis+1:]
            else:
                if axis>0:
                    oshape=shape[0:axis]+[12*lout*lout//4,4]
                else:
                    oshape=[12*lout*lout//4,4]
                if len(shape)>axis:
                    oshape=oshape+shape[axis+1:]

            return(self.backend.bk_reduce_mean(self.backend.bk_reshape(im,oshape),axis=axis+1))
        
    #--------------------------------------------------------
    def up_grade(self,im,nout,axis=0,nouty=None):
        
        if self.use_2D:
            ishape=list(im.shape)
            if len(ishape)<axis+2:
                if not self.silent:
                    print('Use of 2D scat with data that has less than 2D')
                return None
                
            if nouty is None:
                nouty=nout
                
            if ishape[axis]==nout and ishape[axis+1]==nouty:
                return im
                
            npix=im.shape[axis]
            npiy=im.shape[axis+1]
            odata=1
            if len(ishape)>axis+2:
                for k in range(axis+2,len(ishape)):
                    odata=odata*ishape[k]
                    
            ndata=1
            for k in range(axis):
                ndata=ndata*ishape[k]

            tim=self.backend.bk_reshape(self.backend.bk_cast(im),[ndata,npix,npiy,odata])

            res=self.backend.bk_resize_image(tim,[nout,nouty])
        
            if axis==0:
                if len(ishape)==2:
                    return self.backend.bk_reshape(res,[nout,nouty])
                else:
                    return self.backend.bk_reshape(res,[nout,nouty]+ishape[axis+2:])
            else:
                if len(ishape)==axis+2:
                    return self.backend.bk_reshape(res,ishape[0:axis]+[nout,nouty])
                else:
                    return self.backend.bk_reshape(res,ishape[0:axis]+[nout,nouty]+ishape[axis+2:])
                
            return self.backend.bk_reshape(res,[nout,nouty])
            
        else:

            lout=int(np.sqrt(im.shape[axis]//12))
            
            if self.pix_interp_val[lout][nout] is None:
                if not self.silent:
                    print('compute lout nout',lout,nout)
                th,ph=hp.pix2ang(nout,np.arange(12*nout**2,dtype='int'),nest=True)
                p, w = hp.get_interp_weights(lout,th,ph,nest=True)
                del th
                del ph
                
                indice=np.zeros([12*nout*nout*4,2],dtype='int')
                p=p.T
                w=w.T
                t=np.argsort(p,1).flatten() # to make oder indices for sparsematrix computation
                t=(t+np.repeat(np.arange(12*nout*nout)*4,4))
                p=p.flatten()[t]
                w=w.flatten()[t]
                indice[:,0]=np.repeat(np.arange(12*nout**2),4)
                indice[:,1]=p

                self.pix_interp_val[lout][nout]=1
                self.weight_interp_val[lout][nout] = self.backend.bk_SparseTensor(self.backend.constant(indice), \
                                                                                  self.backend.constant(self.backend.bk_cast(w.flatten())), \
                                                                                  dense_shape=[12*nout**2,12*lout**2])

            if lout==nout:
                imout=im
            else:

                ishape=list(im.shape)
                odata=1
                for k in range(axis+1,len(ishape)):
                    odata=odata*ishape[k]
                    
                ndata=1
                for k in range(axis):
                    ndata=ndata*ishape[k]
                tim=self.backend.bk_reshape(self.backend.bk_cast(im),[ndata,12*lout**2,odata])
                if tim.dtype==self.all_cbk_type:
                    rr=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(self.weight_interp_val[lout][nout]
                                                                                   ,self.backend.bk_real(tim[0])),[1,12*nout**2,odata])
                    ii=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(self.weight_interp_val[lout][nout]
                                                                                   ,self.backend.bk_imag(tim[0])),[1,12*nout**2,odata])
                    imout=self.backend.bk_complex(rr,ii)
                else:
                    imout=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(self.weight_interp_val[lout][nout]
                                                                                    ,tim[0]),[1,12*nout**2,odata])

                for k in range(1,ndata):
                    if tim.dtype==self.all_cbk_type:
                        rr=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(self.weight_interp_val[lout][nout]
                                                                                       ,self.backend.bk_real(tim[k])),[1,12*nout**2,odata])
                        ii=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(self.weight_interp_val[lout][nout]
                                                                                       ,self.backend.bk_imag(tim[k])),[1,12*nout**2,odata])
                        imout=self.backend.bk_concat([imout,self.backend.bk_complex(rr,ii)],0)
                    else:
                        imout=self.backend.bk_concat([imout,self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(self.weight_interp_val[lout][nout]
                                                                                                                    ,tim[k]),[1,12*nout**2,odata])],0)

                if axis==0:
                    if len(ishape)==1:
                        return self.backend.bk_reshape(imout,[12*nout**2])
                    else:
                        return self.backend.bk_reshape(imout,[12*nout**2]+ishape[axis+1:])
                else:
                    if len(ishape)==axis+1:
                        return self.backend.bk_reshape(imout,ishape[0:axis]+[12*nout**2])
                    else:
                        return self.backend.bk_reshape(imout,ishape[0:axis]+[12*nout**2]+ishape[axis+1:])
        return(imout)

    #--------------------------------------------------------
    def fill_1d(self,i_arr,nullval=0):
        arr=i_arr.copy()
        # Indices des éléments non nuls
        non_zero_indices = np.where(arr!=nullval)[0]
        
        # Indices de tous les éléments
        all_indices = np.arange(len(arr))
        
        # Interpoler linéairement en utilisant np.interp
        # np.interp(x, xp, fp) : x sont les indices pour lesquels on veut obtenir des valeurs
        # xp sont les indices des données existantes, fp sont les valeurs des données existantes
        interpolated_values = np.interp(all_indices, non_zero_indices, arr[non_zero_indices])
        
        # Mise à jour du tableau original
        arr[arr==nullval] = interpolated_values[arr==nullval]
        
        return arr

    def fill_2d(self,i_arr,nullval=0):
        arr=i_arr.copy()
        # Créer une grille de coordonnées correspondant aux indices du tableau
        x, y = np.indices(arr.shape)
        
        # Extraire les coordonnées des points non nuls ainsi que leurs valeurs
        non_zero_points = np.array((x[arr != nullval], y[arr != nullval])).T
        non_zero_values = arr[arr != nullval]
        
        # Extraire les coordonnées des points nuls
        zero_points = np.array((x[arr == nullval], y[arr == nullval])).T

        # Interpolation linéaire
        interpolated_values = griddata(non_zero_points, non_zero_values, zero_points, method='linear')

        # Remplacer les valeurs nulles par les valeurs interpolées
        arr[arr == nullval] = interpolated_values

        return arr
    
    def fill_healpy(self,i_map,nmax=10,nullval=hp.UNSEEN):
        map=1*i_map
        # Trouver les pixels nuls
        nside = hp.npix2nside(len(map))
        null_indices = np.where(map == nullval)[0]
        
        itt=0
        while null_indices.shape[0]>0 and itt<nmax:
            # Trouver les coordonnées theta, phi pour les pixels nuls
            theta, phi = hp.pix2ang(nside, null_indices)
            
            # Interpoler les valeurs en utilisant les pixels voisins
            # La fonction get_interp_val peut être utilisée pour obtenir les valeurs interpolées
            # pour des positions données en theta et phi.
            i_idx = hp.get_all_neighbours(nside, theta, phi)
            
            i_w=(map[i_idx]!=nullval)*(i_idx!=-1)
            vv=np.sum(i_w,0)
            interpolated_values=np.sum(i_w*map[i_idx],0)

            # Remplacer les valeurs nulles par les valeurs interpolées
            map[null_indices[vv>0]] = interpolated_values[vv>0]/vv[vv>0]

            null_indices = np.where(map == nullval)[0]
            itt+=1
        
        return map
    
    #--------------------------------------------------------
    def ud_grade_1d(self,im,nout,axis=0):
        npix=im.shape[axis]
        
        ishape=list(im.shape)
        odata=1
        for k in range(axis+1,len(ishape)):
            odata=odata*ishape[k]
                    
        ndata=1
        for k in range(axis):
            ndata=ndata*ishape[k]

        nscale=npix//nout
        if npix%nscale==0:
            tim=self.backend.bk_reshape(self.backend.bk_cast(im),[ndata,npix//nscale,nscale,odata])
        else:
            im=self.backend.bk_reshape(self.backend.bk_cast(im),[ndata,npix,odata])
            tim=self.backend.bk_reshape(self.backend.bk_cast(im[:,0:nscale*(npix//nscale),:]),[ndata,npix//nscale,nscale,odata])
        res = self.backend.bk_reduce_mean(tim,2)
        
        if axis==0:
            if len(ishape)==1:
                return self.backend.bk_reshape(res,[nout])
            else:
                return self.backend.bk_reshape(res,[nout]+ishape[axis+1:])
        else:
            if len(ishape)==axis+1:
                return self.backend.bk_reshape(res,ishape[0:axis]+[nout])
            else:
                return self.backend.bk_reshape(res,ishape[0:axis]+[nout]+ishape[axis+1:])
        return self.backend.bk_reshape(res,[nout])
        
    #--------------------------------------------------------
    def up_grade_2_1d(self,im,axis=0):
        
        npix=im.shape[axis]
        
        ishape=list(im.shape)
        odata=1
        for k in range(axis+1,len(ishape)):
            odata=odata*ishape[k]
                    
        ndata=1
        for k in range(axis):
            ndata=ndata*ishape[k]
            
        tim=self.backend.bk_reshape(self.backend.bk_cast(im),[ndata,npix,odata])

        res2=self.backend.bk_expand_dims(self.backend.bk_concat([(tim[:,1:,:]+3*tim[:,:-1,:])/4,tim[:,-1:,:]],1),-2)
        res1=self.backend.bk_expand_dims(self.backend.bk_concat([tim[:,0:1,:],(tim[:,1:,:]*3+tim[:,:-1,:])/4],1),-2)
        res = self.backend.bk_concat([res1,res2],-2)
        
        if axis==0:
            if len(ishape)==1:
                return self.backend.bk_reshape(res,[npix*2])
            else:
                return self.backend.bk_reshape(res,[npix*2]+ishape[axis+1:])
        else:
            if len(ishape)==axis+1:
                return self.backend.bk_reshape(res,ishape[0:axis]+[npix*2])
            else:
                return self.backend.bk_reshape(res,ishape[0:axis]+[npix*2]+ishape[axis+1:])
        return self.backend.bk_reshape(res,[npix*2])

        
    #--------------------------------------------------------
    def convol_1d(self,im,axis=0):
        
        xx=np.arange(5)-2
        w=np.exp(-0.17328679514*(xx)**2)
        c=np.cos((xx)*np.pi/2)
        s=np.sin((xx)*np.pi/2)

        wr=np.array(w*c).reshape(xx.shape[0],1,1)
        wi=np.array(w*s).reshape(xx.shape[0],1,1)
        
        npix=im.shape[axis]
        
        ishape=list(im.shape)
        odata=1
        for k in range(axis+1,len(ishape)):
            odata=odata*ishape[k]
                    
        ndata=1
        for k in range(axis):
            ndata=ndata*ishape[k]
        
        if odata>1:
            wr=np.repeat(wr,odata,2)
            wi=np.repeat(wi,odata,2)
            
        wr=self.backend.bk_cast(self.backend.constant(wr))
        wi=self.backend.bk_cast(self.backend.constant(wi))
        
        tim = self.backend.bk_reshape(self.backend.bk_cast(im),[ndata,npix,odata])

        if tim.dtype==self.all_cbk_type:
            rr1 = self.backend.bk_conv1d(self.backend.bk_real(tim),wr)
            ii1 = self.backend.bk_conv1d(self.backend.bk_real(tim),wi)
            rr2 = self.backend.bk_conv1d(self.backend.bk_imag(tim),wr)
            ii2 = self.backend.bk_conv1d(self.backend.bk_imag(tim),wi)
            res=self.backend.bk_complex(rr1-ii2,ii1+rr2)
        else:
            rr = self.backend.bk_conv1d(tim,wr)
            ii = self.backend.bk_conv1d(tim,wi)
            
            res=self.backend.bk_complex(rr,ii)
            
        if axis==0:
            if len(ishape)==1:
                return self.backend.bk_reshape(res,[npix])
            else:
                return self.backend.bk_reshape(res,[npix]+ishape[axis+1:])
        else:
            if len(ishape)==axis+1:
                return self.backend.bk_reshape(res,ishape[0:axis]+[npix])
            else:
                return self.backend.bk_reshape(res,ishape[0:axis]+[npix]+ishape[axis+1:])
        return self.backend.bk_reshape(res,[npix])

    
    #--------------------------------------------------------
    def smooth_1d(self,im,axis=0):
        
        xx=np.arange(5)-2
        w=np.exp(-0.17328679514*(xx)**2)
        w=w/w.sum()
        w=np.array(w).reshape(xx.shape[0],1,1)
        
        npix=im.shape[axis]
        
        ishape=list(im.shape)
        odata=1
        for k in range(axis+1,len(ishape)):
            odata=odata*ishape[k]
                    
        ndata=1
        for k in range(axis):
            ndata=ndata*ishape[k]

        if odata>1:
            w=np.repeat(w,odata,2)
            
        w=self.backend.bk_cast(self.backend.constant(w))
        
        tim = self.backend.bk_reshape(self.backend.bk_cast(im),[ndata,npix,odata])

        if tim.dtype==self.all_cbk_type:
            rr = self.backend.bk_conv1d(self.backend.bk_real(tim),w)
            ii = self.backend.bk_conv1d(self.backend.bk_real(tim),w)
            res=self.backend.bk_complex(rr,ii)
        else:
            res=self.backend.bk_conv1d(tim,w)
            
        if axis==0:
            if len(ishape)==1:
                return self.backend.bk_reshape(res,[npix])
            else:
                return self.backend.bk_reshape(res,[npix]+ishape[axis+1:])
        else:
            if len(ishape)==axis+1:
                return self.backend.bk_reshape(res,ishape[0:axis]+[npix])
            else:
                return self.backend.bk_reshape(res,ishape[0:axis]+[npix]+ishape[axis+1:])
        return self.backend.bk_reshape(res,[npix])
        
    #--------------------------------------------------------
    def up_grade_1d(self,im,nout,axis=0):
        
        lout=int(im.shape[axis])
        nscale=int(np.log(nout//lout)/np.log(2))
        res=self.backend.bk_cast(im)
        for k in range(nscale):
            res=self.up_grade_2_1d(res,axis=axis)
        return(res)
        
    # ---------------------------------------------−---------
    def init_index(self,nside,kernel=-1):

        if kernel==-1:
            l_kernel=self.KERNELSZ
        else:
            l_kernel=kernel
            
        
        try:
            if self.use_2D:
                tmp=np.load('%s/W%d_%s_%d_IDX.npy'%(self.TEMPLATE_PATH,l_kernel**2,TMPFILE_VERSION,nside))
            else:
                tmp=np.load('%s/FOSCAT_%s_W%d_%d_%d_PIDX.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,l_kernel**2,self.NORIENT,nside))
        except:
            if self.use_2D==False:
                if self.KERNELSZ*self.KERNELSZ>12*nside*nside:
                    l_kernel=3
                    
                aa=np.cos(np.arange(self.NORIENT)/self.NORIENT*np.pi).reshape(1,self.NORIENT)
                bb=np.sin(np.arange(self.NORIENT)/self.NORIENT*np.pi).reshape(1,self.NORIENT)
                x,y,z=hp.pix2vec(nside,np.arange(12*nside*nside),nest=True)
                to,po=hp.pix2ang(nside,np.arange(12*nside*nside),nest=True)

                wav=np.zeros([12*nside*nside,l_kernel**2,self.NORIENT],dtype='complex')
                wwav=np.zeros([12*nside*nside,l_kernel**2])
                iwav=np.zeros([12*nside*nside,l_kernel**2],dtype='int')

                scale=4
                if nside>scale*2:
                    th,ph=hp.pix2ang(nside//scale,np.arange(12*(nside//scale)**2),nest=True)
                else:
                    lidx=np.arange(12*nside*nside)

                pw=np.pi/4.0
                pw2=1/2
                amp=1.0
                
                if l_kernel==5:
                    pw=np.pi/4.0
                    pw2=1/2.25
                    amp=1.0/9.2038
                    
                elif l_kernel==3:
                    pw=1.0/np.sqrt(2)
                    pw2=1.0
                    amp=1/8.45
                    
                elif l_kernel==7:
                    pw=np.pi/4.0
                    pw2=1.0/3.0
                    
                for k in range(12*nside*nside):
                    if k%(nside*nside)==0:
                        if not self.silent:
                            print('Pre-compute nside=%6d %.2f%%'%(nside,100*k/(12*nside*nside)))
                    if nside>scale*2:
                        lidx=hp.get_all_neighbours(nside//scale,th[k//(scale*scale)],ph[k//(scale*scale)],nest=True)
                        lidx=np.concatenate([lidx,np.array([(k//(scale*scale))])],0)
                        lidx=np.repeat(lidx*(scale*scale),(scale*scale))+ \
                              np.tile(np.arange((scale*scale)),lidx.shape[0])
        
                    delta=(x[lidx]-x[k])**2+(y[lidx]-y[k])**2+(z[lidx]-z[k])**2
                    pidx=np.where(delta<(10)/(nside**2))[0]
                    if len(pidx)<l_kernel**2:
                        pidx=np.arange(delta.shape[0])
                    
                    w=np.exp(-pw2*delta[pidx]*(nside**2))
                    pidx=pidx[np.argsort(-w)[0:l_kernel**2]]
                    pidx=pidx[np.argsort(lidx[pidx])]

                    w=np.exp(-pw2*delta[pidx]*(nside**2))
                    iwav[k]=lidx[pidx]
                    wwav[k]=w
                    rot=[po[k]/np.pi*180.0,90+(-to[k])/np.pi*180.0]
                    r=hp.Rotator(rot=rot)
                    ty,tx=r(to[iwav[k]],po[iwav[k]])
                    ty=ty-np.pi/2
                        
                    xx=np.expand_dims(pw*nside*np.pi*tx/np.cos(ty),-1)
                    yy=np.expand_dims(pw*nside*np.pi*ty,-1)
                    
                    wav[k,:,:]=(np.cos(xx*aa+yy*bb)+complex(0.0,1.0)*np.sin(xx*aa+yy*bb))*np.expand_dims(w,-1)
    
                wav=wav-np.expand_dims(np.mean(wav,1),1)
                wav=amp*wav/np.expand_dims(np.std(wav,1),1)
                wwav=wwav/np.expand_dims(np.sum(wwav,1),1)
                
                nk=l_kernel*l_kernel
                indice=np.zeros([12*nside*nside*nk*self.NORIENT,2],dtype='int')
                lidx=np.arange(self.NORIENT)
                for i in range(12*nside*nside):
                    indice[i*nk*self.NORIENT:i*nk*self.NORIENT+nk*self.NORIENT,0]=i*self.NORIENT+np.repeat(lidx,nk)
                    indice[i*nk*self.NORIENT:i*nk*self.NORIENT+nk*self.NORIENT,1]=np.tile(iwav[i],self.NORIENT)

                indice2=np.zeros([12*nside*nside*nk,2],dtype='int')
                for i in range(12*nside*nside):
                    indice2[i*nk:i*nk+nk,0]=i
                    indice2[i*nk:i*nk+nk,1]=iwav[i]
                
                w=np.zeros([12*nside*nside,wav.shape[2],wav.shape[1]],dtype='complex')
                for i in range(wav.shape[1]):
                    for j in range(wav.shape[2]):
                        w[:,j,i]=wav[:,i,j]
                wav=w.flatten()
                wwav=wwav.flatten()
                
                if not self.silent:
                    print('Write FOSCAT_%s_W%d_%d_%d_PIDX.npy'%(TMPFILE_VERSION,self.KERNELSZ**2,self.NORIENT,nside))
                np.save('%s/FOSCAT_%s_W%d_%d_%d_PIDX.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,self.KERNELSZ**2,self.NORIENT,nside),indice)
                np.save('%s/FOSCAT_%s_W%d_%d_%d_WAVE.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,self.KERNELSZ**2,self.NORIENT,nside),wav)
                np.save('%s/FOSCAT_%s_W%d_%d_%d_PIDX2.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,self.KERNELSZ**2,self.NORIENT,nside),indice2)
                np.save('%s/FOSCAT_%s_W%d_%d_%d_SMOO.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,self.KERNELSZ**2,self.NORIENT,nside),wwav)
            else:
                if l_kernel**2==9:
                    if self.rank==0:
                        self.comp_idx_w9(nside)
                elif l_kernel**2==25:
                    if self.rank==0:
                        self.comp_idx_w25(nside)
                else:
                    if self.rank==0:
                        if not self.silent:
                            print('Only 3x3 and 5x5 kernel have been developped for Healpix and you ask for %dx%d'%(KERNELSZ,KERNELSZ))
                        return None

        self.barrier()  
        if self.use_2D:          
            tmp=np.load('%s/W%d_%s_%d_IDX.npy'%(self.TEMPLATE_PATH,l_kernel**2,TMPFILE_VERSION,nside))
        else:
            tmp=np.load('%s/FOSCAT_%s_W%d_%d_%d_PIDX.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,self.KERNELSZ**2,self.NORIENT,nside))
            tmp2=np.load('%s/FOSCAT_%s_W%d_%d_%d_PIDX2.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,self.KERNELSZ**2,self.NORIENT,nside))
            wr=np.load('%s/FOSCAT_%s_W%d_%d_%d_WAVE.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,self.KERNELSZ**2,self.NORIENT,nside)).real
            wi=np.load('%s/FOSCAT_%s_W%d_%d_%d_WAVE.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,self.KERNELSZ**2,self.NORIENT,nside)).imag
            ws=self.slope*np.load('%s/FOSCAT_%s_W%d_%d_%d_SMOO.npy'%(self.TEMPLATE_PATH,TMPFILE_VERSION,self.KERNELSZ**2,self.NORIENT,nside))

            wr=self.backend.bk_SparseTensor(self.backend.constant(tmp),self.backend.constant(self.backend.bk_cast(wr)),dense_shape=[12*nside**2*self.NORIENT,12*nside**2])
            wi=self.backend.bk_SparseTensor(self.backend.constant(tmp),self.backend.constant(self.backend.bk_cast(wi)),dense_shape=[12*nside**2*self.NORIENT,12*nside**2])
            ws=self.backend.bk_SparseTensor(self.backend.constant(tmp2),self.backend.constant(self.backend.bk_cast(ws)),dense_shape=[12*nside**2,12*nside**2])
                
        if kernel==-1:
            self.Idx_Neighbours[nside]=tmp
                
        if self.use_2D:
            if kernel!=-1:
                return tmp
            
        return wr,wi,ws,tmp

        
    # ---------------------------------------------−---------
    # Compute x [....,a,....] to [....,a*a,....]
    #NOT YET TESTED OR IMPLEMENTED
    def auto_cross_2(x,axis=0):
        shape=np.array(x.shape)
        if axis==0:
            y1=self.reshape(x,[shape[0],1,np.cumprod(shape[1:])])
            y2=self.reshape(x,[1,shape[0],np.cumprod(shape[1:])])
            oshape=np.concat([shape[0],shape[0],shape[1:]])
            return(self.reshape(y1*y2,oshape))
    
    # ---------------------------------------------−---------
    # Compute x [....,a,....,b,....] to [....,b*b,....,a*a,....]
    #NOT YET TESTED OR IMPLEMENTED
    def auto_cross_2(x,axis1=0,axis2=1):
        shape=np.array(x.shape)
        if axis==0:
            y1=self.reshape(x,[shape[0],1,np.cumprod(shape[1:])])
            y2=self.reshape(x,[1,shape[0],np.cumprod(shape[1:])])
            oshape=np.concat([shape[0],shape[0],shape[1:]])
            return(self.reshape(y1*y2,oshape))
        
    
    # ---------------------------------------------−---------
    # convert swap axes tensor x [....,a,....,b,....] to [....,b,....,a,....]
    def swapaxes(self,x,axis1,axis2):
        shape=list(x.shape)
        if axis1<0:
            laxis1=len(shape)+axis1
        else:
            laxis1=axis1
        if axis2<0:
            laxis2=len(shape)+axis2
        else:
            laxis2=axis2
        
        naxes=len(shape)
        thelist=[i for i in range(naxes)]
        thelist[laxis1]=laxis2
        thelist[laxis2]=laxis1
        return self.backend.bk_transpose(x,thelist)
    
    # ---------------------------------------------−---------
    # Mean using mask x [....,Npix,....], mask[Nmask,Npix]  to [....,Nmask,....]
    # if use_2D
    # Mean using mask x [....,12,Nside+2*off,Nside+2*off,....], mask[Nmask,12,Nside+2*off,Nside+2*off]  to [....,Nmask,....]
    def masked_mean(self,x,mask,axis=0,rank=0,calc_var=False):
        
        #==========================================================================
        # in input data=[Nbatch,...,X[,Y],NORIENT[,NORIENT]]
        # in input mask=[Nmask,X[,Y]]
        # if self.use_2D :  X[,Y]] = [X,Y]
        # if second level:  NORIENT[,NORIENT]= NORIENT,NORIENT
        #==========================================================================
        
        shape=list(x.shape)
        
        if not self.use_2D:
            nside=int(np.sqrt(x.shape[axis]//12))
            
        l_mask=mask
        if self.mask_norm:
            sum_mask=self.backend.bk_reduce_sum(self.backend.bk_reshape(l_mask,[l_mask.shape[0],np.prod(np.array(l_mask.shape[1:]))]),1)
            if not self.use_2D:
                l_mask=12*nside*nside*l_mask/self.backend.bk_reshape(sum_mask,[l_mask.shape[0]]+[1 for i in l_mask.shape[1:]])
            else:
                l_mask=mask.shape[1]*mask.shape[2]*l_mask/self.backend.bk_reshape(sum_mask,[l_mask.shape[0]]+[1 for i in l_mask.shape[1:]])
            
        if self.use_2D:
            if self.padding=='VALID' and shape[axis]!=l_mask.shape[1]:
                l_mask=l_mask[:,self.KERNELSZ//2:-self.KERNELSZ//2+1,self.KERNELSZ//2:-self.KERNELSZ//2+1]
                if shape[axis]!=l_mask.shape[1]:
                    l_mask=l_mask[:,self.KERNELSZ//2:-self.KERNELSZ//2+1,self.KERNELSZ//2:-self.KERNELSZ//2+1]
            else:
                l_mask=l_mask[:,self.KERNELSZ//2:-self.KERNELSZ//2+1,self.KERNELSZ//2:-self.KERNELSZ//2+1]
                
        # data=[Nbatch,...,X[,Y],NORIENT[,NORIENT]] => data=[Nbatch,...,KERNELSZ//2:-self.KERNELSZ//2,KERNELSZ//2:-self.KERNELSZ//2,NORIENT[,NORIENT]]
        if self.use_2D:
            ichannel=1
            for i in range(axis):
                ichannel*=shape[i]
            ochannel=1
            for i in range(axis+2,len(shape)):
                ochannel*=shape[i]
            l_x=self.backend.bk_reshape(x,[ichannel,shape[axis],shape[axis+1],ochannel])
            oshape=[k for k in shape]
            oshape[axis]=oshape[axis]-self.KERNELSZ+1
            oshape[axis+1]=oshape[axis+1]-self.KERNELSZ+1
            l_x=self.backend.bk_reshape(l_x[:,self.KERNELSZ//2:-self.KERNELSZ//2+1,self.KERNELSZ//2:-self.KERNELSZ//2+1,:],oshape)
        else:
            l_x=x
            
        # data=[Nbatch,...,X[,Y],NORIENT[,NORIENT]] => data=[Nbatch,1,...,X[,Y],NORIENT[,NORIENT]]
        l_x=self.backend.bk_expand_dims(l_x,1)
        
        # mask=[Nmask,X[,Y]] => mask=[1,Nmask,X[,Y]]
        l_mask=self.backend.bk_expand_dims(l_mask,0)
        
        # mask=[1,Nmask,X[,Y]] => mask=[1,Nmask,....,X[,Y]]
        for i in range(1,axis):
            l_mask=self.backend.bk_expand_dims(l_mask,axis)
            
        if l_x.dtype==self.all_cbk_type:
            l_mask=self.backend.bk_complex(l_mask,self.backend.bk_cast(0.0*l_mask))
            
        if self.use_2D:

            # mask=[1,Nmask,....,X,Y] => mask=[1,Nmask,....,X,Y,....]
            for i in range(axis+2,len(x.shape)):
                l_mask=self.backend.bk_expand_dims(l_mask,-1)

            shape1=list(l_mask.shape)
            shape2=list(l_x.shape)

            oshape1=shape1[0:axis+1]+[shape1[axis+1]*shape1[axis+2]]+shape1[axis+3:]
            oshape2=shape2[0:axis+1]+[shape2[axis+1]*shape2[axis+2]]+shape2[axis+3:]
            
            mtmp=self.backend.bk_reshape(l_mask,oshape1)
            vtmp=self.backend.bk_reshape(l_x,oshape2)

            v1=self.backend.bk_reduce_sum(mtmp*vtmp,axis=axis+1)
            v2=self.backend.bk_reduce_sum(mtmp*vtmp*vtmp,axis=axis+1)
            vh=self.backend.bk_reduce_sum(mtmp,axis=axis+1)

            res=v1/vh
            if calc_var:
                if self.backend.bk_is_complex(vtmp):
                    res2=self.backend.bk_complex(self.backend.bk_sqrt(self.backend.bk_real(v2)/self.backend.bk_real(vh)
                                                                      -self.backend.bk_real(res)*self.backend.bk_real(res)), \
                                                 self.backend.bk_sqrt(self.backend.bk_imag(v2)/self.backend.bk_real(vh)
                                                                      -self.backend.bk_imag(res)*self.backend.bk_imag(res)))
                else:
                    res2=self.backend.bk_sqrt((v2/vh-res*res)/(vh))
                return res,res2
            else:
                return res
        else:
            # mask=[1,Nmask,....,X] => mask=[1,Nmask,....,X,....]
            for i in range(axis+1,len(x.shape)):
                l_mask=self.backend.bk_expand_dims(l_mask,-1)
                
            v1=self.backend.bk_reduce_sum(l_mask*l_x,axis=axis+1)
            v2=self.backend.bk_reduce_sum(l_mask*l_x*l_x,axis=axis+1)
            vh=self.backend.bk_reduce_sum(l_mask,axis=axis+1)
            
            res=v1/vh
            if calc_var:
                if self.backend.bk_is_complex(l_x):
                    res2=self.backend.bk_complex(self.backend.bk_sqrt((self.backend.bk_real(v2)/self.backend.bk_real(vh)
                                                                      -self.backend.bk_real(res)*self.backend.bk_real(res))/self.backend.bk_real(v2)), \
                                                 self.backend.bk_sqrt((self.backend.bk_imag(v2)/self.backend.bk_real(vh)
                                                                      -self.backend.bk_imag(res)*self.backend.bk_imag(res))/self.backend.bk_real(v2)))
                else:
                    res2=self.backend.bk_sqrt((v2/vh-res*res)/(vh))
                return res,res2
            else:
                return res
        
    # ---------------------------------------------−---------
    # convert tensor x [....,a,b,....] to [....,a*b,....]
    def reduce_dim(self,x,axis=0):
        shape=list(x.shape)
        
        if axis<0:
            laxis=len(shape)+axis
        else:
            laxis=axis
            
        if laxis>0 :
            oshape=shape[0:laxis]
            oshape.append(shape[laxis]*shape[laxis+1])
        else:
            oshape=[shape[laxis]*shape[laxis+1]]
            
        if laxis<len(shape)-1:
            oshape.extend(shape[laxis+2:])
            
        return(self.backend.bk_reshape(x,oshape))
        
        
    # ---------------------------------------------−---------
    def conv2d(self,image,ww,axis=0):

        if len(ww.shape)==2:
            norient=ww.shape[1]
        else:
            norient=ww.shape[2]

        shape=image.shape

        if axis>0:
            o_shape=shape[0]
            for k in range(1,axis+1):
                o_shape=o_shape*shape[k]
        else:
            o_shape=image.shape[0]
            
        if len(shape)>axis+3:
            ishape=shape[axis+3]
            for k in range(axis+4,len(shape)):
                ishape=ishape*shape[k]
                
            oshape=[o_shape,shape[axis+1],shape[axis+2],ishape]

            #l_image=self.swapaxes(self.bk_reshape(image,oshape),-1,-3)
            l_image=self.backend.bk_reshape(image,oshape)

            l_ww=np.zeros([self.KERNELSZ,self.KERNELSZ,ishape,ishape*norient])
            for k in range(ishape):
                l_ww[:,:,k,k*norient:(k+1)*norient]=ww.reshape(self.KERNELSZ,self.KERNELSZ,norient)
            
            if self.backend.bk_is_complex(l_image):
                r=self.backend.conv2d(self.backend.bk_real(l_image),
                                      l_ww,
                                      strides=[1, 1, 1, 1],
                                      padding=self.padding)
                i=self.backend.conv2d(self.backend.bk_imag(l_image),
                                      l_ww,
                                      strides=[1, 1, 1, 1],
                                      padding=self.padding)
                res=self.backend.bk_complex(r,i)
            else:
                res=self.backend.conv2d(l_image,l_ww,strides=[1, 1, 1, 1],padding=self.padding)

            res=self.backend.bk_reshape(res,[o_shape,shape[axis+1],shape[axis+2],ishape,norient])
        else:
            oshape=[o_shape,shape[axis+1],shape[axis+2],1]
            l_ww=self.backend.bk_reshape(ww,[self.KERNELSZ,self.KERNELSZ,1,norient])

            tmp=self.backend.bk_reshape(image,oshape)
            if self.backend.bk_is_complex(tmp):
                r=self.backend.conv2d(self.backend.bk_real(tmp),
                                      l_ww,
                                      strides=[1, 1, 1, 1],
                                      padding=self.padding)
                i=self.backend.conv2d(self.backend.bk_imag(tmp),
                                         l_ww,
                                         strides=[1, 1, 1, 1],
                                         padding=self.padding)
                res=self.backend.bk_complex(r,i)
            else:
                res=self.backend.conv2d(tmp,
                                        l_ww,
                                        strides=[1, 1, 1, 1],
                                        padding=self.padding)

        return self.backend.bk_reshape(res,shape+[norient])
    
    # ---------------------------------------------−---------
    def convol(self,in_image,axis=0):

        image=self.backend.bk_cast(in_image)
        
        if self.use_2D:
            
            ishape=list(in_image.shape)
            if len(ishape)<axis+2:
                if not self.silent:
                    print('Use of 2D scat with data that has less than 2D')
                return None
                
            npix=ishape[axis]
            npiy=ishape[axis+1]
            odata=1
            if len(ishape)>axis+2:
                for k in range(axis+2,len(ishape)):
                    odata=odata*ishape[k]
                    
            ndata=1
            for k in range(axis):
                ndata=ndata*ishape[k]

            tim=self.backend.bk_reshape(self.backend.bk_cast(in_image),[ndata,npix,npiy,odata])

            if self.backend.bk_is_complex(tim):
                rr1=self.backend.conv2d(self.backend.bk_real(tim),self.ww_RealT[odata],strides=[1, 1, 1, 1],padding=self.padding)
                ii1=self.backend.conv2d(self.backend.bk_real(tim),self.ww_ImagT[odata],strides=[1, 1, 1, 1],padding=self.padding)
                rr2=self.backend.conv2d(self.backend.bk_imag(tim),self.ww_RealT[odata],strides=[1, 1, 1, 1],padding=self.padding)
                ii2=self.backend.conv2d(self.backend.bk_imag(tim),self.ww_ImagT[odata],strides=[1, 1, 1, 1],padding=self.padding)
                res=self.backend.bk_complex(rr1-ii2,ii1+rr2)
            else:
                rr=self.backend.conv2d(tim,self.ww_RealT[odata],strides=[1, 1, 1, 1],padding=self.padding)
                ii=self.backend.conv2d(tim,self.ww_ImagT[odata],strides=[1, 1, 1, 1],padding=self.padding)
                res=self.backend.bk_complex(rr,ii)
                
            if axis==0:
                if len(ishape)==2:
                    return self.backend.bk_reshape(res,[res.shape[1],res.shape[2],self.NORIENT])
                else:
                    return self.backend.bk_reshape(res,[res.shape[1],res.shape[2],self.NORIENT]+ishape[axis+2:])
            else:
                if len(ishape)==axis+2:
                    return self.backend.bk_reshape(res,ishape[0:axis]+[res.shape[1],res.shape[2],self.NORIENT])
                else:
                    return self.backend.bk_reshape(res,ishape[0:axis]+[res.shape[1],res.shape[2],self.NORIENT]+ishape[axis+2:])
                
            return self.backend.bk_reshape(res,[nout,nouty])
                
        else:
            nside=int(np.sqrt(image.shape[axis]//12))

            if self.Idx_Neighbours[nside] is None:
                if self.InitWave is None:
                    wr,wi,ws,widx=self.init_index(nside)
                else:
                    wr,wi,ws,widx=self.InitWave(self,nside)
        
                self.Idx_Neighbours[nside]=1 #self.backend.constant(tmp)
                self.ww_Real[nside]=wr
                self.ww_Imag[nside]=wi
                self.w_smooth[nside]=ws
                
            l_ww_real=self.ww_Real[nside]
            l_ww_imag=self.ww_Imag[nside]
            
            ishape=list(image.shape)
            odata=1
            for k in range(axis+1,len(ishape)):
                odata=odata*ishape[k]
            
            if axis>0:
                ndata=1
                for k in range(axis):
                    ndata=ndata*ishape[k]
                tim=self.backend.bk_reshape(self.backend.bk_cast(image),[ndata,12*nside**2,odata])
                if tim.dtype==self.all_cbk_type:
                    rr1=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_real,self.backend.bk_real(tim[0])),[1,12*nside**2,self.NORIENT,odata])
                    ii1=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_imag,self.backend.bk_real(tim[0])),[1,12*nside**2,self.NORIENT,odata])
                    rr2=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_real,self.backend.bk_imag(tim[0])),[1,12*nside**2,self.NORIENT,odata])
                    ii2=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_imag,self.backend.bk_imag(tim[0])),[1,12*nside**2,self.NORIENT,odata])
                    res=self.backend.bk_complex(rr1-ii2,ii1+rr2)
                else:
                    rr=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_real,tim[0]),[1,12*nside**2,self.NORIENT,odata])
                    ii=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_imag,tim[0]),[1,12*nside**2,self.NORIENT,odata])
                    res=self.backend.bk_complex(rr,ii)
                
                for k in range(1,ndata):
                    if tim.dtype==self.all_cbk_type:
                        rr1=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_real,self.backend.bk_real(tim[k])),[1,12*nside**2,self.NORIENT,odata])
                        ii1=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_imag,self.backend.bk_real(tim[k])),[1,12*nside**2,self.NORIENT,odata])
                        rr2=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_real,self.backend.bk_imag(tim[k])),[1,12*nside**2,self.NORIENT,odata])
                        ii2=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_imag,self.backend.bk_imag(tim[k])),[1,12*nside**2,self.NORIENT,odata])
                        res=self.backend.bk_concat([res,self.backend.bk_complex(rr1-ii2,ii1+rr2)],0)
                    else:
                        rr=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_real,tim[k]),[1,12*nside**2,self.NORIENT,odata])
                        ii=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_imag,tim[k]),[1,12*nside**2,self.NORIENT,odata])
                        res=self.backend.bk_concat([res,self.backend.bk_complex(rr,ii)],0)
                    
                if len(ishape)==axis+1:
                    return self.backend.bk_reshape(res,ishape[0:axis]+[12*nside**2,self.NORIENT])
                else:
                    return self.backend.bk_reshape(res,ishape[0:axis]+[12*nside**2]+ishape[axis+1:]+[self.NORIENT])
                
            if axis==0:
                tim=self.backend.bk_reshape(self.backend.bk_cast(image),[12*nside**2,odata])
                if tim.dtype==self.all_cbk_type:
                    rr1=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_real,self.backend.bk_real(tim)),[12*nside**2,self.NORIENT,odata])
                    ii1=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_imag,self.backend.bk_real(tim)),[12*nside**2,self.NORIENT,odata])
                    rr2=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_real,self.backend.bk_imag(tim)),[12*nside**2,self.NORIENT,odata])
                    ii2=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_imag,self.backend.bk_imag(tim)),[12*nside**2,self.NORIENT,odata])
                    res=self.backend.bk_complex(rr1-ii2,ii1+rr2)
                else:
                    rr=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_real,tim),[12*nside**2,self.NORIENT,odata])
                    ii=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_ww_imag,tim),[12*nside**2,self.NORIENT,odata])
                    res=self.backend.bk_complex(rr,ii)
                
                if len(ishape)==1:
                    return self.backend.bk_reshape(res,[12*nside**2,self.NORIENT])
                else:
                    return self.backend.bk_reshape(res,[12*nside**2]+ishape[axis+1:]+[self.NORIENT])
        return(res)
        

    # ---------------------------------------------−---------
    def smooth(self,in_image,axis=0):

        image=self.backend.bk_cast(in_image)
        
        if self.use_2D:
            
            ishape=list(in_image.shape)
            if len(ishape)<axis+2:
                if not self.silent:
                    print('Use of 2D scat with data that has less than 2D')
                return None
                
            npix=ishape[axis]
            npiy=ishape[axis+1]
            odata=1
            if len(ishape)>axis+2:
                for k in range(axis+2,len(ishape)):
                    odata=odata*ishape[k]
                    
            ndata=1
            for k in range(axis):
                ndata=ndata*ishape[k]

            tim=self.backend.bk_reshape(self.backend.bk_cast(in_image),[ndata,npix,npiy,odata])

            if self.backend.bk_is_complex(tim):
                rr=self.backend.conv2d(self.backend.bk_real(tim),self.ww_SmoothT[odata],strides=[1, 1, 1, 1],padding=self.padding)
                ii=self.backend.conv2d(self.backend.bk_imag(tim),self.ww_SmoothT[odata],strides=[1, 1, 1, 1],padding=self.padding)
                res=self.backend.bk_complex(rr,ii)
            else:
                res=self.backend.conv2d(tim,self.ww_SmoothT[odata],strides=[1, 1, 1, 1],padding=self.padding)
                    
            if axis==0:
                if len(ishape)==2:
                    return self.backend.bk_reshape(res,[res.shape[1],res.shape[2]])
                else:
                    return self.backend.bk_reshape(res,[res.shape[1],res.shape[2]]+ishape[axis+2:])
            else:
                if len(ishape)==axis+2:
                    return self.backend.bk_reshape(res,ishape[0:axis]+[res.shape[1],res.shape[2]])
                else:
                    return self.backend.bk_reshape(res,ishape[0:axis]+[res.shape[1],res.shape[2]]+ishape[axis+2:])
                
            return self.backend.bk_reshape(res,[nout,nouty])
                
        else:
            nside=int(np.sqrt(image.shape[axis]//12))

            if self.Idx_Neighbours[nside] is None:
                
                if self.InitWave is None:
                    wr,wi,ws,widx=self.init_index(nside)
                else:
                    wr,wi,ws,widx=self.InitWave(self,nside)
        
                self.Idx_Neighbours[nside]=1
                self.ww_Real[nside]=wr
                self.ww_Imag[nside]=wi
                self.w_smooth[nside]=ws

            l_w_smooth=self.w_smooth[nside]
            ishape=list(image.shape)
            
            odata=1
            for k in range(axis+1,len(ishape)):
                odata=odata*ishape[k]
                
            if axis==0:
                tim=self.backend.bk_reshape(image,[12*nside**2,odata])
                if tim.dtype==self.all_cbk_type:
                    rr=self.backend.bk_sparse_dense_matmul(l_w_smooth,self.backend.bk_real(tim))
                    ri=self.backend.bk_sparse_dense_matmul(l_w_smooth,self.backend.bk_imag(tim))
                    res=self.backend.bk_complex(rr,ri)
                else:
                    res=self.backend.bk_sparse_dense_matmul(l_w_smooth,tim)
                if len(ishape)==1:
                    return self.backend.bk_reshape(res,[12*nside**2])
                else:
                    return self.backend.bk_reshape(res,[12*nside**2]+ishape[axis+1:])
                
            if axis>0:
                ndata=ishape[0]
                for k in range(1,axis):
                    ndata=ndata*ishape[k]
                tim=self.backend.bk_reshape(image,[ndata,12*nside**2,odata])
                if tim.dtype==self.all_cbk_type:
                    rr=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_w_smooth,self.backend.bk_real(tim[0])),[1,12*nside**2,odata])
                    ri=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_w_smooth,self.backend.bk_imag(tim[0])),[1,12*nside**2,odata])
                    res=self.backend.bk_complex(rr,ri)
                else:
                    res=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_w_smooth,tim[0]),[1,12*nside**2,odata])
                    
                for k in range(1,ndata):
                    if tim.dtype==self.all_cbk_type:
                        rr=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_w_smooth,self.backend.bk_real(tim[k])),[1,12*nside**2,odata])
                        ri=self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_w_smooth,self.backend.bk_imag(tim[k])),[1,12*nside**2,odata])
                        res=self.backend.bk_concat([res,self.backend.bk_complex(rr,ri)],0)
                    else:
                        res=self.backend.bk_concat([res,self.backend.bk_reshape(self.backend.bk_sparse_dense_matmul(l_w_smooth,tim[k]),[1,12*nside**2,odata])],0)

                if len(ishape)==axis+1:
                    return self.backend.bk_reshape(res,ishape[0:axis]+[12*nside**2])
                else:
                    return self.backend.bk_reshape(res,ishape[0:axis]+[12*nside**2]+ishape[axis+1:])
                
                
        return(res)
    
    # ---------------------------------------------−---------
    def get_kernel_size(self):
        return(self.KERNELSZ)
    
    # ---------------------------------------------−---------
    def get_nb_orient(self):
        return(self.NORIENT)
    
    # ---------------------------------------------−---------
    def get_ww(self,nside=1):
        return(self.ww_Real[nside],self.ww_Imag[nside])
    
    # ---------------------------------------------−---------
    def plot_ww(self):
        c,s=self.get_ww()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16,6))
        npt=int(np.sqrt(c.shape[0]))
        for i in range(c.shape[1]):
            plt.subplot(2,c.shape[1],1+i)
            plt.imshow(c[:,i].reshape(npt,npt),cmap='jet',vmin=-c.max(),vmax=c.max())
            plt.subplot(2,c.shape[1],1+i+c.shape[1])
            plt.imshow(s[:,i].reshape(npt,npt),cmap='jet',vmin=-c.max(),vmax=c.max())
            sys.stdout.flush()
        plt.show()

        
    
    
    
