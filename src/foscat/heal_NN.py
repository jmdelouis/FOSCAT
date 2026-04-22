import pickle

import numpy as np

import foscat.scat_cov as sc

class CNN:

    def __init__(
            self,
            nparam=1,
            KERNELSZ=3,
            NORIENT=4,
            chanlist=[],
            in_nside=1,
            n_chan_in=1,
            SEED=1234,
            add_undersample_data=False,
            all_type='float32',
            filename=None,
            scat_operator=None,
            BACKEND='tensorflow'
    ):

        if filename is not None:
            outlist = pickle.load(open("%s.pkl" % (filename), "rb"))
            self.scat_operator = sc.funct(KERNELSZ=outlist[3],
                                            NORIENT= outlist[9],
                                            all_type=outlist[7])
            self.KERNELSZ = self.scat_operator.KERNELSZ
            self.all_type = self.scat_operator.all_type
            self.npar = outlist[2]
            self.nscale = outlist[5]
            self.chanlist = outlist[0]
            self.in_nside = outlist[4]
            self.nbatch = outlist[1]
            self.n_chan_in = outlist[8]
            self.NORIENT = outlist[9]
            self.x = self.scat_operator.backend.bk_cast(outlist[6])
            self.out_nside = self.in_nside // (2**(self.nscale+1))
        else:
            self.add_undersample_data=add_undersample_data
            self.nscale = len(chanlist)-1
            self.npar = nparam
            self.n_chan_in = n_chan_in
            if scat_operator is None:
                self.scat_operator = sc.funct(
                    KERNELSZ=KERNELSZ,
                    NORIENT=NORIENT,
                    all_type=all_type)
            else:
                self.scat_operator = scat_operator

            self.chanlist = chanlist
            self.KERNELSZ = self.scat_operator.KERNELSZ
            self.NORIENT = self.scat_operator.NORIENT
            self.all_type = self.scat_operator.all_type
            self.in_nside = in_nside
            self.out_nside = self.in_nside // (2**(self.nscale+1))
            self.backend = self.scat_operator.backend
            np.random.seed(SEED)
            self.x = self.scat_operator.backend.bk_cast(
                np.random.rand(self.get_number_of_weights())
            )
        self.mpi_size = self.scat_operator.mpi_size
        self.mpi_rank = self.scat_operator.mpi_rank
        self.BACKEND = BACKEND
        self.gpupos = self.scat_operator.gpupos
        self.ngpu = self.scat_operator.ngpu
        self.gpulist = self.scat_operator.gpulist

    def save(self, filename):

        outlist = [
            self.chanlist,
            self.nbatch,
            self.npar,
            self.KERNELSZ,
            self.in_nside,
            self.nscale,
            self.get_weights().numpy(),
            self.all_type,
            self.n_chan_in,
            self.NORIENT,
        ]

        myout = open("%s.pkl" % (filename), "wb")
        pickle.dump(outlist, myout)
        myout.close()

    def get_number_of_weights(self):
        totnchan = 0
        for i in range(self.nscale):
            totnchan = totnchan + (self.chanlist[i]+int(self.add_undersample_data)* self.n_chan_in) * self.chanlist[i + 1]
        return (
            self.npar * 12 * self.out_nside**2 * (self.chanlist[self.nscale]+int(self.add_undersample_data)* self.n_chan_in)*self.NORIENT
            + totnchan * self.KERNELSZ * (self.KERNELSZ//2+1)*self.NORIENT*self.NORIENT
            + self.KERNELSZ * (self.KERNELSZ//2+1) * self.n_chan_in * self.chanlist[0]*self.NORIENT
        )

    def set_weights(self, x):
        self.x = x

    def get_weights(self):
        return self.x

    def init_wave(self):
        w0=np.zeros([self.n_chan_in, self.KERNELSZ * (self.KERNELSZ//2+1),  self.chanlist[0], self.NORIENT])
        if self.KERNELSZ==3:
            w0[:,0]=-0.2
            w0[:,1]=-0.5
            w0[:,2]=-0.2
            w0[:,3]=0.2
            w0[:,4]=0.5
            w0[:,5]=0.2
        if self.KERNELSZ==5:
            w0[:,0]=-0.1
            w0[:,1]=-0.2
            w0[:,2]=-0.5
            w0[:,3]=-0.2
            w0[:,4]=-0.1
            w0[:,10]=0.1
            w0[:,11]=0.2
            w0[:,12]=0.5
            w0[:,13]=0.2
            w0[:,14]=0.1
        
        a=2*np.sqrt(6/(12 * self.out_nside**2 * self.chanlist[self.nscale]*self.NORIENT*self.npar))
        x=(np.random.rand(self.get_number_of_weights())-0.5)*a
        
        w0=w0.flatten()
        x[0:w0.shape[0]]=w0
        nn = self.KERNELSZ * (self.KERNELSZ//2+1) * self.n_chan_in * self.chanlist[0]*self.NORIENT
        
        for k in range(self.nscale):
            ww = np.zeros([self.chanlist[k], self.NORIENT, self.KERNELSZ * (self.KERNELSZ//2+1),  self.chanlist[k + 1], self.NORIENT])
            
            if self.KERNELSZ==3:
                ww[:,:,0]=-0.2
                ww[:,:,1]=-0.5
                ww[:,:,2]=-0.2
                ww[:,:,3]=0.2
                ww[:,:,4]=0.5
                ww[:,:,5]=0.2
            if self.KERNELSZ==5:
                ww[:,:,0]=-0.1
                ww[:,:,1]=-0.2
                ww[:,:,2]=-0.5
                ww[:,:,3]=-0.2
                ww[:,:,4]=-0.1
                ww[:,:,10]=0.1
                ww[:,:,11]=0.2
                ww[:,:,12]=0.5
                ww[:,:,13]=0.2
                ww[:,:,14]=0.1
            x[nn : nn + self.KERNELSZ
                    * (self.KERNELSZ//2+1)
                    * self.NORIENT*self.NORIENT
                    * self.chanlist[k]
                    * self.chanlist[k + 1]
                ]=ww.flatten()
                
            nn = nn + (self.KERNELSZ * (self.KERNELSZ//2+1)
                * self.NORIENT*self.NORIENT
                * self.chanlist[k]
                * self.chanlist[k + 1])
            
        self.x = self.scat_operator.backend.bk_cast(x)
        
    def calc_matrix_first_layer(self,noise_map):
        # Décalage circulaire par matrice de permutation
        def circ_shift_matrix(N,k):
            return np.roll(np.eye(N), shift=-k, axis=1)
        
        im=self.scat_operator.convol(noise_map)
        mm=np.mean(abs(im.cpu().numpy()),0)
        Norient=mm.shape[1]
        xx=np.cos(np.arange(Norient)/Norient*2*np.pi)
        yy=np.sin(np.arange(Norient)/Norient*2*np.pi)

        a=np.sum(mm*xx[None,:,None],1)
        b=np.sum(mm*yy[None,:,None],1)
        o=np.fmod(Norient*np.arctan2(-b,a)/(2*np.pi)+Norient,Norient)
        xx=np.arange(Norient)
        alpha = o[:,None,:]-xx[None,:,None]
        beta = np.fmod(1+o[:,None,:]-xx[None,:,None],Norient)
        alpha=(1-alpha)*(alpha<1)*(alpha>0)+beta*(beta<1)*(beta>0)

        m=np.zeros([mm.shape[0],4,4,mm.shape[2]])
        for k in range(4):
            m[:,k,:,:]=np.roll(alpha,k,1)
        m=np.mean(m,0)
        return self.scat_operator.backend.bk_cast(m[None,None,...])
        
    def eval(self, in_im, 
            indices=None, 
            weights=None, 
            out_map=False, 
            first_layer_rot=None,
            activation='relu'):

        x = self.x
        ww = self.backend.bk_reshape(
            x[0 : self.KERNELSZ * (self.KERNELSZ//2+1) * self.n_chan_in * self.chanlist[0]*self.NORIENT],
            [self.n_chan_in, 1 , self.KERNELSZ * (self.KERNELSZ//2+1),  self.chanlist[0], self.NORIENT],
        )
        nn = self.KERNELSZ * (self.KERNELSZ//2+1) * self.n_chan_in * self.chanlist[0]*self.NORIENT

        im = self.scat_operator.healpix_layer(in_im[:,:,None,:], ww)
        
        if first_layer_rot is not None:
            im = self.backend.bk_reshape(im,[im.shape[0],im.shape[1],self.NORIENT,1,im.shape[3]])
            im = self.backend.bk_reduce_sum(im*first_layer_rot,2)
            
        
        if activation=='relu':
            im = self.backend.bk_relu(im)
        elif activation=='abs':
            im = self.backend.bk_abs(im)
        
        im = self.backend.bk_reduce_sum(self.backend.bk_reshape(im,[im.shape[0],im.shape[1],self.NORIENT,im.shape[3]//4,4]),4)

        if self.add_undersample_data:
            l_im=self.backend.bk_repeat(self.backend.bk_reshape(in_im,[in_im.shape[0],in_im.shape[1],1,in_im.shape[2]]),self.NORIENT,2)
            l_im=self.backend.bk_reduce_sum(
                                           self.backend.bk_reshape(l_im,[in_im.shape[0],in_im.shape[1],self.NORIENT,l_im.shape[3]//4,4]), 4)
            im=self.backend.bk_concat([im,l_im],1)
                
            
        if out_map:
            return im
        
        for k in range(self.nscale):
            ww = self.scat_operator.backend.bk_reshape(
                x[
                    nn : nn
                    + self.KERNELSZ
                    * (self.KERNELSZ//2+1)
                    * self.NORIENT*self.NORIENT
                    * (self.chanlist[k]+int(self.add_undersample_data)* self.n_chan_in)
                    * self.chanlist[k + 1]
                ],
                [self.chanlist[k]+int(self.add_undersample_data)* self.n_chan_in,
                 self.NORIENT,
                 self.KERNELSZ * (self.KERNELSZ//2+1),
                 self.chanlist[k + 1],
                 self.NORIENT],
            )
            nn = (
                nn
                + self.KERNELSZ
                * (self.KERNELSZ//2+1)
                * self.NORIENT*self.NORIENT
                * (self.chanlist[k]+int(self.add_undersample_data)* self.n_chan_in)
                * self.chanlist[k + 1]
            )
            if indices is None:
                im = self.scat_operator.healpix_layer(im, ww)
            else:
                im = self.scat_operator.healpix_layer(
                    im, ww, indices=indices[k], weights=weights[k]
                )
            
            if activation=='relu':
                im = self.backend.bk_relu(im)
            elif activation=='abs':
                im = self.backend.bk_abs(im)
            im = self.backend.bk_reduce_sum(self.backend.bk_reshape(im,[im.shape[0],im.shape[1],self.NORIENT,im.shape[3]//4,4]),4)

            if self.add_undersample_data:
                l_im=self.backend.bk_reduce_sum(
                    self.backend.bk_reshape(l_im,[l_im.shape[0],l_im.shape[1],self.NORIENT,l_im.shape[3]//4,4]), 4)
                im=self.backend.bk_concat([im,l_im],1)

        ww = self.scat_operator.backend.bk_reshape(
            x[
                nn : nn
                + self.npar * 12 * self.out_nside**2 * (self.chanlist[self.nscale]+int(self.add_undersample_data)* self.n_chan_in)*self.NORIENT
            ],
            [12 * self.out_nside**2 * (self.chanlist[self.nscale]+int(self.add_undersample_data)* self.n_chan_in)*self.NORIENT, self.npar],
        )

        im = self.scat_operator.backend.bk_matmul(
            self.scat_operator.backend.bk_reshape(
                im, [im.shape[0], im.shape[1] * im.shape[2] * im.shape[3]]
            ),
            ww,
        )
        #im = self.scat_operator.backend.bk_reshape(im, [self.npar])
        #im = self.scat_operator.backend.bk_relu(im)
        return im
        
class GCNN:

    def __init__(
        self,
        nparam=1,
        KERNELSZ=3,
        NORIENT=4,
        chanlist=[],
        in_nside=1,
        out_chan=1,
        SEED=1234,
        all_type='float32',
        filename=None,
        scat_operator=None,
        BACKEND='tensorflow'
    ):

        if filename is not None:
            outlist = pickle.load(open("%s.pkl" % (filename), "rb"))
            self.scat_operator = sc.funct(KERNELSZ=outlist[3],NORIENT=outlist[8], all_type=outlist[7])
            self.KERNELSZ = self.scat_operator.KERNELSZ
            self.all_type = self.scat_operator.all_type
            self.npar = outlist[2]
            self.nscale = outlist[5]
            self.chanlist = outlist[0]
            self.in_nside = outlist[4]
            self.nbatch = outlist[1]
            self.NORIENT = outlist[8]
            self.out_chan = outlist[9]
            self.x = self.scat_operator.backend.bk_cast(outlist[6])
            self.out_nside = self.in_nside // (2**self.nscale)
        else:
            self.nscale = len(chanlist)-1
            self.npar = nparam
            
            if scat_operator is None:
                self.scat_operator = sc.funct(
                    KERNELSZ=KERNELSZ,
                    NORIENT=NORIENT,
                    all_type=all_type)
            else:
                self.scat_operator = scat_operator

            self.chanlist = chanlist
            self.KERNELSZ = self.scat_operator.KERNELSZ
            self.NORIENT = self.scat_operator.NORIENT
            self.all_type = self.scat_operator.all_type
            self.in_nside = in_nside
            self.out_nside = self.in_nside * (2**self.nscale)
            self.out_chan = out_chan
            self.backend = self.scat_operator.backend
            np.random.seed(SEED)
            self.x = self.scat_operator.backend.bk_cast(
                np.random.rand(self.get_number_of_weights())
            )
        self.mpi_size = self.scat_operator.mpi_size
        self.mpi_rank = self.scat_operator.mpi_rank
        self.BACKEND = BACKEND
        self.gpupos = self.scat_operator.gpupos
        self.ngpu = self.scat_operator.ngpu
        self.gpulist = self.scat_operator.gpulist

    def save(self, filename):

        outlist = [
            self.chanlist,
            self.nbatch,
            self.npar,
            self.KERNELSZ,
            self.in_nside,
            self.nscale,
            self.get_weights().numpy(),
            self.all_type,
            self.NORIENT,
            self.out_chan
        ]

        myout = open("%s.pkl" % (filename), "wb")
        pickle.dump(outlist, myout)
        myout.close()

    def get_number_of_weights(self):
        totnchan = 0
        for i in range(self.nscale):
            totnchan = totnchan + self.chanlist[i] * self.chanlist[i + 1]
        return (
            self.npar * 12 * self.in_nside**2 * self.chanlist[0]*self.NORIENT
            + totnchan * self.KERNELSZ * (self.KERNELSZ//2+1)*self.NORIENT*self.NORIENT
            + self.chanlist[-1]*self.out_chan*self.NORIENT
        )

    def set_weights(self, x):
        self.x = x

    def get_weights(self):
        return self.x

    def eval(self, im, indices=None, weights=None):

        x = self.x
        
        ww = self.backend.bk_reshape(
            x[0:self.npar * 12 * self.in_nside**2 * self.chanlist[0]*self.NORIENT],
            [self.npar,12 * self.in_nside**2 * self.chanlist[0]*self.NORIENT],
        )

        im = self.scat_operator.backend.bk_matmul(im,ww)
        
        im = self.backend.bk_reshape(im,[im.shape[0],self.chanlist[0],self.NORIENT,12 * self.in_nside**2])
        
        nn = self.npar * 12 * self.in_nside**2 * self.chanlist[0]
        
        for k in range(self.nscale):
            
            im = self.scat_operator.backend.bk_relu(im)
            
            im = self.backend.bk_reshape(
                        self.scat_operator.backend.bk_repeat(im,4,axis=-1),
                        [im.shape[0],im.shape[1],self.NORIENT,im.shape[3]*4])
            
            ww = self.scat_operator.backend.bk_reshape(
                x[
                    nn : nn
                    + self.KERNELSZ
                    * (self.KERNELSZ//2+1)
                    * self.NORIENT *self.NORIENT
                    * self.chanlist[k]
                    * self.chanlist[k + 1]
                ],
                [self.chanlist[k] , self.NORIENT, self.KERNELSZ * (self.KERNELSZ//2+1),  self.chanlist[k + 1],self.NORIENT],
            )
            nn = (
                nn
                + self.KERNELSZ
                * (self.KERNELSZ//2+1)
                * self.NORIENT *self.NORIENT
                * self.chanlist[k]
                * self.chanlist[k + 1]
            )
            
            if indices is None:
                im = self.scat_operator.healpix_layer(im, ww)
            else:
                im = self.scat_operator.healpix_layer(
                    im, ww, indices=indices[k], weights=weights[k]
                )

        ww = self.scat_operator.backend.bk_reshape(
                x[
                    nn : nn
                    + self.chanlist[-1]*self.NORIENT
                    * self.out_chan
                ],
                [1,self.chanlist[-1],self.NORIENT, self.out_chan, 1],
            )
        im = self.backend.bk_reduce_mean(im[:,:,:,None]*ww,[1,2])
        #im = self.scat_operator.backend.bk_relu(im)
         
        return im
