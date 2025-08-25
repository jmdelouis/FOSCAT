import numpy as np

import foscat.scat_cov as sc
import foscat.HOrientedConvol as hs

class UNET:

    def __init__(
            self,
            nparam=1,
            KERNELSZ=3,
            NORIENT=4,
            chanlist=None,
            in_nside=1,
            n_chan_in=1,
            n_chan_out=1,
            cell_ids=None,
            SEED=1234,
            filename=None,
    ):
        self.f=sc.funct(KERNELSZ=KERNELSZ)
        
        if chanlist is None:
            nlayer=int(np.log2(in_nside))
            chanlist=[4*2**k for k in range(nlayer)]
        else:
            nlayer=len(chanlist)
        print('N_layer ',nlayer)

        n=0
        wconv={}
        hconv={}
        l_cell_ids={}
        self.KERNELSZ=KERNELSZ
        kernelsz=self.KERNELSZ

        # create the CNN part
        l_nside=in_nside
        l_cell_ids[0]=cell_ids.copy()
        l_data=self.f.backend.bk_cast(np.ones([1,1,l_cell_ids[0].shape[0]]))
        l_chan=n_chan_in
        print('Initial chan %d Npix=%d'%(l_chan,l_data.shape[2]))
        for l in range(nlayer):
            print('Layer %d Npix=%d'%(l,l_data.shape[2]))
            # init double convol weights
            wconv[2*l]=n
            nw=l_chan*chanlist[l]*kernelsz*kernelsz
            print('Layer %d conv [%d,%d]'%(l,l_chan,chanlist[l]))
            n+=nw
            wconv[2*l+1]=n
            nw=chanlist[l]*chanlist[l]*kernelsz*kernelsz
            print('Layer %d conv [%d,%d]'%(l,chanlist[l],chanlist[l]))
            n+=nw

            hconvol=hs.HOrientedConvol(l_nside,3,cell_ids=l_cell_ids[l])
            hconvol.make_idx_weights()
            hconv[l]=hconvol
            l_data,n_cell_ids=self.f.ud_grade_2(l_data,cell_ids=l_cell_ids[l],nside=l_nside)
            l_cell_ids[l+1]=self.f.backend.to_numpy(n_cell_ids)
            l_nside//=2
            # plus one to add the input downgrade data
            l_chan=chanlist[l]+n_chan_in

        self.n_cnn=n
        self.l_cell_ids=l_cell_ids
        self.wconv=wconv
        self.hconv=hconv

        # create the transpose CNN part
        m_cell_ids={}
        m_cell_ids[0]=l_cell_ids[nlayer]
        t_wconv={}
        t_hconv={}
        
        for l in range(nlayer):
            #upgrade data
            l_chan+=n_chan_in
            l_data=self.f.up_grade(l_data,l_nside*2,
                                   cell_ids=l_cell_ids[nlayer-l],
                                   o_cell_ids=l_cell_ids[nlayer-1-l],
                                   nside=l_nside)
            print('Transpose Layer %d Npix=%d'%(l,l_data.shape[2]))
            
            
            m_cell_ids[l]=l_cell_ids[nlayer-1-l]
            l_nside*=2
            
            # init double convol weights
            t_wconv[2*l]=n
            nw=l_chan*l_chan*kernelsz*kernelsz
            print('Transpose Layer %d conv [%d,%d]'%(l,l_chan,l_chan))
            n+=nw
            t_wconv[2*l+1]=n
            out_chan=n_chan_out
            if nlayer-1-l>0:
                out_chan+=chanlist[nlayer-1-l]
            print('Transpose Layer %d conv [%d,%d]'%(l,l_chan,out_chan))
            nw=l_chan*out_chan*kernelsz*kernelsz
            n+=nw

            hconvol=hs.HOrientedConvol(l_nside,3,cell_ids=m_cell_ids[l])
            hconvol.make_idx_weights()
            t_hconv[l]=hconvol

            # plus one to add the input downgrade data
            l_chan=out_chan
        print('Final chan %d Npix=%d'%(out_chan,l_data.shape[2]))
        self.n_cnn=n
        self.m_cell_ids=l_cell_ids
        self.t_wconv=t_wconv
        self.t_hconv=t_hconv
        self.x=self.f.backend.bk_cast((np.random.rand(n)-0.5)/self.KERNELSZ)
        self.nside=in_nside
        self.n_chan_in=n_chan_in
        self.n_chan_out=n_chan_out
        self.chanlist=chanlist

    def get_param(self):
        return self.x

    def set_param(self,x):
        self.x=self.f.backend.bk_cast(x)
        
    def eval(self,data):
        # create the CNN part
        l_nside=self.nside
        l_chan=self.n_chan_in
        l_data=data
        m_data=data
        nlayer=len(self.chanlist)
        kernelsz=self.KERNELSZ
        ud_data={}
        
        for l in range(nlayer):
            # init double convol weights
            nw=l_chan*self.chanlist[l]*kernelsz*kernelsz
            ww=self.x[self.wconv[2*l]:self.wconv[2*l]+nw]
            ww=self.f.backend.bk_reshape(ww,[l_chan,
                                             self.chanlist[l],
                                             kernelsz*kernelsz])
            l_data = self.hconv[l].Convol_torch(l_data,ww)
            
            nw=self.chanlist[l]*self.chanlist[l]*kernelsz*kernelsz
            ww=self.x[self.wconv[2*l+1]:self.wconv[2*l+1]+nw]
            ww=self.f.backend.bk_reshape(ww,[self.chanlist[l],
                                             self.chanlist[l],
                                             kernelsz*kernelsz])
            
            l_data = self.hconv[l].Convol_torch(l_data,ww)
            
            l_data,_=self.f.ud_grade_2(l_data,
                                       cell_ids=self.l_cell_ids[l],
                                       nside=l_nside)

            ud_data[l]=m_data
            
            m_data,_=self.f.ud_grade_2(m_data,
                                       cell_ids=self.l_cell_ids[l],
                                       nside=l_nside)

            l_data = self.f.backend.bk_concat([m_data,l_data],1)
            
            l_nside//=2
            # plus one to add the input downgrade data
            l_chan=self.chanlist[l]+self.n_chan_in

        for l in range(nlayer):
            l_chan+=self.n_chan_in
            l_data=self.f.up_grade(l_data,l_nside*2,
                                   cell_ids=self.l_cell_ids[nlayer-l],
                                   o_cell_ids=self.l_cell_ids[nlayer-1-l],
                                   nside=l_nside)
            

            l_data = self.f.backend.bk_concat([ud_data[nlayer-1-l],l_data],1)
            l_nside*=2
            
            # init double convol weights
            out_chan=self.n_chan_out
            if nlayer-1-l>0:
                out_chan+=self.chanlist[nlayer-1-l]
            nw=l_chan*l_chan*kernelsz*kernelsz
            ww=self.x[self.t_wconv[2*l]:self.t_wconv[2*l]+nw]
            ww=self.f.backend.bk_reshape(ww,[l_chan,
                                             l_chan,
                                             kernelsz*kernelsz])
            
            c_data = self.t_hconv[l].Convol_torch(l_data,ww)
            
            nw=l_chan*out_chan*kernelsz*kernelsz
            ww=self.x[self.t_wconv[2*l+1]:self.t_wconv[2*l+1]+nw]
            ww=self.f.backend.bk_reshape(ww,[l_chan,
                                             out_chan,
                                             kernelsz*kernelsz])
            l_data = self.t_hconv[l].Convol_torch(c_data,ww)

            # plus one to add the input downgrade data
            l_chan=out_chan
            
        return l_data
