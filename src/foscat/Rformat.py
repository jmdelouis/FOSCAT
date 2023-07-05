import numpy as np

class Rformat:
    def __init__(self,
                 im,
                 off,
                 axis,
                 chans=12):
        self.data=im
        self.shape=im.shape
        self.axis=axis
        self.off=off
        self.chans=chans
        self.nside=im.shape[axis+1]-2*off

    def get(self):
        return self.data
    
    def get_data(self):
        if self.axis==0:
            return self.data[:,self.off:-self.off,self.off:-self.off]
        if self.axis==1:
            return self.data[:,:,self.off:-self.off,self.off:-self.off]
        if self.axis==2:
            return self.data[:,:,:,self.off:-self.off,self.off:-self.off]
        if self.axis==3:
            return self.data[:,:,:,:,self.off:-self.off,self.off:-self.off]

        print('get_data is implemented till axis==3 and axis=',self.axis)
        exit(0)

    def __add__(self,other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, Rformat)
        
        if isinstance(other, Rformat):
            return Rformat(self.get()+other.get(),self.off,self.axis,chans=self.chans)
        else:
            return Rformat(self.get()+other,self.off,self.axis,chans=self.chans)
        
    def __radd__(self,other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, Rformat)
        
        if isinstance(other, Rformat):
            return Rformat(self.get()+other.get(),self.off,self.axis,chans=self.chans)
        else:
            return Rformat(self.get()+other,self.off,self.axis,chans=self.chans)

    def __sub__(self,other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, Rformat)
        
        if isinstance(other, Rformat):
            return Rformat(self.get()-other.get(),self.off,self.axis,chans=self.chans)
        else:
            return Rformat(self.get()-other,self.off,self.axis,chans=self.chans)
            
    def __neg__(self):
        
        return Rformat(-self.get(),self.off,self.axis,chans=self.chans)

    def __mul__(self,other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, Rformat)
        
        if isinstance(other, Rformat):
            return Rformat(self.get()*other.get(),self.off,self.axis,chans=self.chans)
        else:
            return Rformat(self.get()*other,self.off,self.axis,chans=self.chans)
        
    def __rmul__(self,other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, Rformat)
        
        if isinstance(other, Rformat):
            return Rformat(self.get()*other.get(),self.off,self.axis,chans=self.chans)
        else:
            return Rformat(self.get()*other,self.off,self.axis,chans=self.chans)

    def __truediv__(self,other):
        assert isinstance(other, float)  or isinstance(other, np.float32) or isinstance(other, int) or \
            isinstance(other, bool) or isinstance(other, Rformat)
        
        if isinstance(other, Rformat):
            return Rformat(self.get()/other.get(),self.off,self.axis,chans=self.chans)
        else:
            return Rformat(self.get()/other,self.off,self.axis,chans=self.chans)
