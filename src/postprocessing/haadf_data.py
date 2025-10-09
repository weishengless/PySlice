from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import logging
import pickle
import hashlib
from .wf_data import WFData

logger = logging.getLogger(__name__)

try:
    import torch ; xp = torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if device.type == 'mps':
        complex_dtype = torch.complex64
        float_dtype = torch.float32
    else:
        complex_dtype = torch.complex128
        float_dtype = torch.float64
except ImportError:
    TORCH_AVAILABLE = False
    xp = np
    print("PyTorch not available, falling back to NumPy")
    complex_dtype = np.complex128
    float_dtype = np.float64

@dataclass
class HAADFData(WFData):
    # inherit all attributes from parent object
    def __init__(self, WFData) -> object:
        self.__class__ = type(WFData.__class__.__name__,
                              (self.__class__, WFData.__class__),
                              {})
        self.__dict__ = WFData.__dict__

    def calculateADF(self, inner_mrad: float = 45, outer_mrad = 150, preview: bool = False) -> np.ndarray:
        self.xs=xp.asarray(sorted(list(set(self.probe_positions[:,0]))))
        self.ys=xp.asarray(sorted(list(set(self.probe_positions[:,1]))))
        self.adf=xp.zeros((len(self.xs),len(self.ys)))
        q=xp.sqrt(self.kxs[:,None]**2+self.kys[None,:]**2)
        #print(np.shape(self.wavefunction_data),np.shape(q))
        radius_inner = (inner_mrad * 1e-3) / self.probe.wavelength
        radius_outer = (outer_mrad * 1e-3) / self.probe.wavelength
        mask=xp.zeros(q.shape, device=self.array.device if TORCH_AVAILABLE else None)
        mask[q>=radius_inner]=1 ; mask[q>=radius_outer]=0
        probe_positions=xp.asarray(self.probe_positions)
        for i,x in enumerate(self.xs):
            for j,y in enumerate(self.ys):
                dxy=xp.sqrt( xp.sum( (probe_positions-xp.asarray([x,y])[None,:])**2,axis=1 ) )
                p=xp.argmin(dxy)
                exits=self.array[p,:,:,:,-1] # which probe position, all frames, kx, ky, last layer
                if preview and i==0 and j==0:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    preview_data = xp.mean(xp.absolute(exits),axis=0)**.1*(1-mask)
                    if TORCH_AVAILABLE:
                        preview_data = preview_data.cpu().numpy()
                    ax.imshow(preview_data, cmap="inferno")
                    plt.show()
                #print(np.shape(exits),p,np.sum(np.absolute(exits)))
                collected = xp.mean(xp.sum( xp.absolute(exits*mask[None,:,:]),axis=(1,2)))
                self.adf[i,j]=collected #; print(collected)
        return self.adf

    def plot(self,filename=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        array = self.adf.T # imshow convention: y,x. our convention: x,y
        extent = ( xp.amin(self.xs) , xp.amax(self.xs) , xp.amin(self.ys) , xp.amax(self.ys) )
        ax.imshow(array, cmap="inferno",extent=extent)
        ax.set_xlabel("x ($\\AA$)")
        ax.set_ylabel("y ($\\AA$)")

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
