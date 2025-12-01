from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import logging
import pickle
import hashlib
from .wf_data import WFData, SEA_ECO_AVAILABLE

# Optional sea-eco integration
if SEA_ECO_AVAILABLE:
    from pySEA.sea_eco.architecture.base_structure_numpy import (
        Signal, Dimensions, Dimension, GeneralMetadata
    )

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
        # Use float_dtype to ensure MPS compatibility (float32 on MPS, float64 otherwise)
        self.xs=xp.asarray(sorted(list(set(self.probe_positions[:,0]))), dtype=float_dtype)
        self.ys=xp.asarray(sorted(list(set(self.probe_positions[:,1]))), dtype=float_dtype)
        self.adf=xp.zeros((len(self.xs),len(self.ys)), dtype=float_dtype)
        q=xp.sqrt(self.kxs[:,None]**2+self.kys[None,:]**2)
        #print(np.shape(self.wavefunction_data),np.shape(q))
        radius_inner = (inner_mrad * 1e-3) / self.probe.wavelength
        radius_outer = (outer_mrad * 1e-3) / self.probe.wavelength
        mask=xp.zeros(q.shape, device=self.array.device if TORCH_AVAILABLE else None, dtype=float_dtype)
        mask[q>=radius_inner]=1 ; mask[q>=radius_outer]=0
        probe_positions=xp.asarray(self.probe_positions, dtype=float_dtype)
        for i,x in enumerate(self.xs):
            for j,y in enumerate(self.ys):
                dxy=xp.sqrt( xp.sum( (probe_positions-xp.asarray([x,y], dtype=float_dtype)[None,:])**2,axis=1 ) )
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

    def to_signal(self, inner_mrad: float = 45, outer_mrad: float = 150) -> 'Signal':
        """
        Convert HAADFData to a sea-eco Signal object.

        Args:
            inner_mrad: Inner collection angle in milliradians (default: 45)
            outer_mrad: Outer collection angle in milliradians (default: 150)

        Returns:
            Signal object containing the HAADF image with proper dimensions.

        Raises:
            ImportError: If sea-eco is not installed
            RuntimeError: If calculateADF() hasn't been called yet
        """
        if not SEA_ECO_AVAILABLE:
            raise ImportError(
                "sea-eco package is required for to_signal(). "
                "Install it with: pip install -e /path/to/sea-eco"
            )

        # Ensure ADF has been calculated
        if not hasattr(self, 'adf') or self.adf is None:
            self.calculateADF(inner_mrad=inner_mrad, outer_mrad=outer_mrad)

        # Convert to numpy if needed
        adf = self.adf
        xs = self.xs
        ys = self.ys
        if TORCH_AVAILABLE:
            if hasattr(adf, 'cpu'):
                adf = adf.cpu().numpy()
            if hasattr(xs, 'cpu'):
                xs = xs.cpu().numpy()
            if hasattr(ys, 'cpu'):
                ys = ys.cpu().numpy()

        dimensions = Dimensions([
            Dimension(name='x', space='position', units='Å', values=np.asarray(xs)),
            Dimension(name='y', space='position', units='Å', values=np.asarray(ys)),
        ], nav_dimensions=[0, 1], sig_dimensions=[])

        # Build metadata
        metadata_dict = {
            'General': {'title': 'HAADF Image', 'signal_type': 'HAADF'},
            'Instrument': {
                'beam_energy': float(self.probe.eV),
                'Detectors': {
                    'HAADF': {
                        'name': 'HAADF Detector',
                        'inner_mrad': inner_mrad,
                        'outer_mrad': outer_mrad,
                    }
                },
                'Scan': {'scan_uuid': None}
            },
            'Simulation': {
                'voltage_eV': float(self.probe.eV),
                'wavelength_A': float(self.probe.wavelength),
                'aperture_mrad': float(self.probe.mrad),
                'inner_mrad': inner_mrad,
                'outer_mrad': outer_mrad,
            }
        }
        metadata = GeneralMetadata(metadata_dict)

        return Signal(
            data=np.asarray(adf),
            name='HAADF',
            dimensions=dimensions,
            signal_type='Image',
            metadata=metadata
        )
