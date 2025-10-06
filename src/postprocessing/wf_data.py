"""
Wave function data structure.
"""
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
from ..multislice.multislice import Probe

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
class WFData:
    """
    Data structure for wave function data with format: probe_positions, frame, kx, ky, layer.

    Attributes:
        probe_positions: List of (x,y) probe positions in Angstroms.
        frame: Time array (frame # * timestep) in picoseconds.
        kx: kx sampling vectors.
        ky: ky sampling vectors.
        layer: Layer indices for multi-layer calculations.
        array: Complex wavefunction array with shape (probe_positions, time, kx, ky, layer).
    """
    probe_positions: List[Tuple[float, float]]
    time: np.ndarray  # Time in picoseconds (frame # * timestep)
    kxs: np.ndarray    # kx sampling vectors
    kys: np.ndarray    # ky sampling vectors
    xs: np.ndarray
    ys: np.ndarray
    layer: np.ndarray # Layer indices
    array: np.ndarray  # Complex reciprocal-space wavefunction array (probe_positions, time, kx, ky, layer)
    probe: Probe

    def plot(self,whichProbe=0,whichTimestep=0,powerscaling=0.25,filename=""):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        array = self.array[whichProbe,whichTimestep,:,:,-1].T # imshow convention: y,x. our convention: x,y
        extent = ( xp.amin(self.kxs) , xp.amax(self.kxs) , xp.amin(self.kys) , xp.amax(self.kys) )
        ax.imshow( xp.absolute(array)**powerscaling, cmap="inferno", extent=extent )
        if len(filename)>3:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_reciprocalspace(self,whichProbe=0,whichTimestep=0):
        self.plot(whichProbe,whichTimestep)

    def plot_realspace(self,whichProbe=0,whichTimestep=0):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        array = self.array[whichProbe,whichTimestep,:,:,-1].T # imshow convention: y,x. our convention: x,y
        array = xp.fft.ifft2(array)
        extent = ( np.amin(self.xs) , np.amax(self.xs) , np.amin(self.ys) , np.amax(self.ys) )
        ax.imshow( xp.absolute(array)**.25, cmap="inferno", extent=extent )
        plt.show()

    def propagate_free_space(self,dz): # UNITS OF ANGSTROM
        kx_grid, ky_grid = xp.meshgrid(self.kxs, self.kys, indexing='ij')
        k_squared = kx_grid**2 + ky_grid**2
        P = xp.exp(-1j * xp.pi * self.probe.wavelength * dz * k_squared)
        #if dz>0:
        self.array = P[None,None,:,:,None] * self.array

    def applyMask(self,radius,realOrReciprocal="reciprocal"):
        if realOrReciprocal == "reciprocal":
            radii = xp.sqrt( self.kxs[:,None]**2 + self.kys[None,:]**2 )
            mask = np.zeros(radii.shape)
            mask[radii<radius]=1
            self.array*=mask[None,None,:,:,None]
        else:
            radii = np.sqrt( ( self.xs[:,None] - np.mean(self.xs) )**2 +\
                ( self.ys[None,:] - np.mean(self.ys) )**2 )
            mask = xp.zeros(radii.shape)
            mask[radii<radius]=1
            kwarg = {"dim":(2,3)} if TORCH_AVAILABLE else {"axes":(2,3)}
            real = xp.fft.ifft2(xp.fft.ifftshift(self.array,**kwarg),**kwarg)
            real *= mask[None,None,:,:,None]
            self.array = xp.fft.fftshift(xp.fft.fft2(real,**kwarg),**kwarg)

