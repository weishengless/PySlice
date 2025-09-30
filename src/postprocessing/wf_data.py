"""
Wave function data structure.
"""
from dataclasses import dataclass
import numpy as np ; xp = np
from typing import List, Tuple
from ..multislice.multislice import Probe

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
        wavefunction_data: Complex wavefunction array with shape (probe_positions, time, kx, ky, layer).
    """
    probe_positions: List[Tuple[float, float]]
    time: np.ndarray  # Time in picoseconds (frame # * timestep)
    kxs: np.ndarray    # kx sampling vectors
    kys: np.ndarray    # ky sampling vectors
    xs: np.ndarray
    ys: np.ndarray
    layer: np.ndarray # Layer indices
    wavefunction_data: np.ndarray  # Complex wavefunction array (probe_positions, time, kx, ky, layer)
    probe: Probe

    def plot(self,whichProbe=0,whichTimestep=0):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        array = self.wavefunction_data[whichProbe,whichTimestep,:,:,-1].T # imshow convention: y,x. our convention: x,y
        extent = ( np.amin(self.kxs) , np.amax(self.kxs) , np.amin(self.kys) , np.amax(self.kys) )
        ax.imshow(np.absolute(array)**.25, cmap="inferno", extent=extent)
        plt.show()

    def plot_reciprocalspace(self,whichProbe=0,whichTimestep=0):
        self.plot(whichProbe,whichTimestep)

    def plot_realspace(self,whichProbe=0,whichTimestep=0):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        array = self.wavefunction_data[whichProbe,whichTimestep,:,:,-1].T # imshow convention: y,x. our convention: x,y
        array = np.fft.ifft2(array)
        extent = ( np.amin(self.xs) , np.amax(self.xs) , np.amin(self.ys) , np.amax(self.ys) )
        ax.imshow(np.absolute(array)**.25, cmap="inferno", extent=extent)
        plt.show()

    def defocus(self,dz): # POSITIVE DEFOCUS PUTS BEAM WAIST ABOVE SAMPLE, UNITS OF ANGSTROM
        kx_grid, ky_grid = xp.meshgrid(self.kxs, self.kys, indexing='ij')
        k_squared = kx_grid**2 + ky_grid**2
        P = xp.exp(-1j * xp.pi * self.probe.wavelength * dz * k_squared)
        #if dz>0:
        self.wavefunction_data = P[None,None,:,:,None] * self.wavefunction_data
        #if dz<0:
        #   self.array = xp.fft.ifft2( xp.fft.fft2( self.array ) / P )

    def applyMask(self,radius,realOrReciprocal="reciprocal"):
        if realOrReciprocal == "reciprocal":
            radii = np.sqrt( self.kxs[:,None]**2 + self.kys[None,:]**2 )
            mask = np.zeros(radii.shape)
            mask[radii<radius]=1
            self.wavefunction_data*=mask[None,None,:,:,None]
        else:
            radii = np.sqrt( ( self.xs[:,None] - np.mean(self.xs) )**2 +\
                ( self.ys[None,:] - np.mean(self.ys) )**2 )
            mask = np.zeros(radii.shape)
            mask[radii<radius]=1
            real = np.fft.ifft2(np.fft.ifftshift(self.wavefunction_data,axes=(2,3)),axes=(2,3))
            real *= mask[None,None,:,:,None]
            self.wavefunction_data = np.fft.fftshift(np.fft.fft2(real,axes=(2,3)),axes=(2,3))

