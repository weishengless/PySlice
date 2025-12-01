"""
Wave function data structure.
"""
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Optional
from ..multislice.multislice import Probe
from pathlib import Path

# Optional sea-eco integration
try:
    from pySEA.sea_eco.architecture.base_structure_numpy import (
        Signal, Dimensions, Dimension, GeneralMetadata
    )
    SEA_ECO_AVAILABLE = True
except ImportError:
    SEA_ECO_AVAILABLE = False

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
    cache_dir: Path

    def plot_reciprocal(self,filename=None,whichProbe=0,whichTimestep=0,powerscaling=0.25,extent=None,avg=False,nuke_zerobeam=False):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        if avg:
            # Average over all timesteps
            array = self.array[whichProbe,:,:,:,-1] # Shape: (time, kx, ky)
            if hasattr(array, 'mean'):  # torch tensor
                array = array.mean(dim=0)  # Average over time dimension
            else:  # numpy array
                array = np.mean(array, axis=0)
        else:
            array = self.array[whichProbe,whichTimestep,:,:,-1] # Shape: (kx, ky)

        # Convert kxs and kys to numpy for indexing
        if hasattr(self.kxs, 'cpu'):
            kxs_np = self.kxs.cpu().numpy()
            kys_np = self.kys.cpu().numpy()
        else:
            kxs_np = np.asarray(self.kxs)
            kys_np = np.asarray(self.kys)

        # If extent is provided, slice the data
        if extent is not None:
            kx_min, kx_max, ky_min, ky_max = extent

            # Find indices for the requested extent
            kx_mask = (kxs_np >= kx_min) & (kxs_np <= kx_max)
            ky_mask = (kys_np >= ky_min) & (kys_np <= ky_max)

            # Slice the array and coordinate arrays
            array = array[kx_mask, :][:, ky_mask]
            actual_extent = (kxs_np[kx_mask][0], kxs_np[kx_mask][-1],
                           kys_np[ky_mask][0], kys_np[ky_mask][-1])
        else:
            # Use full extent
            kxs_min = float(kxs_np.min())
            kxs_max = float(kxs_np.max())
            kys_min = float(kys_np.min())
            kys_max = float(kys_np.max())
            actual_extent = (kxs_min, kxs_max, kys_min, kys_max)

        # Transpose for imshow convention
        array = array.T  # imshow convention: y,x. our convention: x,y
        if nuke_zerobeam:
            array[np.argmin(np.absolute(kys_np)),np.argmin(np.absolute(kxs_np))]=0

        # Convert to numpy array if it's a tensor
        # Apply powerscaling to intensity (|Ψ|²)
        img_data = (xp.absolute(array)**2)**powerscaling
        if hasattr(img_data, 'cpu'):
            img_data = img_data.cpu().numpy()
        elif hasattr(img_data, '__array__'):
            img_data = np.asarray(img_data)
        ax.imshow(img_data, cmap="inferno", extent=actual_extent, origin='lower')
        ax.set_xlabel("kx ($\\AA^{-1}$)")
        ax.set_ylabel("ky ($\\AA^{-1}$)")

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    plot = plot_reciprocal

    def plot_phase(self,filename=None,whichProbe=0,whichTimestep=0,extent=None,avg=False):
        """
        Plot the phase of the wavefunction in real space.

        Args:
            whichProbe: Probe index
            whichTimestep: Timestep index
            extent: Optional (xmin, xmax, ymin, ymax) to zoom
            avg: If True, average over all timesteps before plotting
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        # Get array (with or without averaging)
        if avg:
            array = self.array[whichProbe,:,:,:,-1] # Shape: (time, kx, ky)
            if hasattr(array, 'mean'):  # torch tensor
                array = array.mean(dim=0)  # Average over time dimension
            else:  # numpy array
                array = np.mean(array, axis=0)
        else:
            array = self.array[whichProbe,whichTimestep,:,:,-1]

        # Transform to real space
        array = xp.fft.ifft2(array)
        xs_np = np.asarray(self.xs)
        ys_np = np.asarray(self.ys)

        # If extent is provided, slice the data
        if extent is not None:
            x_min, x_max, y_min, y_max = extent

            # Find indices for the requested extent
            x_mask = (xs_np >= x_min) & (xs_np <= x_max)
            y_mask = (ys_np >= y_min) & (ys_np <= y_max)

            # Slice the array
            array = array[x_mask, :][:, y_mask]
            actual_extent = (xs_np[x_mask][0], xs_np[x_mask][-1],
                           ys_np[y_mask][0], ys_np[y_mask][-1])
        else:
            # Use full extent
            actual_extent = (float(xs_np.min()), float(xs_np.max()),
                           float(ys_np.min()), float(ys_np.max()))

        # Transpose for imshow convention
        array = array.T  # imshow convention: y,x. our convention: x,y

        # Get phase
        phase_data = xp.angle(array)
        if hasattr(phase_data, 'cpu'):
            phase_data = phase_data.cpu().numpy()
        elif hasattr(phase_data, '__array__'):
            phase_data = np.asarray(phase_data)

        # Plot with phase colormap
        im = ax.imshow(phase_data, cmap='hsv', extent=actual_extent, origin='lower',
                       vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im, ax=ax, label='Phase (radians)')
        ax.set_title('Phase in real space')
        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_realspace(self,whichProbe=0,whichTimestep=0,extent=None,avg=False):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        # Get array (with or without averaging)
        if avg:
            array = self.array[whichProbe,:,:,:,-1] # Shape: (time, kx, ky)
            if hasattr(array, 'mean'):  # torch tensor
                array = array.mean(dim=0)  # Average over time dimension
            else:  # numpy array
                array = np.mean(array, axis=0)
        else:
            array = self.array[whichProbe,whichTimestep,:,:,-1]

        array = array.T # imshow convention: y,x. our convention: x,y
        array = xp.fft.ifft2(array)

        # Use provided extent or calculate from data
        if extent is None:
            extent = ( np.amin(self.xs) , np.amax(self.xs) , np.amin(self.ys) , np.amax(self.ys) )

        # Convert to numpy array if it's a tensor
        img_data = xp.absolute(array)**.25
        if hasattr(img_data, 'cpu'):
            img_data = img_data.cpu().numpy()
        elif hasattr(img_data, '__array__'):
            img_data = np.asarray(img_data)

        ax.imshow( img_data, cmap="inferno", extent=extent )
        plt.show()

    def propagate_free_space(self,dz): # UNITS OF ANGSTROM
        kx_grid, ky_grid = xp.meshgrid(self.kxs, self.kys, indexing='ij')
        k_squared = kx_grid**2 + ky_grid**2
        P = xp.exp(-1j * xp.pi * self.probe.wavelength * dz * k_squared)
        if TORCH_AVAILABLE and isinstance(self.array, torch.Tensor):
            P = P.to(self.array.device)
        #if dz>0:
        self.array = P[None,None,:,:,None] * self.array

    def applyMask(self, radius, realOrReciprocal="reciprocal"):
        if realOrReciprocal == "reciprocal":
            radii = xp.sqrt( self.kxs[:,None]**2 + self.kys[None,:]**2 )
            mask = xp.zeros(radii.shape, device=self.array.device if TORCH_AVAILABLE else None)
            mask[radii<radius]=1
            self.array*=mask[None,None,:,:,None]
        else:
            radii = np.sqrt( ( self.xs[:,None] - np.mean(self.xs) )**2 +\
                ( self.ys[None,:] - np.mean(self.ys) )**2 )
            mask = xp.zeros(radii.shape, device=self.array.device if TORCH_AVAILABLE else None)
            mask[radii<radius]=1
            kwarg = {"dim":(2,3)} if TORCH_AVAILABLE else {"axes":(2,3)}
            real = xp.fft.ifft2(xp.fft.ifftshift(self.array,**kwarg),**kwarg)
            real *= mask[None,None,:,:,None]
            self.array = xp.fft.fftshift(xp.fft.fft2(real,**kwarg),**kwarg)

    def to_signal(self, layer_index: int = -1, probe_index: Optional[int] = None) -> 'Signal':
        """
        Convert WFData to a sea-eco Signal object.

        Args:
            layer_index: Which layer to export (default: -1, last layer)
            probe_index: Which probe position to export. If None, exports all probes.

        Returns:
            Signal object containing the wavefunction data with proper dimensions

        Raises:
            ImportError: If sea-eco is not installed
        """
        if not SEA_ECO_AVAILABLE:
            raise ImportError(
                "sea-eco package is required for to_signal(). "
                "Install it with: pip install -e /path/to/sea-eco"
            )

        # Convert array to numpy if needed
        array = self.array
        if TORCH_AVAILABLE and hasattr(array, 'cpu'):
            array = array.cpu().numpy()

        # Helper to convert tensors to numpy
        def to_numpy(x):
            if hasattr(x, 'cpu'):
                return x.cpu().numpy()
            return np.asarray(x)

        # Convert coordinate arrays
        time_arr = to_numpy(self.time)
        kxs_arr = to_numpy(self.kxs)
        kys_arr = to_numpy(self.kys)

        # Extract the specified layer
        if probe_index is not None:
            # Single probe: shape becomes (time, kx, ky)
            data = array[probe_index, :, :, :, layer_index]
            probe_pos = self.probe_positions[probe_index]

            dimensions = Dimensions([
                Dimension(name='time', space='temporal', units='ps',
                         values=time_arr),
                Dimension(name='kx', space='scattering', units='Å⁻¹',
                         values=kxs_arr),
                Dimension(name='ky', space='scattering', units='Å⁻¹',
                         values=kys_arr),
            ], nav_dimensions=[0], sig_dimensions=[1, 2])
        else:
            # All probes: shape is (probe, time, kx, ky)
            data = array[:, :, :, :, layer_index]
            probe_pos = self.probe_positions

            dimensions = Dimensions([
                Dimension(name='probe', space='position', units='Å',
                         values=np.arange(len(self.probe_positions))),
                Dimension(name='time', space='temporal', units='ps',
                         values=time_arr),
                Dimension(name='kx', space='scattering', units='Å⁻¹',
                         values=kxs_arr),
                Dimension(name='ky', space='scattering', units='Å⁻¹',
                         values=kys_arr),
            ], nav_dimensions=[0, 1], sig_dimensions=[2, 3])

        # Build metadata from simulation parameters
        metadata_dict = {
            'General': {
                'title': 'Multislice Wavefunction',
                'signal_type': 'Wavefunction'
            },
            'Instrument': {
                'beam_energy': float(self.probe.eV),
                'Detectors': {
                    'Simulated': {
                        'name': 'Multislice Simulation',
                        'voltage_eV': float(self.probe.eV),
                        'wavelength_A': float(self.probe.wavelength),
                        'aperture_mrad': float(self.probe.mrad),
                    }
                },
                'Scan': {
                    'scan_uuid': None,
                }
            },
            'Simulation': {
                'voltage_eV': float(self.probe.eV),
                'wavelength_A': float(self.probe.wavelength),
                'aperture_mrad': float(self.probe.mrad),
                'layer_index': int(layer_index),
                'probe_positions': [list(p) for p in self.probe_positions],
            }
        }
        metadata = GeneralMetadata(metadata_dict)

        return Signal(
            data=data,
            name='WFData',
            dimensions=dimensions,
            signal_type='Diffraction',
            metadata=metadata
        )
