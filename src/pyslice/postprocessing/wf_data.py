"""
Wave function data structure.
"""
import numpy as np
from typing import List, Tuple, Optional
from ..multislice.multislice import Probe
from ..data import Signal, Dimensions, Dimension, GeneralMetadata
from pathlib import Path

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


class WFData(Signal):
    """
    Data structure for wave function data with format: probe_positions, frame, kx, ky, layer.

    Inherits from Signal for sea-eco compatibility.

    Attributes:
        probe_positions: List of (x,y) probe positions in Angstroms.
        time: Time array (frame # * timestep) in picoseconds.
        kxs: kx sampling vectors.
        kys: ky sampling vectors.
        xs: x real-space coordinates.
        ys: y real-space coordinates.
        layer: Layer indices for multi-layer calculations.
        array: Complex wavefunction array with shape (probe_positions, time, kx, ky, layer).
        probe: Probe object with beam parameters.
        cache_dir: Path to cache directory.
    """

    def __init__(
        self,
        probe_positions: List[Tuple[float, float]],
        time: np.ndarray,
        kxs: np.ndarray,
        kys: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        layer: np.ndarray,
        array: np.ndarray,
        probe: Probe,
        cache_dir: Path,
    ):
        # Store raw attributes (may be tensors for GPU operations)
        self.probe_positions = probe_positions
        self._time = time
        self._kxs = kxs
        self._kys = kys
        self._xs = xs
        self._ys = ys
        self._layer = layer
        self.probe = probe
        self.cache_dir = cache_dir

        # Helper to convert tensors to numpy for Dimensions
        def to_numpy(x):
            if hasattr(x, 'cpu'):
                return x.cpu().numpy()
            return np.asarray(x)

        # Build Dimensions for Signal
        time_arr = to_numpy(time)
        kxs_arr = to_numpy(kxs)
        kys_arr = to_numpy(kys)
        layer_arr = to_numpy(layer) if layer is not None else np.array([0])

        dimensions = Dimensions([
            Dimension(name='probe', space='position',
                     values=np.arange(len(probe_positions))),
            Dimension(name='time', space='temporal', units='ps',
                     values=time_arr),
            Dimension(name='kx', space='scattering', units='Å⁻¹',
                     values=kxs_arr),
            Dimension(name='ky', space='scattering', units='Å⁻¹',
                     values=kys_arr),
            Dimension(name='layer', space='position',
                     values=layer_arr),
        ], nav_dimensions=[0, 1], sig_dimensions=[2, 3, 4])

        # Build metadata from simulation parameters
        # Flatten probe_positions for HDF5 compatibility, store n_probes to reshape on load
        pp_array = np.array(probe_positions).flatten().tolist()
        metadata_dict = {
            'General': {
                'title': 'Multislice Wavefunction',
                'signal_type': 'Wavefunction'
            },
            'Simulation': {
                'voltage_eV': float(probe.eV),
                'wavelength_A': float(probe.wavelength),
                'aperture_mrad': float(probe.mrad),
                'probe_positions': pp_array,
                'n_probes': len(probe_positions),
            }
        }
        metadata = GeneralMetadata(metadata_dict)

        # Initialize Signal base class (must be called before setting _array
        # because Signal.__init__ sets self.data which calls our setter)
        super().__init__(
            data=None,
            name='WFData',
            dimensions=dimensions,
            signal_type='Diffraction',
            metadata=metadata
        )

        # Store array AFTER super().__init__ to avoid being overwritten
        self._array = array

    @property
    def data(self):
        """Lazy conversion to numpy for Signal compatibility."""
        if self._array is None:
            return None
        if hasattr(self._array, 'cpu'):
            return self._array.cpu().numpy()
        return np.asarray(self._array)

    @data.setter
    def data(self, value):
        self._array = value

    @property
    def array(self):
        """Backward compatible alias for internal array (may be tensor or numpy)."""
        return self._array

    @array.setter
    def array(self, value):
        self._array = value

    def __getattr__(self, name):
        """Auto-convert coordinate arrays from tensor to numpy on access."""
        coord_attrs = {'time', 'kxs', 'kys', 'xs', 'ys', 'layer'}
        if name in coord_attrs:
            raw = object.__getattribute__(self, f'_{name}')
            if raw is None:
                return None
            if hasattr(raw, 'cpu'):
                return raw.cpu().numpy()
            return np.asarray(raw)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def to_hdf5_group(self, parent_group, force_datasets=['data'], name=None):
        """Override to convert tensors to numpy and exclude non-serializable attrs."""
        # Put numpy data directly in __dict__ for serialization
        # (to_dict pulls from __dict__, not properties)
        if hasattr(self._array, 'cpu'):
            self.__dict__['data'] = self._array.cpu().numpy()
        else:
            self.__dict__['data'] = np.asarray(self._array)

        # Temporarily remove non-serializable attributes (but keep _array for data property)
        orig_probe = self.probe
        orig_cache_dir = self.cache_dir
        orig_probe_positions = self.probe_positions
        orig_kxs = self._kxs
        orig_kys = self._kys
        orig_time = self._time
        orig_xs = self._xs
        orig_ys = self._ys
        orig_layer = self._layer

        del self.probe
        del self.cache_dir
        del self.probe_positions
        del self._kxs
        del self._kys
        del self._time
        del self._xs
        del self._ys
        del self._layer
        # Note: DON'T delete _array - the data property depends on it for hasattr check

        # Call parent implementation
        result = super().to_hdf5_group(parent_group, force_datasets=force_datasets, name=name)

        # Restore all original attributes and clean up
        del self.__dict__['data']  # Remove the temp data from __dict__
        self.probe = orig_probe
        self.cache_dir = orig_cache_dir
        self.probe_positions = orig_probe_positions
        self._kxs = orig_kxs
        self._kys = orig_kys
        self._time = orig_time
        self._xs = orig_xs
        self._ys = orig_ys
        self._layer = orig_layer

        return result

    def plot_reciprocal(self,filename=None,whichProbe=0,whichTimestep=0,powerscaling=0.25,extent=None,avg=False,nuke_zerobeam=False):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        if avg:
            # Average over all timesteps
            array = self._array[whichProbe,:,:,:,-1] # Shape: (time, kx, ky)
            if hasattr(array, 'mean'):  # torch tensor
                array = array.mean(dim=0)  # Average over time dimension
            else:  # numpy array
                array = np.mean(array, axis=0)
        else:
            array = self._array[whichProbe,whichTimestep,:,:,-1] # Shape: (kx, ky)

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
            array = self._array[whichProbe,:,:,:,-1] # Shape: (time, kx, ky)
            if hasattr(array, 'mean'):  # torch tensor
                array = array.mean(dim=0)  # Average over time dimension
            else:  # numpy array
                array = np.mean(array, axis=0)
        else:
            array = self._array[whichProbe,whichTimestep,:,:,-1]

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
            array = self._array[whichProbe,:,:,:,-1] # Shape: (time, kx, ky)
            if hasattr(array, 'mean'):  # torch tensor
                array = array.mean(dim=0)  # Average over time dimension
            else:  # numpy array
                array = np.mean(array, axis=0)
        else:
            array = self._array[whichProbe,whichTimestep,:,:,-1]

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
        kx_grid, ky_grid = xp.meshgrid(self._kxs, self._kys, indexing='ij')
        k_squared = kx_grid**2 + ky_grid**2
        P = xp.exp(-1j * xp.pi * self.probe.wavelength * dz * k_squared)
        if TORCH_AVAILABLE and isinstance(self._array, torch.Tensor):
            P = P.to(self._array.device)
        #if dz>0:
        self._array = P[None,None,:,:,None] * self._array

    def applyMask(self, radius, realOrReciprocal="reciprocal"):
        if realOrReciprocal == "reciprocal":
            radii = xp.sqrt( self._kxs[:,None]**2 + self._kys[None,:]**2 )
            mask = xp.zeros(radii.shape, device=self._array.device if TORCH_AVAILABLE else None)
            mask[radii<radius]=1
            self._array*=mask[None,None,:,:,None]
        else:
            # Use numpy for _xs/_ys since they're numpy arrays, then convert result
            radii_np = np.sqrt( ( self._xs[:,None] - np.mean(self._xs) )**2 +\
                ( self._ys[None,:] - np.mean(self._ys) )**2 )
            if TORCH_AVAILABLE:
                radii = xp.tensor(radii_np, dtype=self._array.real.dtype, device=self._array.device)
            else:
                radii = radii_np
            mask = xp.zeros(radii.shape, device=self._array.device if TORCH_AVAILABLE else None)
            mask[radii<radius]=1
            kwarg = {"dim":(2,3)} if TORCH_AVAILABLE else {"axes":(2,3)}
            real = xp.fft.ifft2(xp.fft.ifftshift(self._array,**kwarg),**kwarg)
            real *= mask[None,None,:,:,None]
            self._array = xp.fft.fftshift(xp.fft.fft2(real,**kwarg),**kwarg)

