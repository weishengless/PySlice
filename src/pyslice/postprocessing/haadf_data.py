"""
HAADF (High Angle Annular Dark Field) data structure.
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import logging
from .wf_data import WFData
from ..data import Signal, Dimensions, Dimension, GeneralMetadata

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


class HAADFData(Signal):
    """
    Data structure for HAADF (High Angle Annular Dark Field) imaging data.

    Inherits from Signal for sea-eco compatibility.

    Attributes:
        probe_positions: Array of (x,y) probe positions in Angstroms.
        xs: x coordinates of the HAADF image.
        ys: y coordinates of the HAADF image.
        adf: The computed ADF image (x × y).
        probe: Probe object with beam parameters.
        cache_dir: Path to cache directory.
    """

    def __init__(self, wf_data: WFData) -> None:
        """
        Initialize HAADFData from WFData.

        Args:
            wf_data: WFData object containing wavefunction data
        """
        # Copy needed attributes from WFData (raw tensors for GPU ops)
        self.probe_positions = wf_data.probe_positions
        self._kxs = wf_data._kxs
        self._kys = wf_data._kys
        self.probe = wf_data.probe
        self.cache_dir = wf_data.cache_dir

        # Store reference to source WFData array for ADF calculation
        self._wf_array = wf_data.array

        # Initialize ADF as None, will be computed by calculateADF
        self._array = None
        self._xs = None
        self._ys = None

        # Build placeholder dimensions (will be updated after calculateADF)
        dimensions = Dimensions([
            Dimension(name='x', space='position', units='Å', values=np.array([0])),
            Dimension(name='y', space='position', units='Å', values=np.array([0])),
        ], nav_dimensions=[0, 1], sig_dimensions=[])

        # Build metadata
        metadata_dict = {
            'General': {
                'title': 'HAADF Image',
                'signal_type': 'HAADF'
            },
            'Simulation': {
                'voltage_eV': float(self.probe.eV),
                'wavelength_A': float(self.probe.wavelength),
                'aperture_mrad': float(self.probe.mrad),
                'probe_positions': [list(p) for p in self.probe_positions],
            }
        }
        metadata = GeneralMetadata(metadata_dict)

        # Initialize Signal base class
        super().__init__(
            data=None,  # We'll override the data property
            name='HAADFData',
            dimensions=dimensions,
            signal_type='Image',
            metadata=metadata
        )

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
    def adf(self):
        """Backward compatible alias for internal ADF array."""
        return self._array

    @adf.setter
    def adf(self, value):
        self._array = value

    @property
    def array(self):
        """Alias for adf (backward compatibility)."""
        return self._array

    def __getattr__(self, name):
        """Auto-convert coordinate arrays from tensor to numpy on access."""
        coord_attrs = {'kxs', 'kys', 'xs', 'ys'}
        if name in coord_attrs:
            raw = object.__getattribute__(self, f'_{name}')
            if raw is None:
                return None
            if hasattr(raw, 'cpu'):
                return raw.cpu().numpy()
            return np.asarray(raw)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def calculateADF(self, inner_mrad: float = 45, outer_mrad: float = 150, preview: bool = False) -> np.ndarray:
        """
        Calculate the ADF (Annular Dark Field) image.

        Args:
            inner_mrad: Inner collection angle in milliradians (default: 45)
            outer_mrad: Outer collection angle in milliradians (default: 150)
            preview: If True, show a preview of the first exit wave with mask

        Returns:
            ADF image array (x × y)
        """
        # Use float_dtype to ensure MPS compatibility (float32 on MPS, float64 otherwise)
        self._xs = xp.asarray(sorted(list(set(self.probe_positions[:,0]))), dtype=float_dtype)
        self._ys = xp.asarray(sorted(list(set(self.probe_positions[:,1]))), dtype=float_dtype)
        self._array = xp.zeros((len(self._xs), len(self._ys)), dtype=float_dtype)

        q = xp.sqrt(self._kxs[:,None]**2 + self._kys[None,:]**2)
        radius_inner = (inner_mrad * 1e-3) / self.probe.wavelength
        radius_outer = (outer_mrad * 1e-3) / self.probe.wavelength

        mask = xp.zeros(q.shape, device=self._wf_array.device if TORCH_AVAILABLE else None, dtype=float_dtype)
        mask[q >= radius_inner] = 1
        mask[q >= radius_outer] = 0

        probe_positions = xp.asarray(self.probe_positions, dtype=float_dtype)

        for i, x in enumerate(self._xs):
            for j, y in enumerate(self._ys):
                dxy = xp.sqrt(xp.sum((probe_positions - xp.asarray([x, y], dtype=float_dtype)[None, :]) ** 2, axis=1))
                p = xp.argmin(dxy)
                exits = self._wf_array[p, :, :, :, -1]  # which probe position, all frames, kx, ky, last layer

                if preview and i == 0 and j == 0:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    preview_data = xp.mean(xp.absolute(exits), axis=0) ** .1 * (1 - mask)
                    if TORCH_AVAILABLE:
                        preview_data = preview_data.cpu().numpy()
                    ax.imshow(preview_data, cmap="inferno")
                    plt.show()

                collected = xp.mean(xp.sum(xp.absolute(exits * mask[None, :, :]), axis=(1, 2)))
                self._array[i, j] = collected

        # Update dimensions with computed xs, ys
        def to_numpy(x):
            if hasattr(x, 'cpu'):
                return x.cpu().numpy()
            return np.asarray(x)

        xs_np = to_numpy(self._xs)
        ys_np = to_numpy(self._ys)

        self._local_dimensions = Dimensions([
            Dimension(name='x', space='position', units='Å', values=xs_np),
            Dimension(name='y', space='position', units='Å', values=ys_np),
        ], nav_dimensions=[0, 1], sig_dimensions=[])

        # Update metadata with detector settings
        if hasattr(self.metadata, 'Simulation'):
            self.metadata.Simulation.inner_mrad = inner_mrad
            self.metadata.Simulation.outer_mrad = outer_mrad

        return self.data  # Return numpy array for backward compatibility

    def plot(self, filename=None):
        """
        Plot the HAADF image.

        Args:
            filename: If provided, save plot to this file instead of displaying
        """
        import matplotlib.pyplot as plt

        if self._array is None:
            raise RuntimeError("calculateADF() must be called before plotting")

        fig, ax = plt.subplots()
        array = self._array.T  # imshow convention: y,x. our convention: x,y

        if TORCH_AVAILABLE and hasattr(array, 'cpu'):
            array = array.cpu().numpy()
        if TORCH_AVAILABLE and hasattr(self._xs, 'cpu'):
            xs = self._xs.cpu().numpy()
            ys = self._ys.cpu().numpy()
        else:
            xs = np.asarray(self._xs)
            ys = np.asarray(self._ys)

        extent = (np.amin(xs), np.amax(xs), np.amin(ys), np.amax(ys))
        ax.imshow(array, cmap="inferno", extent=extent)
        ax.set_xlabel("x ($\\AA$)")
        ax.set_ylabel("y ($\\AA$)")

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

