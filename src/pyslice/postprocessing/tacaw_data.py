"""
Core data structure for TACAW EELS calculations.
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import logging, os
from .wf_data import WFData
from ..data import Signal, Dimensions, Dimension, GeneralMetadata
from pyslice.backend import to_cpu

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


class TACAWData(Signal):
    """
    Data structure for storing TACAW EELS results with format: probe_positions, frequency, kx, ky.

    Inherits from Signal for sea-eco compatibility.

    Attributes:
        probe_positions: List of (x,y) probe positions in Angstroms.
        frequencies: Frequencies in THz.
        kxs: kx sampling vectors (e.g., in Å⁻¹).
        kys: ky sampling vectors (e.g., in Å⁻¹).
        xs: x real-space coordinates.
        ys: y real-space coordinates.
        intensity: Intensity array |Ψ(ω,q)|² (probe_positions, frequency, kx, ky).
        probe: Probe object with beam parameters.
        cache_dir: Path to cache directory.
    """

    def __init__(self, wf_data: WFData, layer_index: int = None, keep_complex: bool = False) -> None:
        """
        Initialize TACAWData from WFData by performing FFT.

        Args:
            wf_data: WFData object containing wavefunction data
            layer_index: Index of the layer to compute FFT for (default: last layer)
            keep_complex: If True, keep complex FFT result instead of intensity
        """
        # Copy needed attributes from WFData (raw tensors for GPU ops)
        self.probe_positions = wf_data.probe_positions
        self._time = wf_data._time
        self._kxs = wf_data._kxs
        self._kys = wf_data._kys
        self._xs = wf_data._xs
        self._ys = wf_data._ys
        self._layer = wf_data._layer
        self.probe = wf_data.probe
        self.cache_dir = wf_data.cache_dir
        self.keep_complex = keep_complex

        # Store reference to source WFData array for FFT computation
        self._wf_array = wf_data.array

        # Initialize intensity as None, will be set by fft_from_wf_data
        self._array = None
        self._frequencies = None

        # Perform FFT to compute intensity
        self._fft_from_wf_data(layer_index)

        # Save computed values before super().__init__
        computed_array = self._array
        computed_frequencies = self._frequencies

        # Helper to convert tensors to numpy for Dimensions
        def to_numpy(x):
            if hasattr(x, 'cpu'):
                return x.cpu().numpy()
            return np.asarray(x)

        # Build Dimensions for Signal
        freq_arr = to_numpy(self._frequencies)
        kxs_arr = to_numpy(self._kxs)
        kys_arr = to_numpy(self._kys)

        dimensions = Dimensions([
            Dimension(name='probe', space='position',
                     values=np.arange(len(self.probe_positions))),
            Dimension(name='frequency', space='spectral', units='THz',
                     values=freq_arr),
            Dimension(name='kx', space='scattering', units='Å⁻¹',
                     values=kxs_arr),
            Dimension(name='ky', space='scattering', units='Å⁻¹',
                     values=kys_arr),
        ], nav_dimensions=[0, 1], sig_dimensions=[2, 3])

        # Build metadata
        metadata_dict = {
            'General': {
                'title': 'TACAW Intensity',
                'signal_type': 'TACAW'
            },
            'Simulation': {
                'voltage_eV': float(self.probe.eV),
                'wavelength_A': float(self.probe.wavelength),
                'aperture_mrad': float(self.probe.mrad),
                'probe_positions': [list(p) for p in self.probe_positions],
            }
        }
        metadata = GeneralMetadata(metadata_dict)

        # Initialize Signal base class (this will set self.data = None)
        super().__init__(
            data=None,
            name='TACAWData',
            dimensions=dimensions,
            signal_type='2D-EELS',
            metadata=metadata
        )

        # Restore computed values AFTER super().__init__
        self._array = computed_array
        self._frequencies = computed_frequencies

    def __getattr__(self, name):
        """Auto-convert coordinate arrays from tensor to numpy on access."""
        coord_attrs = {'time', 'kxs', 'kys', 'xs', 'ys', 'layer', 'frequencies'}
        if name in coord_attrs:
            raw = object.__getattribute__(self, f'_{name}')
            if raw is None:
                return None
            if hasattr(raw, 'cpu'):
                return raw.cpu().numpy()
            return np.asarray(raw)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

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
    def intensity(self):
        """Backward compatible alias for internal intensity array."""
        return self._array

    @intensity.setter
    def intensity(self, value):
        self._array = value

    @property
    def array(self):
        """Alias for intensity (backward compatibility with WFData interface)."""
        return self._array

    def _fft_from_wf_data(self, layer_index: int = None):
        """
        Perform FFT along the time axis for a specific layer to convert to TACAW data.
        This implements the JACR method: Ψ(t,q,r) → |Ψ(ω,q,r)|² via FFT.

        Args:
            layer_index: Index of the layer to compute FFT for (default: last layer)
        """
        if os.path.exists(self.cache_dir / "tacaw.npy"):
            self._frequencies = np.load(self.cache_dir / "tacaw_freq.npy")
            self._array = np.load(self.cache_dir / "tacaw.npy")
            if TORCH_AVAILABLE:
                self._frequencies = xp.Tensor(self._frequencies)
                self._array = xp.Tensor(self._array)
            return

        # Default to last layer if not specified
        if layer_index is None:
            layer_index = len(self._layer) - 1

        # Validate layer index
        if layer_index < 0 or layer_index >= len(self._layer):
            raise ValueError(f"layer_index {layer_index} out of range [0, {len(self._layer)-1}]")

        # Compute frequencies from time sampling
        n_freq = len(self._time)
        dt = self._time[1] - self._time[0]
        self._frequencies = np.fft.fftfreq(n_freq, d=dt)
        self._frequencies = np.fft.fftshift(self._frequencies)

        # Extract wavefunction data for the specified layer
        # Shape: (probe_positions, time, kx, ky, layer)
        wf_layer = self._wf_array[:, :, :, :, layer_index]

        # Perform FFT along time axis (axis=1) for each probe position and k-point
        # Following abeels.py approach: subtract mean to avoid high zero-frequency peak
        if TORCH_AVAILABLE and hasattr(wf_layer, 'dim'):  # Check if it's a torch tensor
            wf_mean = torch.mean(wf_layer, dim=1, keepdim=True)
            wf_fft = torch.fft.fft(wf_layer - wf_mean, dim=1)
            wf_fft = torch.fft.fftshift(wf_fft, dim=1)
        else:
            wf_mean = np.mean(wf_layer, axis=1, keepdims=True)
            wf_fft = np.fft.fft(wf_layer - wf_mean, axis=1)
            wf_fft = np.fft.fftshift(wf_fft, axes=1)

        # Compute intensity |Ψ(ω,q)|² from the frequency-domain wavefunction
        if TORCH_AVAILABLE and hasattr(wf_fft, 'dim'):  # Check if it's a torch tensor
            if self.keep_complex:
                self._array = wf_fft
            else:
                self._array = torch.abs(wf_fft)**2
        else:
            if self.keep_complex:
                self._array = wf_fft
            else:
                self._array = np.abs(wf_fft)**2

        np.save(self.cache_dir / "tacaw_freq.npy", self._frequencies)
        np.save(self.cache_dir / "tacaw.npy", self._array.detach().cpu().numpy() if TORCH_AVAILABLE and hasattr(self._array, 'cpu') else self._array)

    # Keep fft_from_wf_data as public alias for backward compatibility
    def fft_from_wf_data(self, layer_index: int = None):
        """Public alias for _fft_from_wf_data (backward compatibility)."""
        self._fft_from_wf_data(layer_index)

    def spectrum(self, probe_index: int = None) -> np.ndarray:
        """
        Extract spectrum for a specific probe position by summing over all k-space.

        Args:
            probe_index: Index of probe position (default: 0). If None, averages over all probes.

        Returns:
            Spectrum array (frequency intensity)
        """
        if probe_index is None:
            # Average over all probe positions
            all_spectra = []
            for i in range(len(self.probe_positions)):
                probe_intensity = self._array[i]  # Shape: (frequency, kx, ky)
                spectrum = xp.sum(probe_intensity, axis=(1, 2))  # Sum over kx, ky
                all_spectra.append(spectrum)

            # Average all spectra
            if TORCH_AVAILABLE and hasattr(all_spectra[0], 'cpu'):
                all_spectra = [s.cpu().numpy() for s in all_spectra]
            spectrum = np.mean(all_spectra, axis=0)
        else:
            if probe_index >= len(self.probe_positions):
                raise ValueError(f"Probe index {probe_index} out of range")

            # Sum intensity data over all k-space for this probe position
            probe_intensity = self._array[probe_index]  # Shape: (frequency, kx, ky)
            spectrum = xp.sum(probe_intensity, axis=(1, 2))  # Sum over kx, ky

            # Convert to numpy if PyTorch tensor
            if TORCH_AVAILABLE and hasattr(spectrum, 'cpu'):
                spectrum = spectrum.cpu().numpy()

        return spectrum

    def spectrum_image(self, frequency: float, probe_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Extract spectrum image at a specific frequency showing intensity in real space (probe positions).

        Args:
            frequency: Frequency value in THz
            probe_indices: List of probe indices to include (default: all probes)

        Returns:
            Spectrum intensity for each probe position (real space map)
        """
        # Find closest frequency index
        freq_idx = np.argmin(np.abs(self.frequencies - frequency))

        # Use all probes if none specified
        if probe_indices is None:
            probe_indices = list(range(len(self.probe_positions)))

        # Extract intensity at this frequency for each selected probe position
        spectrum_intensities = []
        for probe_idx in probe_indices:
            # Sum intensity data over all k-space for this probe at this frequency
            probe_intensity = self._array[probe_idx, freq_idx, :, :]

            # Sum over k-space using appropriate method
            if TORCH_AVAILABLE and hasattr(probe_intensity, 'sum'):
                probe_intensity_sum = probe_intensity.sum()
                if hasattr(probe_intensity_sum, 'cpu'):
                    probe_intensity_sum = probe_intensity_sum.cpu().numpy()
            else:
                probe_intensity_sum = np.sum(probe_intensity)

            spectrum_intensities.append(probe_intensity_sum)

        return np.array(spectrum_intensities)

    def diffraction(self, probe_index: int = None, space: str = "reciprocal") -> np.ndarray:
        """
        Extract diffraction pattern for a specific probe position by summing over all frequencies.

        Args:
            probe_index: Index of probe position (default: 0). If None, averages over all probes.

        Returns:
            Diffraction pattern (kx, ky) - intensity summed over all frequencies
        """
        if probe_index is None:
            # Average over all probe positions
            all_diffractions = []
            for i in range(len(self.probe_positions)):
                probe_intensity = self._array[i]  # Shape: (frequency, kx, ky)
                diffraction_pattern = xp.sum(probe_intensity, axis=0)  # Sum over frequencies
                all_diffractions.append(diffraction_pattern)

            # Average all diffraction patterns
            if TORCH_AVAILABLE and hasattr(all_diffractions[0], 'cpu'):
                all_diffractions = [d.cpu().numpy() for d in all_diffractions]
            diffraction_pattern = np.mean(all_diffractions, axis=0)
        else:
            if probe_index >= len(self.probe_positions):
                raise ValueError(f"Probe index {probe_index} out of range")

            # Sum intensity data over all frequencies for this probe position
            probe_intensity = self._array[probe_index]  # Shape: (frequency, kx, ky)
            diffraction_pattern = xp.sum(probe_intensity, axis=0)  # Sum over frequencies

            # Convert to numpy if PyTorch tensor
            if TORCH_AVAILABLE and hasattr(diffraction_pattern, 'cpu'):
                diffraction_pattern = diffraction_pattern.cpu().numpy()

        if space == "real":
            diffraction_pattern = np.absolute(np.fft.ifft2(diffraction_pattern))

        return diffraction_pattern

    def spectral_diffraction(self, frequency: float, probe_index: int = None, space: str = "reciprocal") -> np.ndarray:
        """
        Extract spectral diffraction pattern at a specific frequency.

        Args:
            frequency: Frequency value in THz
            probe_index: Index of probe position (default: None). If None, averages over all probes.

        Returns:
            Spectral diffraction pattern (kx, ky) at the specified frequency
        """
        # Find closest frequency index
        freq_idx = np.argmin(np.abs(self.frequencies - frequency))

        if probe_index is None:
            # Average over all probe positions
            all_spectral_diffractions = []
            for i in range(len(self.probe_positions)):
                spectral_diffraction = self._array[i, freq_idx, :, :]
                all_spectral_diffractions.append(spectral_diffraction)

            # Average all spectral diffraction patterns
            if TORCH_AVAILABLE and hasattr(all_spectral_diffractions[0], 'cpu'):
                all_spectral_diffractions = [sd.cpu().numpy() for sd in all_spectral_diffractions]
            spectral_diffraction = np.mean(all_spectral_diffractions, axis=0)
        else:
            if probe_index >= len(self.probe_positions):
                raise ValueError(f"Probe index {probe_index} out of range")

            # Extract intensity data at this frequency and probe position
            spectral_diffraction = self._array[probe_index, freq_idx, :, :]

            # Convert to numpy if PyTorch tensor
            if TORCH_AVAILABLE and hasattr(spectral_diffraction, 'cpu'):
                spectral_diffraction = spectral_diffraction.cpu().numpy()

        if space == "real":
            spectral_diffraction = np.absolute(np.fft.ifft2(spectral_diffraction))

        return spectral_diffraction

    def masked_spectrum(self, mask: np.ndarray, probe_index: int = None) -> np.ndarray:
        """
        Extract spectrum with spatial masking in k-space.

        Args:
            mask: Spatial mask array with shape (kx, ky)
            probe_index: Index of probe position (default: None). If None, averages over all probes.

        Returns:
            Masked spectrum (frequency intensity) with k-space mask applied
        """
        kxs = to_cpu(self.kxs)
        kys = to_cpu(self.kys)

        if mask.shape != (len(kxs), len(kys)):
            raise ValueError(f"Mask shape {mask.shape} doesn't match k-space shape ({len(kxs)}, {len(kys)})")

        if probe_index is None:
            # Average over all probe positions
            all_masked_spectra = []
            for i in range(len(self.probe_positions)):
                probe_intensity = self._array[i]  # Shape: (frequency, kx, ky)
                masked_intensity = probe_intensity * mask[None, :, :]  # Broadcast mask to all frequencies
                masked_spectrum = xp.sum(masked_intensity, axis=(1, 2))  # Sum over masked k-space
                all_masked_spectra.append(masked_spectrum)

            # Average all masked spectra
            if TORCH_AVAILABLE and hasattr(all_masked_spectra[0], 'cpu'):
                all_masked_spectra = [ms.cpu().numpy() for ms in all_masked_spectra]
            masked_spectrum = np.mean(all_masked_spectra, axis=0)
        else:
            if probe_index >= len(self.probe_positions):
                raise ValueError(f"Probe index {probe_index} out of range")

            # Extract intensity data for this probe
            probe_intensity = self._array[probe_index]  # Shape: (frequency, kx, ky)

            # Apply spatial mask in k-space
            masked_intensity = probe_intensity * mask[None, :, :]  # Broadcast mask to all frequencies
            masked_spectrum = xp.sum(masked_intensity, axis=(1, 2))  # Sum over masked k-space

            # Convert to numpy if PyTorch tensor
            if TORCH_AVAILABLE and hasattr(masked_spectrum, 'cpu'):
                masked_spectrum = masked_spectrum.cpu().numpy()

        return masked_spectrum

    def dispersion(self, kx_path: np.ndarray, ky_path: np.ndarray, probe_index: int = None, space: str = "reciprocal") -> np.ndarray:
        """
        Extract dispersion relation from actual TACAW intensity data.

        Args:
            kx_path: kx values for dispersion calculation
            ky_path: ky values for dispersion calculation
            probe_index: Index of probe position (default: None). If None, averages over all probes.

        Returns:
            Dispersion relation array with shape (n_frequencies, n_k_points)
            Real intensity data from TACAW simulation
        """

        kx=self.kxs ; ky=self.kys
        if space == "real":
            kx=self.xs ; ky=self.ys

        # Convert to CPU/numpy for indexing operations
        kx = to_cpu(kx)
        ky = to_cpu(ky)

        # Find closest indices in our kxs/kys arrays for the requested paths
        kx_indices = []
        for kx_val in kx_path:
            idx = np.argmin(np.abs(kx - kx_val))
            kx_indices.append(idx)
        kx_indices = np.array(kx_indices)

        ky_indices = []
        for ky_val in ky_path:
            idx = np.argmin(np.abs(ky - ky_val))
            ky_indices.append(idx)
        ky_indices = np.array(ky_indices)

        # Create dispersion array
        n_frequencies = len(self.frequencies)
        n_k_points = len(kx_indices)
        dispersion = np.zeros((n_frequencies, n_k_points),dtype=complex)

        if probe_index is None:
            probe_index = np.arange(len(self.probe_positions))

        # loop frequencies first so we can cheaply iFFT kx,ky if we need to
        for w in range(n_frequencies):
            # all specified probe positions, this frequency, all kx,ky
            w_slice = self._array[probe_index, w, :, :]
            # optionally iFFT across kx,ky
            if space == "real":
                kwarg = {"dim":(1,2)} if TORCH_AVAILABLE else {"axes":(1,2)}
                w_slice = xp.fft.ifft2(w_slice,**kwarg)
            # bring to CPU
            if TORCH_AVAILABLE and hasattr(w_slice, 'cpu'):
                w_slice = w_slice.cpu().numpy()
            # sum across probe positions
            w_slice = np.mean(w_slice,axis=0)
            # select values at positions
            for i, (kx_idx, ky_idx) in enumerate(zip(kx_indices, ky_indices)):
                dispersion[w,i] = w_slice[ kx_idx, ky_idx ]

        return np.absolute(dispersion)

    # Since there are multiple things returnable by the above functions, i'm just offering up a generic heatmap plotter function here, where you pass Z,x,y
    def plot(self,intensities,xvals,yvals,xlabel="kx ($\\AA^{-1}$)",ylabel="ky ($\\AA^{-1}$)",filename=None,title=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        array = np.absolute(to_cpu(intensities)) # imshow convention: y,x. our convention: x,y
        aspect = None

        if isinstance(xvals,str):
            if xvals in ["kx","k"]:
                xlabel = "kx ($\\AA^{-1}$)" ; xvals = to_cpu(self.kxs)
            elif xvals == "ky":
                xlabel = "ky ($\\AA^{-1}$)" ; xvals = to_cpu(self.kys)
            elif xvals == "x":
                xlabel = "x ($\\AA$)" ; xvals = to_cpu(self.xs)
            elif xvals == "y":
                xlabel = "y ($\\AA$)" ; xvals = to_cpu(self.ys)

        if isinstance(yvals,str):
            if yvals == "omega":
                aspect = "auto"
            if yvals == "kx":
                ylabel = "kx ($\\AA^{-1}$)" ; yvals = to_cpu(self.kxs)
            elif yvals in ["ky","k"]:
                ylabel = "ky ($\\AA^{-1}$)" ; yvals = to_cpu(self.kys)
            elif yvals == "x":
                ylabel = "x ($\\AA$)" ; yvals = to_cpu(self.xs)
            elif yvals == "y":
                ylabel = "y ($\\AA$)" ; yvals = to_cpu(self.ys)
            elif yvals == "omega":
                ylabel = "frequency (THz)" ; yvals = to_cpu(self.frequencies)

        extent = ( np.amin(xvals) , np.amax(xvals) , np.amin(yvals) , np.amax(yvals) )
        ax.imshow(array, cmap="inferno",extent=extent,aspect=aspect)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()


class SEDData(TACAWData):
    """
    Data structure for SED (Spectral Energy Density) calculations.

    Functionally identical to TACAWData - both compute |Ψ(ω,q)|² from time-domain wavefunction data
    via FFT along the time axis.

    Attributes:
        probe_positions: List of (x,y) probe positions in Angstroms.
        frequencies: Frequencies in THz.
        kxs: kx sampling vectors (e.g., in Å⁻¹).
        kys: ky sampling vectors (e.g., in Å⁻¹).
        intensity: Intensity array |Ψ(ω,q)|² (probe_positions, frequency, kx, ky).
    """

    def __init__(self, wf_data: WFData, layer_index: int = None, keep_complex: bool = False) -> None:
        """
        Initialize SEDData from WFData by performing FFT.

        Args:
            wf_data: WFData object containing wavefunction data
            layer_index: Index of the layer to compute FFT for (default: last layer)
            keep_complex: If True, keep complex FFT result instead of intensity
        """
        super().__init__(wf_data, layer_index, keep_complex)
