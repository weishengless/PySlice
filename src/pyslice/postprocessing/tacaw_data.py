"""
Core data structure for TACAW EELS calculations.
"""
from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import logging, os
from .wf_data import WFData, SEA_ECO_AVAILABLE
from pyslice.backend import to_cpu

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
class TACAWData(WFData):
    # inherit all attributes from parent object
    def __init__(self, WFData, layer_index: int = None, keep_complex: bool = False) -> object:
        self.__class__ = type(WFData.__class__.__name__,
                              (self.__class__, WFData.__class__),
                              {})
        self.__dict__ = WFData.__dict__
        self.keep_complex = keep_complex
        self.fft_from_wf_data(layer_index)
 

    """
    Data structure for storing TACAW EELS results with format: probe_positions, frequency, kx, ky.

    Attributes:
        probe_positions: List of (x,y) probe positions in Angstroms.
        frequency: Frequencies in THz.
        kx: kx sampling vectors (e.g., in Å⁻¹).
        ky: ky sampling vectors (e.g., in Å⁻¹).
        intensity: Intensity array |Ψ(ω,q)|² (probe_positions, frequency, kx, ky).
    """
    probe_positions: List[Tuple[float, float]]
    frequency: np.ndarray  # frequencies in THz
    kx: np.ndarray  # kx sampling vectors
    ky: np.ndarray  # ky sampling vectors
    intensity: np.ndarray  # Intensity array |Ψ(ω,q)|² (probe_positions, frequency, kx, ky)
    keep_complex: bool

    def fft_from_wf_data(self, layer_index: int = None):
        """
        Perform FFT along the time axis for a specific layer to convert to TACAW data.
        This implements the JACR method: Ψ(t,q,r) → |Ψ(ω,q,r)|² via FFT.

        Args:
            layer_index: Index of the layer to compute FFT for (default: last layer)

        Returns:
            TACAWData object with intensity data |Ψ(ω,q)|² for the specified layer
        """

        if os.path.exists(self.cache_dir / "tacaw.npy"):
            self.frequencies = np.load(self.cache_dir / "tacaw_freq.npy")
            self.intensity = np.load(self.cache_dir / "tacaw.npy")
            if TORCH_AVAILABLE:
                self.frequencies = xp.Tensor(self.frequencies)
                self.intensity = xp.Tensor(self.intensity)
            return

        # Default to last layer if not specified
        if layer_index is None:
            layer_index = len(self.layer) - 1

        # Validate layer index
        if layer_index < 0 or layer_index >= len(self.layer):
            raise ValueError(f"layer_index {layer_index} out of range [0, {len(self.layer)-1}]")

        # Compute frequencies from time sampling
        n_freq = len(self.time)
        dt = self.time[1] - self.time[0] 
        self.frequencies = np.fft.fftfreq(n_freq, d=dt)
        self.frequencies = np.fft.fftshift(self.frequencies)

        # Extract wavefunction data for the specified layer
        # Shape: (probe_positions, time, kx, ky, layer)
        wf_layer = self.array[:, :, :, :, layer_index]
        
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
                self.intensity = wf_fft
            else:
                self.intensity = torch.abs(wf_fft)**2
        else:
            if self.keep_complex:
                self.intensity = wf_fft
            else:
                self.intensity = np.abs(wf_fft)**2

        np.save(self.cache_dir / "tacaw_freq.npy", self.frequencies)
        np.save(self.cache_dir / "tacaw.npy", self.intensity.detach().cpu().numpy())


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
                probe_intensity = self.intensity[i]  # Shape: (frequency, kx, ky)
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
            probe_intensity = self.intensity[probe_index]  # Shape: (frequency, kx, ky)
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
            probe_intensity = self.intensity[probe_idx, freq_idx, :, :]
            
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
                probe_intensity = self.intensity[i]  # Shape: (frequency, kx, ky)
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
            probe_intensity = self.intensity[probe_index]  # Shape: (frequency, kx, ky)
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
                spectral_diffraction = self.intensity[i, freq_idx, :, :]
                all_spectral_diffractions.append(spectral_diffraction)
            
            # Average all spectral diffraction patterns
            if TORCH_AVAILABLE and hasattr(all_spectral_diffractions[0], 'cpu'):
                all_spectral_diffractions = [sd.cpu().numpy() for sd in all_spectral_diffractions]
            spectral_diffraction = np.mean(all_spectral_diffractions, axis=0)
        else:
            if probe_index >= len(self.probe_positions):
                raise ValueError(f"Probe index {probe_index} out of range")

            # Extract intensity data at this frequency and probe position
            spectral_diffraction = self.intensity[probe_index, freq_idx, :, :]
            
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
        if mask.shape != (len(self.kx), len(self.ky)):
            raise ValueError(f"Mask shape {mask.shape} doesn't match k-space shape ({len(self.kx)}, {len(self.ky)})")

        if probe_index is None:
            # Average over all probe positions
            all_masked_spectra = []
            for i in range(len(self.probe_positions)):
                probe_intensity = self.intensity[i]  # Shape: (frequency, kx, ky)
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
            probe_intensity = self.intensity[probe_index]  # Shape: (frequency, kx, ky)
            
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
            w_slice = self.intensity[probe_index, w, :, :]
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

    def to_signal(self, output: str = 'intensity', probe_index: Optional[int] = None,
                  kx_path: Optional[np.ndarray] = None, ky_path: Optional[np.ndarray] = None) -> 'Signal':
        """
        Convert TACAWData to a sea-eco Signal object.

        Args:
            output: Type of data to export. Options:
                - 'intensity': Full 4D data (probe, frequency, kx, ky)
                - 'spectrum': 1D spectrum summed over k-space (frequency,)
                - 'diffraction': 2D diffraction summed over frequencies (kx, ky)
                - 'dispersion': 2D dispersion along k-path (frequency, k). Requires kx_path and ky_path.
            probe_index: Which probe position to export. If None, averages/exports all probes.
            kx_path: kx values for dispersion path (required if output='dispersion')
            ky_path: ky values for dispersion path (required if output='dispersion')

        Returns:
            Signal object with appropriate dimensions for the output type.

        Raises:
            ImportError: If sea-eco is not installed
            ValueError: If output type is invalid or dispersion requested without k-path
        """
        if not SEA_ECO_AVAILABLE:
            raise ImportError(
                "sea-eco package is required for to_signal(). "
                "Install it with: pip install -e /path/to/sea-eco"
            )

        # Convert coordinate arrays to numpy
        frequencies = to_cpu(self.frequencies)
        kxs = to_cpu(self.kxs)
        kys = to_cpu(self.kys)

        if output == 'intensity':
            intensity = self.intensity
            if TORCH_AVAILABLE and hasattr(intensity, 'cpu'):
                intensity = intensity.cpu().numpy()

            if probe_index is not None:
                data = np.asarray(intensity[probe_index])
                dimensions = Dimensions([
                    Dimension(name='frequency', space='spectral', units='THz', values=np.asarray(frequencies)),
                    Dimension(name='kx', space='scattering', units='Å⁻¹', values=np.asarray(kxs)),
                    Dimension(name='ky', space='scattering', units='Å⁻¹', values=np.asarray(kys)),
                ], nav_dimensions=[0], sig_dimensions=[1, 2])
            else:
                data = np.asarray(intensity)
                dimensions = Dimensions([
                    Dimension(name='probe', space='position', units='Å', values=np.arange(len(self.probe_positions))),
                    Dimension(name='frequency', space='spectral', units='THz', values=np.asarray(frequencies)),
                    Dimension(name='kx', space='scattering', units='Å⁻¹', values=np.asarray(kxs)),
                    Dimension(name='ky', space='scattering', units='Å⁻¹', values=np.asarray(kys)),
                ], nav_dimensions=[0, 1], sig_dimensions=[2, 3])
            signal_type = '2D-EELS'
            name = 'TACAW_intensity'

        elif output == 'spectrum':
            data = self.spectrum(probe_index=probe_index)
            dimensions = Dimensions([
                Dimension(name='frequency', space='spectral', units='THz', values=np.asarray(frequencies)),
            ], nav_dimensions=[], sig_dimensions=[0])
            signal_type = '1D-EELS'
            name = 'TACAW_spectrum'

        elif output == 'diffraction':
            data = self.diffraction(probe_index=probe_index)
            dimensions = Dimensions([
                Dimension(name='kx', space='scattering', units='Å⁻¹', values=np.asarray(kxs)),
                Dimension(name='ky', space='scattering', units='Å⁻¹', values=np.asarray(kys)),
            ], nav_dimensions=[], sig_dimensions=[0, 1])
            signal_type = 'Diffraction'
            name = 'TACAW_diffraction'

        elif output == 'dispersion':
            if kx_path is None or ky_path is None:
                raise ValueError("kx_path and ky_path are required for dispersion output")
            data = self.dispersion(kx_path, ky_path, probe_index=probe_index)
            k_path = np.sqrt(kx_path**2 + ky_path**2)
            dimensions = Dimensions([
                Dimension(name='frequency', space='spectral', units='THz', values=np.asarray(frequencies)),
                Dimension(name='k', space='scattering', units='Å⁻¹', values=np.asarray(k_path)),
            ], nav_dimensions=[], sig_dimensions=[0, 1])
            signal_type = '2D-EELS'
            name = 'TACAW_dispersion'

        else:
            raise ValueError(f"Unknown output type: {output}. Use 'intensity', 'spectrum', 'diffraction', or 'dispersion'")

        # Build metadata
        metadata_dict = {
            'General': {'title': f'TACAW {output.capitalize()}', 'signal_type': f'TACAW {output}'},
            'Instrument': {
                'beam_energy': float(self.probe.eV),
                'Detectors': {
                    'Simulated': {
                        'name': 'TACAW Simulation',
                        'voltage_eV': float(self.probe.eV),
                        'wavelength_A': float(self.probe.wavelength),
                        'aperture_mrad': float(self.probe.mrad),
                    }
                },
                'Scan': {'scan_uuid': None}
            },
            'Simulation': {
                'voltage_eV': float(self.probe.eV),
                'wavelength_A': float(self.probe.wavelength),
                'aperture_mrad': float(self.probe.mrad),
                'probe_positions': [list(p) for p in self.probe_positions],
                'probe_index': probe_index,
            }
        }
        metadata = GeneralMetadata(metadata_dict)

        return Signal(data=data, name=name, dimensions=dimensions, signal_type=signal_type, metadata=metadata)

@dataclass
class SEDData:
    # inherit all attributes from parent object
    def __init__(self, WFData, layer_index: int = None, keep_complex: bool = False) -> object:
        self.__class__ = type(WFData.__class__.__name__,
                              (self.__class__, WFData.__class__),
                              {})
        self.__dict__ = WFData.__dict__
        self.keep_complex = keep_complex
        self.fft_from_wf_data(layer_index)
 

    """
    Data structure for storing TACAW EELS results with format: probe_positions, frequency, kx, ky.

    Attributes:
        probe_positions: List of (x,y) probe positions in Angstroms.
        frequency: Frequencies in THz.
        kx: kx sampling vectors (e.g., in Å⁻¹).
        ky: ky sampling vectors (e.g., in Å⁻¹).
        intensity: Intensity array |Ψ(ω,q)|² (probe_positions, frequency, kx, ky).
    """
    probe_positions: List[Tuple[float, float]]
    frequency: np.ndarray  # frequencies in THz
    kx: np.ndarray  # kx sampling vectors
    ky: np.ndarray  # ky sampling vectors
    intensity: np.ndarray  # Intensity array |Ψ(ω,q)|² (probe_positions, frequency, kx, ky)
    keep_complex: bool


# Example usage (for testing within this file)
if __name__ == '__main__':
    # Create dummy data matching the new format
    probe_positions = [(0.0, 0.0), (1.5, 0.0), (0.0, 1.5), (1.5, 1.5)]
    frequencies = np.linspace(0, 50, 100)  # THz
    kx = np.linspace(-1, 1, 32)
    ky = np.linspace(-1, 1, 32)

    # Create dummy intensity data for TACAW (frequency domain)
    intensity = np.random.rand(len(probe_positions), len(frequencies), len(kx), len(ky))

    tacaw_obj = TACAWData(
        probe_positions=probe_positions,
        frequency=frequencies,
        kx=kx,
        ky=ky,
        intensity=intensity
    )

    print("TACAWData object created with simplified format.")
    print(f"Probe positions: {tacaw_obj.probe_positions}")
    print(f"Frequency range (THz): {tacaw_obj.frequency.min():.2f} - {tacaw_obj.frequency.max():.2f}")
    print(f"kx range: {tacaw_obj.kx.min():.2f} - {tacaw_obj.kx.max():.2f}")
    print(f"ky range: {tacaw_obj.ky.min():.2f} - {tacaw_obj.ky.max():.2f}")

    # Test the postprocessing methods
    print("\n--- Postprocessing Method Examples ---")

    # Test spectrum
    spectrum_data = tacaw_obj.spectrum(probe_index=0)
    print(f"Spectrum for probe 0: {spectrum_data.shape} array")

    # Test spectrum image (real space intensity at specific frequency)
    spec_img = tacaw_obj.spectrum_image(frequency=10.0, probe_indices=[0, 1])
    print(f"Spectrum image at 10 THz for 2 probes: {spec_img.shape} array (real space intensities)")

    # Test diffraction
    diff_pattern = tacaw_obj.diffraction(probe_index=0)
    print(f"Diffraction pattern for probe 0: {diff_pattern.shape} array")

    # Test spectral diffraction
    spec_diff = tacaw_obj.spectral_diffraction(frequency=15.0, probe_index=0)
    print(f"Spectral diffraction at 15 THz: {spec_diff.shape} array")

    # Test masked spectrum
    mask = np.ones((len(tacaw_obj.kx), len(tacaw_obj.ky)))
    mask[:len(tacaw_obj.kx)//2, :] = 0  # Mask first half of k-space
    masked_spec = tacaw_obj.masked_spectrum(mask, probe_index=0)
    print(f"Masked spectrum: {masked_spec.shape} array")

    # Test dispersion
    disp = tacaw_obj.dispersion()
    print(f"Dispersion relation: {disp.shape} array")

    print("\nAll postprocessing methods working!") 