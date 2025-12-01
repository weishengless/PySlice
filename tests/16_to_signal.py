"""
Test to_signal() methods for converting PySlice outputs to sea-eco Signal format.

Requires sea-eco to be installed:
    pip install -e /path/to/sea-eco
"""
import sys
import os
import numpy as np

try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice.io.loader import Loader
from pyslice.multislice.multislice import probe_grid
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.postprocessing.wf_data import WFData, SEA_ECO_AVAILABLE
from pyslice.postprocessing.tacaw_data import TACAWData
from pyslice.postprocessing.haadf_data import HAADFData

# Check if sea-eco is available
if not SEA_ECO_AVAILABLE:
    print("ERROR: sea-eco is not installed. Install with: pip install -e /path/to/sea-eco")
    sys.exit(1)

from pySEA.sea_eco.architecture.base_structure_numpy import Signal, Dimensions, Dimension

# Test parameters
dump = "inputs/hBN_truncated.lammpstrj"
dt = 0.005
types = {1: "B", 2: "N"}
a, b = 2.4907733333333337, 2.1570729817355123

print("=" * 60)
print("Testing to_signal() methods")
print("=" * 60)


def test_wfdata_to_signal():
    """Test WFData.to_signal() conversion."""
    print("\n--- Testing WFData.to_signal() ---")

    # Load trajectory and run simulation
    trajectory = Loader(dump, timestep=dt, atom_mapping=types).load()
    calculator = MultisliceCalculator()
    calculator.setup(trajectory, aperture=0, voltage_eV=100e3, sampling=0.1, slice_thickness=0.5)
    wf_data = calculator.run()

    # Test conversion with all probes
    signal = wf_data.to_signal()

    assert isinstance(signal, Signal), "Should return a Signal object"
    assert signal.data is not None, "Signal should have data"
    assert signal.dimensions is not None, "Signal should have dimensions"
    assert len(signal.dimensions) == 4, f"Expected 4 dimensions, got {len(signal.dimensions)}"

    # Check dimension names
    dim_names = signal.dimensions.get_names()
    assert 'probe' in dim_names, "Should have 'probe' dimension"
    assert 'time' in dim_names, "Should have 'time' dimension"
    assert 'kx' in dim_names, "Should have 'kx' dimension"
    assert 'ky' in dim_names, "Should have 'ky' dimension"

    # Check metadata
    assert signal.metadata is not None, "Signal should have metadata"
    assert hasattr(signal.metadata, 'Simulation'), "Should have Simulation metadata"

    print(f"  Data shape: {signal.data.shape}")
    print(f"  Dimensions: {dim_names}")
    print(f"  Signal type: {signal.signal_type}")
    print("  PASSED: WFData.to_signal() all probes")

    # Test conversion with single probe
    signal_single = wf_data.to_signal(probe_index=0)
    assert len(signal_single.dimensions) == 3, f"Single probe should have 3 dimensions, got {len(signal_single.dimensions)}"
    print("  PASSED: WFData.to_signal() single probe")

    return wf_data


def test_tacaw_to_signal(wf_data):
    """Test TACAWData.to_signal() conversion with all output types."""
    print("\n--- Testing TACAWData.to_signal() ---")

    tacaw = TACAWData(wf_data)

    # Test 'intensity' output (default)
    signal = tacaw.to_signal(output='intensity')
    assert isinstance(signal, Signal), "Should return a Signal object"
    assert signal.signal_type == '2D-EELS', f"Expected '2D-EELS', got {signal.signal_type}"
    dim_names = signal.dimensions.get_names()
    assert 'frequency' in dim_names, "Should have 'frequency' dimension"
    assert 'kx' in dim_names, "Should have 'kx' dimension"
    assert 'ky' in dim_names, "Should have 'ky' dimension"
    print(f"  intensity: shape={signal.data.shape}, dims={dim_names}")
    print("  PASSED: output='intensity'")

    # Test 'spectrum' output
    signal_spectrum = tacaw.to_signal(output='spectrum')
    assert signal_spectrum.signal_type == '1D-EELS', f"Expected '1D-EELS', got {signal_spectrum.signal_type}"
    assert len(signal_spectrum.dimensions) == 1, f"Spectrum should have 1 dimension, got {len(signal_spectrum.dimensions)}"
    assert signal_spectrum.dimensions[0].name == 'frequency', "Should have 'frequency' dimension"
    print(f"  spectrum: shape={signal_spectrum.data.shape}")
    print("  PASSED: output='spectrum'")

    # Test 'diffraction' output
    signal_diff = tacaw.to_signal(output='diffraction')
    assert signal_diff.signal_type == 'Diffraction', f"Expected 'Diffraction', got {signal_diff.signal_type}"
    assert len(signal_diff.dimensions) == 2, f"Diffraction should have 2 dimensions, got {len(signal_diff.dimensions)}"
    dim_names_diff = signal_diff.dimensions.get_names()
    assert 'kx' in dim_names_diff and 'ky' in dim_names_diff, "Should have 'kx' and 'ky' dimensions"
    print(f"  diffraction: shape={signal_diff.data.shape}")
    print("  PASSED: output='diffraction'")

    # Test 'dispersion' output
    kxs = np.asarray(tacaw.kxs.cpu() if hasattr(tacaw.kxs, 'cpu') else tacaw.kxs)
    kx_path = kxs[kxs >= 0]
    kx_path = kx_path[kx_path <= 4/a]
    ky_path = np.zeros(len(kx_path)) + 2/b

    signal_disp = tacaw.to_signal(output='dispersion', kx_path=kx_path, ky_path=ky_path)
    assert signal_disp.signal_type == '2D-EELS', f"Expected '2D-EELS', got {signal_disp.signal_type}"
    assert len(signal_disp.dimensions) == 2, f"Dispersion should have 2 dimensions, got {len(signal_disp.dimensions)}"
    dim_names_disp = signal_disp.dimensions.get_names()
    assert 'frequency' in dim_names_disp and 'k' in dim_names_disp, "Should have 'frequency' and 'k' dimensions"
    print(f"  dispersion: shape={signal_disp.data.shape}")
    print("  PASSED: output='dispersion'")

    # Test with single probe
    signal_single = tacaw.to_signal(output='intensity', probe_index=0)
    assert len(signal_single.dimensions) == 3, f"Single probe should have 3 dimensions, got {len(signal_single.dimensions)}"
    print("  PASSED: single probe")

    # Test invalid output type
    try:
        tacaw.to_signal(output='invalid')
        assert False, "Should raise ValueError for invalid output type"
    except ValueError as e:
        print(f"  PASSED: raises ValueError for invalid output")

    # Test dispersion without k-path
    try:
        tacaw.to_signal(output='dispersion')
        assert False, "Should raise ValueError when kx_path/ky_path not provided"
    except ValueError as e:
        print(f"  PASSED: raises ValueError when k-path missing")

    return tacaw


def test_haadf_to_signal():
    """Test HAADFData.to_signal() conversion."""
    print("\n--- Testing HAADFData.to_signal() ---")

    # Load trajectory and run simulation with probe grid
    trajectory = Loader(dump, timestep=dt, atom_mapping=types).load()
    # Trim trajectory to reduce memory usage
    trajectory = trajectory.slice_timesteps(ith=10)  # Every 10th frame

    # Create a small probe grid for HAADF
    xy = probe_grid([0, 2*a], [0, 2*b], 2, 2)

    calculator = MultisliceCalculator()
    calculator.setup(trajectory, aperture=30, voltage_eV=100e3, sampling=0.2,
                     slice_thickness=1.0, probe_positions=xy)
    wf_data = calculator.run()

    haadf = HAADFData(wf_data)

    # Test conversion (this will also calculate ADF if not already done)
    signal = haadf.to_signal(inner_mrad=45, outer_mrad=150)

    assert isinstance(signal, Signal), "Should return a Signal object"
    assert signal.signal_type == 'Image', f"Expected 'Image', got {signal.signal_type}"
    assert signal.data is not None, "Signal should have data"
    assert len(signal.dimensions) == 2, f"Expected 2 dimensions, got {len(signal.dimensions)}"

    dim_names = signal.dimensions.get_names()
    assert 'x' in dim_names and 'y' in dim_names, "Should have 'x' and 'y' dimensions"

    # Check that dimensions have correct space
    for dim in signal.dimensions.dimensions:
        assert dim.space == 'position', f"Dimension {dim.name} should have space='position'"

    # Check metadata
    assert signal.metadata is not None, "Signal should have metadata"
    assert hasattr(signal.metadata, 'Simulation'), "Should have Simulation metadata"
    assert signal.metadata.Simulation.inner_mrad == 45, "Should store inner_mrad in metadata"
    assert signal.metadata.Simulation.outer_mrad == 150, "Should store outer_mrad in metadata"

    print(f"  Data shape: {signal.data.shape}")
    print(f"  Dimensions: {dim_names}")
    print(f"  Signal type: {signal.signal_type}")
    print("  PASSED: HAADFData.to_signal()")

    return signal


def test_signal_serialization(signal):
    """Test that converted signals can be serialized to .sea files."""
    print("\n--- Testing Signal serialization ---")

    output_path = "outputs/test_signal.sea"
    os.makedirs("outputs", exist_ok=True)

    # Save to .sea file
    signal.to_sea(output_path)
    assert os.path.exists(output_path), "Should create .sea file"
    print(f"  Saved to: {output_path}")

    # Load back
    loaded_signal = Signal()
    loaded_signal.from_sea(output_path)

    assert loaded_signal.data is not None, "Loaded signal should have data"
    assert np.allclose(loaded_signal.data, signal.data), "Loaded data should match original"
    assert loaded_signal.signal_type == signal.signal_type, "Signal type should match"
    print("  PASSED: Signal serialization round-trip")


if __name__ == '__main__':
    # Run tests
    wf_data = test_wfdata_to_signal()
    tacaw = test_tacaw_to_signal(wf_data)
    haadf_signal = test_haadf_to_signal()
    test_signal_serialization(haadf_signal)

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
