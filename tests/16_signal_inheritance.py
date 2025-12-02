"""
Test Signal inheritance for PySlice data classes.

Tests that WFData, TACAWData, and HAADFData correctly inherit from Signal
and maintain backward compatibility.
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
from pyslice.postprocessing.wf_data import WFData
from pyslice.postprocessing.tacaw_data import TACAWData
from pyslice.postprocessing.haadf_data import HAADFData
from pyslice.data import Signal, Dimensions, Dimension

# Test parameters
dump = "inputs/hBN_truncated.lammpstrj"
dt = 0.005
types = {1: "B", 2: "N"}
a, b = 2.4907733333333337, 2.1570729817355123

print("=" * 60)
print("Testing Signal Inheritance")
print("=" * 60)


def test_wfdata_is_signal():
    """Test that WFData inherits from Signal."""
    print("\n--- Testing WFData is Signal ---")

    # Load trajectory and run simulation
    trajectory = Loader(dump, timestep=dt, atom_mapping=types).load()
    calculator = MultisliceCalculator()
    calculator.setup(trajectory, aperture=0, voltage_eV=100e3, sampling=0.1, slice_thickness=0.5)
    wf_data = calculator.run()

    # Test inheritance
    assert isinstance(wf_data, Signal), "WFData should be an instance of Signal"
    assert isinstance(wf_data, WFData), "WFData should still be an instance of WFData"
    print("  PASSED: WFData is instance of Signal")

    # Test Signal attributes exist
    assert hasattr(wf_data, 'dimensions'), "WFData should have dimensions attribute"
    assert hasattr(wf_data, 'metadata'), "WFData should have metadata attribute"
    assert hasattr(wf_data, 'signal_type'), "WFData should have signal_type attribute"
    assert wf_data.signal_type == 'Diffraction', f"Expected signal_type='Diffraction', got {wf_data.signal_type}"
    print("  PASSED: WFData has Signal attributes")

    # Test dimensions
    assert wf_data.dimensions is not None, "WFData should have dimensions"
    assert len(wf_data.dimensions) == 5, f"Expected 5 dimensions, got {len(wf_data.dimensions)}"
    dim_names = wf_data.dimensions.get_names()
    assert 'probe' in dim_names, "Should have 'probe' dimension"
    assert 'time' in dim_names, "Should have 'time' dimension"
    assert 'kx' in dim_names, "Should have 'kx' dimension"
    assert 'ky' in dim_names, "Should have 'ky' dimension"
    assert 'layer' in dim_names, "Should have 'layer' dimension"
    print(f"  Dimensions: {dim_names}")
    print("  PASSED: WFData dimensions correct")

    # Test backward compatibility - original attributes still work
    assert hasattr(wf_data, 'probe_positions'), "Should have probe_positions"
    assert hasattr(wf_data, 'time'), "Should have time"
    assert hasattr(wf_data, 'kxs'), "Should have kxs"
    assert hasattr(wf_data, 'kys'), "Should have kys"
    assert hasattr(wf_data, 'array'), "Should have array property"
    print("  PASSED: WFData backward compatible attributes")

    # Test data property (lazy numpy conversion)
    data = wf_data.data
    assert isinstance(data, np.ndarray), "data property should return numpy array"
    print(f"  Data shape: {data.shape}")
    print("  PASSED: WFData data property works")

    return wf_data


def test_tacawdata_is_signal(wf_data):
    """Test that TACAWData inherits from Signal."""
    print("\n--- Testing TACAWData is Signal ---")

    tacaw = TACAWData(wf_data)

    # Test inheritance
    assert isinstance(tacaw, Signal), "TACAWData should be an instance of Signal"
    assert isinstance(tacaw, TACAWData), "TACAWData should still be an instance of TACAWData"
    print("  PASSED: TACAWData is instance of Signal")

    # Test Signal attributes
    assert tacaw.signal_type == '2D-EELS', f"Expected signal_type='2D-EELS', got {tacaw.signal_type}"
    assert tacaw.dimensions is not None, "TACAWData should have dimensions"
    dim_names = tacaw.dimensions.get_names()
    assert 'frequency' in dim_names, "Should have 'frequency' dimension"
    assert 'kx' in dim_names, "Should have 'kx' dimension"
    assert 'ky' in dim_names, "Should have 'ky' dimension"
    print(f"  Dimensions: {dim_names}")
    print("  PASSED: TACAWData dimensions correct")

    # Test backward compatibility
    assert hasattr(tacaw, 'frequencies'), "Should have frequencies"
    assert hasattr(tacaw, 'intensity'), "Should have intensity property"
    assert hasattr(tacaw, 'kxs'), "Should have kxs"
    assert hasattr(tacaw, 'kys'), "Should have kys"
    print("  PASSED: TACAWData backward compatible attributes")

    # Test data property
    data = tacaw.data
    assert isinstance(data, np.ndarray), "data property should return numpy array"
    print(f"  Data shape: {data.shape}")
    print("  PASSED: TACAWData data property works")

    return tacaw


def test_haadf_is_signal():
    """Test that HAADFData inherits from Signal."""
    print("\n--- Testing HAADFData is Signal ---")

    # Load trajectory and run simulation with probe grid
    trajectory = Loader(dump, timestep=dt, atom_mapping=types).load()
    trajectory = trajectory.slice_timesteps(ith=10)  # Every 10th frame

    # Create a small probe grid for HAADF
    xy = probe_grid([0, 2*a], [0, 2*b], 2, 2)

    calculator = MultisliceCalculator()
    calculator.setup(trajectory, aperture=30, voltage_eV=100e3, sampling=0.2,
                     slice_thickness=1.0, probe_positions=xy)
    wf_data = calculator.run()

    haadf = HAADFData(wf_data)

    # Test inheritance
    assert isinstance(haadf, Signal), "HAADFData should be an instance of Signal"
    assert isinstance(haadf, HAADFData), "HAADFData should still be an instance of HAADFData"
    print("  PASSED: HAADFData is instance of Signal")

    # Test Signal attributes
    assert haadf.signal_type == 'Image', f"Expected signal_type='Image', got {haadf.signal_type}"
    assert haadf.dimensions is not None, "HAADFData should have dimensions"
    print("  PASSED: HAADFData Signal attributes")

    # Calculate ADF
    haadf.calculateADF(inner_mrad=45, outer_mrad=150)
    assert haadf.adf is not None, "ADF should be calculated"
    assert haadf.data is not None, "data property should work after calculateADF"
    print(f"  ADF shape: {haadf.data.shape}")
    print("  PASSED: HAADFData ADF calculation works")

    # Check dimensions were updated
    dim_names = haadf.dimensions.get_names()
    assert 'x' in dim_names and 'y' in dim_names, "Should have 'x' and 'y' dimensions"
    print(f"  Dimensions: {dim_names}")
    print("  PASSED: HAADFData dimensions correct")

    # Test metadata
    assert haadf.metadata is not None, "Should have metadata"
    assert hasattr(haadf.metadata, 'Simulation'), "Should have Simulation metadata"
    print("  PASSED: HAADFData metadata")

    return haadf


def test_signal_serialization(haadf):
    """Test that Signal objects can be serialized to .sea files."""
    print("\n--- Testing Signal serialization ---")

    output_path = "outputs/test_signal.sea"
    os.makedirs("outputs", exist_ok=True)

    # Create a plain Signal from HAADF data for serialization
    # (HAADFData has torch tensors that need special handling)
    signal = Signal(
        data=haadf.data,  # This is numpy via the property
        name='TestSignal',
        dimensions=Dimensions([
            Dimension(name='x', space='position', units='Å',
                     values=np.asarray(haadf.xs.cpu() if hasattr(haadf.xs, 'cpu') else haadf.xs)),
            Dimension(name='y', space='position', units='Å',
                     values=np.asarray(haadf.ys.cpu() if hasattr(haadf.ys, 'cpu') else haadf.ys)),
        ], nav_dimensions=[0, 1], sig_dimensions=[]),
        signal_type='Image'
    )

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
    wf_data = test_wfdata_is_signal()
    tacaw = test_tacawdata_is_signal(wf_data)
    haadf = test_haadf_is_signal()
    test_signal_serialization(haadf)

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
