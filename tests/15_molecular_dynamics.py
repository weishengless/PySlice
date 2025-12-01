"""
Molecular dynamics test using ORB force fields.

This test requires the orb-models package to be installed:
    pip install "pyslice[md]"

Set LOAD_FROM_CACHE = True to skip MD and load from previous run.
"""

import logging
import os
import sys

# Configure logging BEFORE importing other modules
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

# Add parent directory to path for running without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import shutil

import numpy as np
from ase.build import bulk

from pyslice.io.loader import Loader
from pyslice.md import MDCalculator
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.postprocessing.tacaw_data import TACAWData

# OPTION: Load trajectory from cache instead of running MD
# Set to True to skip MD and load from previous run
LOAD_FROM_CACHE = False
OUTPUT_DIR = "tests/outputs/md_output"

if LOAD_FROM_CACHE and os.path.exists(f"{OUTPUT_DIR}/production.traj"):
    print("=" * 70)
    print("LOADING TRAJECTORY FROM CACHE")
    print("=" * 70)
    loader = Loader(filename=f"{OUTPUT_DIR}/production.traj", timestep=0.001)
    trajectory = loader.load()
    print(
        f"Loaded trajectory: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms"
    )
else:
    # Clean up previous MD outputs
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    # CREATE TEST SYSTEM
    print("=" * 70)
    print("MOLECULAR DYNAMICS TEST")
    print("=" * 70)
    print("\nCreating test system (Si)...")

    # Create orthogonal diamond structure
    atoms = bulk("Si", "diamond", a=5.43, orthorhombic=True) * (30, 30, 3)
    print(f"System: {len(atoms)} atoms")
    print(f"Cell:\n{atoms.get_cell()}")

    # SETUP MD CALCULATOR
    # Auto-detect best device (CUDA > MPS > CPU)
    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\nInitializing MDCalculator on {device}...")
    md_calc = MDCalculator(model_name="orb-v3-direct-20-omat", device=device)

    # SETUP SIMULATION
    print("Setting up MD simulation...")
    md_calc.setup(
        atoms=atoms,
        temperature=300,
        timestep=5.0,
        ensemble="nvt",
        friction=0.15,
        temp_tolerance=5.0,
        temp_threshold=5.0,
        energy_threshold=0.05,
        min_equilibration_steps=100,
        max_equilibration_steps=10000,
        production_steps=2000,
        check_interval=10,
        save_interval=2,
        output_dir=OUTPUT_DIR,
    )

    # RUN MD SIMULATION
    print("\nRunning MD simulation... (this may take a while)")
    trajectory = md_calc.run()

# VALIDATE TRAJECTORY
print("\n" + "=" * 70)
print("VALIDATING TRAJECTORY")
print("=" * 70)

assert trajectory.n_frames > 0
print(f"Trajectory has {trajectory.n_frames} frames")

assert trajectory.n_atoms > 0
print(f"Trajectory has {trajectory.n_atoms} atoms")

assert trajectory.positions.shape == (trajectory.n_frames, trajectory.n_atoms, 3)
print(f"Positions shape: {trajectory.positions.shape}")

assert trajectory.velocities.shape == (trajectory.n_frames, trajectory.n_atoms, 3)
print(f"Velocities shape: {trajectory.velocities.shape}")

assert trajectory.box_matrix.shape == (3, 3)
print(f"Box matrix shape: {trajectory.box_matrix.shape}")

assert trajectory.timestep > 0
print(f"Timestep: {trajectory.timestep} ps")

# CHECK TEMPERATURE FROM ASE TRAJECTORY
from ase.io import read as aseread

traj_file = f"{OUTPUT_DIR}/production.traj"
ase_traj = aseread(traj_file, index=":")
temps = [atoms_snap.get_temperature() for atoms_snap in ase_traj]
mean_temp = np.mean(temps)
print(f"Mean temperature: {mean_temp:.1f} K (target: 300 K)")

# CHECK FILES EXIST
assert os.path.exists(f"{OUTPUT_DIR}/production.traj")
print(f"Production trajectory exists")

assert os.path.exists(f"{OUTPUT_DIR}/production.log")
print(f"Production log exists")

assert os.path.exists(f"{OUTPUT_DIR}/equilibration.traj")
print(f"Equilibration trajectory exists")

assert os.path.exists(f"{OUTPUT_DIR}/equilibration.log")
print(f"Equilibration log exists")

# TEST TRAJECTORY METHODS
print("\n" + "=" * 70)
print("TESTING TRAJECTORY METHODS")
print("=" * 70)

mean_pos = trajectory.get_mean_positions()
assert mean_pos.shape == (trajectory.n_atoms, 3)
print(f"get_mean_positions: {mean_pos.shape}")

# TEST CONVERSION VIA LOADER
print("\n" + "=" * 70)
print("TESTING ASE TO PYSLICE CONVERSION")
print("=" * 70)

loader = Loader(filename=traj_file, timestep=0.001)
loaded_trajectory = loader.load()
assert loaded_trajectory.n_frames == trajectory.n_frames
assert loaded_trajectory.n_atoms == trajectory.n_atoms
print(
    f"Loader conversion: {loaded_trajectory.n_frames} frames, {loaded_trajectory.n_atoms} atoms"
)

# PLOT TEMPERATURE EVOLUTION
print("\n" + "=" * 70)
print("PLOTTING TEMPERATURE EVOLUTION")
print("=" * 70)

import matplotlib.pyplot as plt

# Read equilibration log
eq_data = np.loadtxt(f"{OUTPUT_DIR}/equilibration.log", comments="#")
eq_steps = eq_data[:, 0]
eq_time = eq_data[:, 1]
eq_temps = eq_data[:, 2]
eq_epot = eq_data[:, 3]
eq_etot = eq_data[:, 5]

# Read production log
prod_data = np.loadtxt(f"{OUTPUT_DIR}/production.log", comments="#")
prod_steps = prod_data[:, 0]
prod_time = prod_data[:, 1]
prod_temps = prod_data[:, 2]
prod_epot = prod_data[:, 3]
prod_etot = prod_data[:, 5]

# Offset production time to continue from end of equilibration
# Normalize production time to start at 0, then add equilibration end time
if len(eq_time) > 0 and len(prod_time) > 0:
    prod_time_offset = prod_time - prod_time[0] + eq_time[-1]
else:
    prod_time_offset = prod_time

# Create 2-panel plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# TEMPERATURE PLOT
ax1.plot(eq_time, eq_temps, "b-", alpha=0.7, label="Equilibration")
ax1.plot(prod_time_offset, prod_temps, "r-", alpha=0.7, label="Production")
ax1.axhline(y=300, color="k", linestyle="--", label="Target (300 K)")
if len(eq_time) > 0 and len(prod_time) > 0:
    ax1.axvline(x=eq_time[-1], color="gray", linestyle=":", alpha=0.5, label="Eq->Prod")
ax1.set_xlabel("Time (ps)")
ax1.set_ylabel("Temperature (K)")
ax1.set_title("MD Temperature Evolution")
ax1.legend()
ax1.grid(alpha=0.3)

# ENERGY PLOT
ax2.plot(eq_time, eq_etot, "b-", alpha=0.7, label="Equilibration (Total)")
ax2.plot(prod_time_offset, prod_etot, "r-", alpha=0.7, label="Production (Total)")
ax2.plot(
    eq_time, eq_epot, "b--", alpha=0.5, linewidth=1, label="Equilibration (Potential)"
)
ax2.plot(
    prod_time_offset,
    prod_epot,
    "r--",
    alpha=0.5,
    linewidth=1,
    label="Production (Potential)",
)
if len(eq_time) > 0 and len(prod_time) > 0:
    ax2.axvline(x=eq_time[-1], color="gray", linestyle=":", alpha=0.5, label="Eq->Prod")
ax2.set_xlabel("Time (ps)")
ax2.set_ylabel("Energy (eV)")
ax2.set_title("MD Energy Evolution")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/md_evolution.png", dpi=150)
print(f"Saved MD evolution plot: {OUTPUT_DIR}/md_evolution.png")

# TACAW CALCULATION ON MD TRAJECTORY
print("\n" + "=" * 70)
print("TACAW CALCULATION")
print("=" * 70)

print(f"Using {trajectory.n_frames} frames for TACAW calculation")

# Setup multislice calculator for parallel beam (aperture=0)
calculator = MultisliceCalculator()
calculator.setup(
    trajectory,
    aperture=0,  # Parallel beam for TACAW
    voltage_eV=100e3,  # 100 keV
    sampling=0.1,  # Angstrom
    slice_thickness=0.5,  # Angstrom
)

print("Running multislice calculation...")
exitwaves = calculator.run()

print("Computing TACAW spectrum...")
tacaw = TACAWData(exitwaves)

# Plot spectral diffraction at 5 THz
frequency_THz = 5.0  # THz
Z = tacaw.spectral_diffraction(frequency_THz)

tacaw.plot(Z**0.1, "kx", "ky", filename=f"{OUTPUT_DIR}/tacaw_spectral.png")


print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
