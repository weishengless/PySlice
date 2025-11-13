import sys,os,glob
from pyslice.io.loader import Loader
from pyslice.multislice.potentials import gridFromTrajectory, Potential
from pyslice.postprocessing.testtools import differ
import matplotlib.pyplot as plt
import numpy as np

# Remove previously-cached npy files (we want to rest reloading them)
outfiles=glob.glob("inputs/*.npy")
for f in outfiles:
	if "-test" not in f:
		os.remove(f)

# https://www.ovito.org/manual/reference/file_formats/file_formats_input.html
# https://www.ovito.org/manual/usage/data_model.html#usage-particle-properties

# define our test input files. all of these should be supported
testFiles={"silicon.positions":{"atom_style":"molecular"},	# lammps input positions file, generated via generatePositions.py
	"silicon.xyz": None,               # xyz file following wikipedia conventions (https://en.wikipedia.org/wiki/XYZ_file_format), generated via generatePositions.py
	"silicon.cif": None,		       # generated via ase.io.read/write from silicon.xyz
	"hBN.cif": None,				   # taken from DOI: 10.1016/j.matlet.2006.07.108 https://materials.springer.com/isp/crystallographic/docs/sd_1923917
	"hBN.xyz": None,				   # taken from DOI: 10.17863/CAM.66112
	"hBN_truncated.lammpstrj": None,   # multiple timesteps, generated via a custom dump command from lammps
    "hBN_GAP_ase.trj": None,           # multiple timesteps, generated via a ASE MD run using the GAP potential from https://doi.org/10.1021/acs.jpcc.0c05831
}


# for each: load, generate potential, plot potential
for i,filename in enumerate(testFiles.keys()):
	print("attempting to load","inputs/"+filename)
	trajectory=Loader("inputs/"+filename,ovitokwargs=testFiles[filename]).load()
	#trajectory = trajectory.generate_random_displacements(n_displacements=10,sigma=1)
	#print(len(trajectory.positions))
	positions = trajectory.positions[0]
	atom_types=trajectory.atom_types
	xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)
	potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
	potential.plot("outputs/figs/06_loaders_"+str(i)+".png")
	differ(positions,"outputs/loaders-test_"+filename+".npy","POSITIONS")

# Test loading from ASE Atoms object (single frame)
print("\nTesting ASE Atoms object loading (single frame)")
from ase import Atoms
from ase.build import bulk
atoms = bulk('Si', 'diamond', a=5.43) * (3, 3, 3)
trajectory_from_atoms = Loader(atoms=atoms).load()
print(f"✓ Loaded from ASE Atoms: {trajectory_from_atoms.n_frames} frames, {trajectory_from_atoms.n_atoms} atoms")
print(f"✓ ASE Atoms single frame loading test PASSED")

# Test loading from ASE Atoms trajectory (multiple frames)
print("\nTesting ASE Atoms trajectory loading (multiple frames)")
from ase.io.trajectory import Trajectory as ASETrajectory
import tempfile
import os

# Create a temporary trajectory file with multiple frames
with tempfile.NamedTemporaryFile(suffix='.traj', delete=False) as tmp:
    tmp_path = tmp.name

try:
    # Create trajectory with 10 frames with slightly perturbed positions
    traj_writer = ASETrajectory(tmp_path, 'w')
    base_atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)

    for i in range(10):
        atoms_frame = base_atoms.copy()
        # Add small random displacement to simulate dynamics
        positions = atoms_frame.get_positions()
        positions += np.random.normal(0, 0.05, positions.shape)
        atoms_frame.set_positions(positions)
        traj_writer.write(atoms_frame)

    traj_writer.close()

    # Read the trajectory back
    traj_reader = ASETrajectory(tmp_path, 'r')

    # Load all frames into a single Trajectory object
    trajectory_multi = Loader(atoms=traj_reader).load()

    print(f"✓ Loaded from ASE trajectory: {trajectory_multi.n_frames} frames, {trajectory_multi.n_atoms} atoms")
    assert trajectory_multi.n_frames == 10
    assert trajectory_multi.n_atoms == 16
    print(f"✓ ASE Atoms multi-frame loading test PASSED")

finally:
    # Clean up temporary file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

