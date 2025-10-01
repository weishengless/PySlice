import sys,os,glob
sys.path.insert(1,"../../")
from src.io.loader import TrajectoryLoader
from src.multislice.potentials import gridFromTrajectory,Potential
import matplotlib.pyplot as plt
import numpy as np

# Remove previously-cached npy files (we want to rest reloading them)
outfiles=glob.glob("*.npy")
for f in outfiles:
	if "-test" not in f:
		os.remove(f)

# https://www.ovito.org/manual/reference/file_formats/file_formats_input.html
# https://www.ovito.org/manual/usage/data_model.html#usage-particle-properties

# define our test input files. all of these should be supported
testFiles={"silicon.positions":{"atom_style":"molecular"},	# lammps input positions file, generated via generatePositions.py
	"silicon.xyz":None,			# xyz file following wikipedia conventions (https://en.wikipedia.org/wiki/XYZ_file_format), generated via generatePositions.py
	"silicon.cif":None,		# generated via ase.io.read/write from silicon.xyz
	"hBN.cif":None,				# taken from DOI: 10.1016/j.matlet.2006.07.108 https://materials.springer.com/isp/crystallographic/docs/sd_1923917
	"hBN.xyz":None,				# taken from DOI: 10.17863/CAM.66112
	"hBN_truncated.lammpstrj":None}	# multiple timesteps, generated via a custom dump command from lammps


# for each: load, generate potential, plot potential
for i,filename in enumerate(testFiles.keys()):
	print("attempting to load",filename)
	trajectory=TrajectoryLoader(filename,ovitokwargs=testFiles[filename]).load()
	#trajectory = trajectory.generate_random_displacements(n_displacements=10,sigma=1)
	#print(len(trajectory.positions))
	positions = trajectory.positions[0]
	atom_types=trajectory.atom_types
	xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)
	potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
	potential.plot(str(i)+".png")