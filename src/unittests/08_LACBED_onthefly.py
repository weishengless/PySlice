import sys,os
sys.path.insert(1,"../../")
from src.io.loader import Loader
from src.multislice.multislice import probe_grid
from src.multislice.calculators import MultisliceCalculator
from src.postprocessing.haadf_data import HAADFData
from src.multislice.potentials import gridFromTrajectory,Potential
import numpy as np
import matplotlib.pyplot as plt
import os,shutil

#dump="inputs/hBN_truncated.lammpstrj" ; dt=.005 ; types={1:"B",2:"N"}
dump="inputs/Si_truncated.lammpstrj" ; dt=.002 ; types={1:"Si"}
#dump="inputs/silicon.cif" ; dt=.002 ; types={1:"Si"}


# LOAD TRAJECTORY
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
# SELECT "RANDOM" TIMESTEPS (use seed for reproducibility)
slice_timesteps = np.arange(trajectory.n_frames)
np.random.seed(5) ; np.random.shuffle(slice_timesteps)
trajectories = [ trajectory.slice_timesteps( [i] ) for i in slice_timesteps[:10] ]
trajectory = trajectories[0].tile_positions([1,1,10],trajectories)
#trajectory = trajectories[0]

# WHAT IS LACBED? 
#  \  Defocused sample yields diffraction spots at
#   \  the original sample plane. Each spot is a
#    \___________/  real-space probe, and the
#    /\       .-'  magnification of the diffraction
#   /  \   .-'/  \  pattern depends on the distance
#  /    \-'  /    \  defocused. With Midgley, you're
# /  .-' \  / '-.  \  interested in this diffraction
#/.-'     \/     '-.\  pattern, but with LACBED, you
#        (__)  apply a selected-area aperture to one
#        /  \  and go back to reciprocal space to 
#       /    \  view the "texture" of a single disk

# OPTION TWO, LET TRAJECTORY.PY DO THE STACKING
calculator=MultisliceCalculator()
calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5)			
calculator.base_probe.defocus(-1000)
exitwaves = calculator.run()
exitwaves.propagate_free_space(1000-calculator.lz)
exitwaves.applyMask(5,"real")
exitwaves.plot_reciprocalspace("outputs/figs/08_LACBED_onthefly.png")

ary=exitwaves.array

# Convert to numpy if it's a torch tensor
if hasattr(ary, 'cpu'):
	ary = ary.cpu().numpy()

print(ary.shape)
if not os.path.exists("outputs/lacbed-test.npy"):
	np.save("outputs/lacbed-test.npy",ary[::3,::3])
else:
	previous=np.load("outputs/lacbed-test.npy")
	F , D = np.absolute(ary)[::3,::3] , np.absolute(previous)
	dz=np.sum( (F-D)**2 ) / np.sum( F**2 ) # a scaling-resistant values-near-zero-resistance residual function
	if dz>1e-6:
		print("ERROR! LACBED DOES NOT MATCH PREVIOUS RUN",dz*100,"%")

