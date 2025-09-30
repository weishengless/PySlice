import sys,os
sys.path.insert(1,"../../")
from src.io.loader import TrajectoryLoader
from src.multislice.multislice import probe_grid
from src.multislice.calculators import MultisliceCalculator
from src.postprocessing.haadf_data import HAADFData
from src.multislice.potentials import gridFromTrajectory,Potential
import numpy as np
import matplotlib.pyplot as plt
import os,shutil

dump="hBN_truncated.lammpstrj" ; dt=.005 ; types={1:"B",2:"N"}
#dump="Si_truncated.lammpstrj" ; dt=.002 ; types={1:"Si"}
#dump="silicon.cif" ; dt=.002 ; types={1:"Si"}


# LOAD TRAJECTORY
trajectory=TrajectoryLoader(dump,timestep=dt,atom_mapping=types).load()
# SELECT "RANDOM" TIMESTEPS (use seed for reproducibility)
slice_timesteps = np.arange(trajectory.n_frames)
np.random.seed(5) ; np.random.shuffle(slice_timesteps)
slice_timesteps = slice_timesteps[:1]
trajectory=trajectory.slice_timesteps( slice_timesteps )

#positions = trajectory.positions[0]
#atom_types=trajectory.atom_types
#xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)
#potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
#potential.plot()


# SET UP SIMULATION
calculator=MultisliceCalculator()
calculator.setup(trajectory,			#  \  defocused sample yields diffraction
	aperture=30,voltage_eV=100e3,		#   \  spots where each spot is a real-space
	sampling=.1,slice_thickness=.5)		#    \  probe with /  diffraction in the image
# DLACBED USES A DEFOCUSED THE PROBE	#     \ __________/  plane where each probe 
calculator.base_probe.defocus(-1000)	#     /\       .-'  previously converged. 
# RUN THE SIMULATION					#    /  \   .-'/  \ 
exitwaves = calculator.run()			#   /    \-'  /    \ 
# REPROPAGATE TO PROBE FOCAL POINT		#  /  .-' \  / '-.  \ 
exitwaves.propagate_free_space(1000-calculator.lz)	# /.-'     \/     '-.\

# PREVIEW UNMASKED REAL AND RECIPROCAL SPACE
exitwaves.plot_reciprocalspace()
exitwaves.plot_realspace()

# APPLY MASK ABOUT REAL-SPACE DIFFRACTED PROBE
exitwaves.applyMask(5,"real")

exitwaves.plot_realspace()
exitwaves.plot_reciprocalspace()

