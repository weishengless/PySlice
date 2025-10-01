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

#dump="hBN_truncated.lammpstrj" ; dt=.005 ; types={1:"B",2:"N"}
dump="Si_truncated.lammpstrj" ; dt=.002 ; types={1:"Si"}
#dump="silicon.cif" ; dt=.002 ; types={1:"Si"}


# LOAD TRAJECTORY
trajectory=TrajectoryLoader(dump,timestep=dt,atom_mapping=types).load()
# SELECT "RANDOM" TIMESTEPS (use seed for reproducibility)
slice_timesteps = np.arange(trajectory.n_frames)
np.random.seed(5) ; np.random.shuffle(slice_timesteps)
trajectories = [ trajectory.slice_timesteps( [i] ) for i in slice_timesteps[:10] ]

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

for i in range(10):
	# SET UP SIMULATION
	calculator=MultisliceCalculator()
	calculator.setup(trajectories[i],aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5)			
	# LACBED USES A DEFOCUSED THE PROBE		
	if i==0:								
		calculator.base_probe.defocus(-1000)
	else:									
		calculator.base_probe.array = last_slice_exit
	# RUN THE SIMULATION					
	exitwaves = calculator.run()
	last_slice_exit = np.fft.ifft2(np.fft.ifftshift(exitwaves.array[0,0,:,:,0]))

# REPROPAGATE TO PROBE FOCAL POINT		
exitwaves.propagate_free_space(1000-10*calculator.lz)

# PREVIEW UNMASKED REAL AND RECIPROCAL SPACE
#exitwaves.plot_reciprocalspace()
#exitwaves.plot_realspace()

# APPLY MASK ABOUT REAL-SPACE DIFFRACTED PROBE
exitwaves.applyMask(5,"real")

#exitwaves.plot_realspace()
exitwaves.plot_reciprocalspace()

