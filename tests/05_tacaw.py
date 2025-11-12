import sys,os
sys.path.insert(1,"../../")
from src.io.loader import Loader
from src.multislice.multislice import probe_grid
from src.multislice.calculators import MultisliceCalculator
from src.postprocessing.tacaw_data import TACAWData
from src.postprocessing.testtools import differ
import numpy as np
import matplotlib.pyplot as plt
import os,shutil

#if os.path.exists("psi_data"):
#	shutil.rmtree("psi_data")

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}
a,b=2.4907733333333337,2.1570729817355123

# LOAD TRAJECTORY
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()

# TRIM TO EVERY OTHER FRAME
#trajectory = trajectory.slice_timesteps(ith=2)

# TRIM TO 10x10 UC
#trajectory=trajectory.slice_positions([0,10*a],[0,10*b])

# TACAW CALCULATION: ALL TIMESTEPS, LET'S DO PARALLEL BEAM
calculator=MultisliceCalculator()
calculator.setup(trajectory,aperture=0,voltage_eV=100e3,sampling=.1,slice_thickness=.5)
exitwaves = calculator.run()

exitwaves.plot(nuke_zerobeam=True,powerscaling=.125)

tacaw = TACAWData(exitwaves)


# MANUAL PLOTTING
#print(tacaw.frequencies, tacaw.frequencies[65])
#intensity_data = tacaw.intensity[0,65,:,:]**.1
#ary = intensity_data.cpu().numpy() if hasattr(intensity_data, 'cpu') else np.asarray(intensity_data)

#fig, ax = plt.subplots()
#ax.imshow(ary.T, cmap="inferno")
#plt.show()
#plt.savefig("outputs/figs/05_tacaw.png")

#differ(ary,"outputs/tacaw-test.npy","TACAW SLICE")

# OR PLOT USING BUILT IN TOOLS: AN ENERGY SLICE:
Z = tacaw.spectral_diffraction(30) ; print(Z.shape)
tacaw.plot(Z**.1,"kx","ky",filename="outputs/figs/05_tacaw.png")

differ(Z**.1,"outputs/tacaw-test.npy","TACAW SLICE")

# OR THE DISPERSION:
kx=np.asarray(tacaw.kxs) ; kx=kx[kx>=0] ; kx=kx[kx<=4/a] ; print("kx",kx.shape)
dispersion = tacaw.dispersion( kx , np.zeros(len(kx))+2/b )
tacaw.plot(dispersion**.125,kx,"omega")


