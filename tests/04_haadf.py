import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')
from pyslice.io.loader import Loader
from pyslice.multislice.multislice import probe_grid
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.postprocessing.haadf_data import HAADFData
from pyslice.postprocessing.testtools import differ
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
# TRIM TO 10x10 UC
trajectory=trajectory.slice_positions([0,10*a],[0,10*b])
# SELECT 10 "RANDOM" TIMESTEPS (use seed for reproducibility)
trajectory=trajectory.get_random_timesteps(3,seed=5)
# SET UP GRID OF HAADF SCAN POINTS
xy=probe_grid([a,3*a],[b,3*b],14,16)

calculator=MultisliceCalculator()
calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_positions=xy,cache_levels=[])
exitwaves = calculator.run()

exitwaves.plot_reciprocal(filename="outputs/figs/04_haadf_cbed.png")
#print(exitwaves.wavefunction_data.shape)
# exitwaves.wavefunction_data is reciprocal-space now! 
#ary=np.mean(np.absolute(exitwaves.wavefunction_data[:,:,:,:,-1]),axis=1)
#q=np.sqrt(exitwaves.kxs[:,None]**2+exitwaves.kys[None,:]**2)
#mask=np.zeros(q.shape) ; mask[q>2]=1
#fig, ax = plt.subplots()
#print(ary.shape,q.shape)
#ax.imshow(np.absolute(ary[0])**.1, cmap="inferno")
#plt.show()
#fig, ax = plt.subplots()
#HAADF=np.sum(np.absolute(ary*mask[None,:]),axis=(1,2)).reshape((len(x),len(y)))
#ax.imshow(HAADF, cmap="inferno")
#plt.show()

haadf=HAADFData(exitwaves)
ary=haadf.calculateADF(preview=False) # use preview=True to view the collection angles of the ADF detector in reciprocal space
xs=haadf.xs ; ys=haadf.ys

#fig, ax = plt.subplots()
#ax.imshow(ary.T, cmap="inferno")
#plt.show()
haadf.plot("outputs/figs/04_haadf.png")

ary=np.asarray(ary)
differ(ary,"outputs/haadf-test.npy","HAADF")
