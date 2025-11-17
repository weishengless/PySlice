import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')
from pyslice.io.loader import Loader
from pyslice.multislice.multislice import probe_grid,Probe,Propagate
from pyslice.multislice.potentials import gridFromTrajectory,Potential
from pyslice.multislice.calculators import MultisliceCalculator
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
trajectory=trajectory.slice_timesteps( [0] )

# POTENTIAL
positions = trajectory.positions[0]
atom_types=trajectory.atom_types
xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)
potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
#potential.plot()

# PROBE
probe=Probe(xs,ys,mrad=30,eV=100e3)
zmax=np.amax(np.absolute(np.asarray(probe.array.cpu())))
# 3D PLOT OF THE PROBE WAIST
#probe.defocus(-300)
zs=np.linspace(-1000,1000,21) # +/- 100 nm (this is units of Angstrom)
probe.defocus(zs[0]) ; dz=zs[1]-zs[0]
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
for z in zs:
	ary=np.absolute(probe.array.cpu())
	# Convert tensors to numpy for matplotlib
	xs_np = probe.xs.cpu().numpy() if hasattr(probe.xs, 'cpu') else probe.xs
	ys_np = probe.ys.cpu().numpy() if hasattr(probe.ys, 'cpu') else probe.ys
	CS=plt.contour(xs_np[::3], ys_np[::3], (z+ary[::3,::3]/zmax).T) #, levels=lv,alpha=alpha,cmap=cmap)
	probe.defocus(dz)
#plt.show()	
plt.savefig("outputs/figs/07_defocus_3D.png")


#probe.plot()
probe=Probe(xs,ys,mrad=30,eV=100e3)
probe.defocus(10*1e2)
probe.plot("outputs/figs/07_defocus_2D.png")

differ(probe.array,"outputs/defocus-test.npy","DEFOCUSED PROBE")

result = Propagate(probe,potential)
if hasattr(result, 'cpu'):
    result = result.cpu()

#import matplotlib.pyplot as plt
#fig, ax = plt.subplots()
#ax.imshow(np.absolute(np.fft.fftshift(np.fft.fft2(result))), cmap="inferno")
#plt.show()

