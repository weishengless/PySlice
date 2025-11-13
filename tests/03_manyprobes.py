import sys,os
sys.path.insert(1,"../../")
from pyslice.io.loader import Loader
from pyslice.multislice.multislice import Probe,Propagate,create_batched_probes
from pyslice.multislice.potentials import gridFromTrajectory,Potential
from pyslice.postprocessing.testtools import differ
import numpy as np
import matplotlib.pyplot as plt
#from ..pyslice.tacaw.ms_calculator_npy import gridFromTrajectory
#from pyslice.tacaw.multislice_npy import Probe,Propagate ; import numpy as xp
#from pyslice.tacaw.multislice_torch import Probe,PropagateBatch,create_batched_probes ; import torch as xp
#from pyslice.tacaw.potential import Potential

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}
a,b=2.4907733333333337,2.1570729817355123

# LOAD MD OUTPUT
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
trajectory=trajectory.slice_positions([0,10*a],[0,10*b])
xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)

# GENERATE PROBE (ENSURE 00_PROBE.PY PASSES BEFORE RUNNING)
probe=Probe(xs,ys,mrad=30,eV=100e3)
x,y=np.meshgrid(np.linspace(a,3*a,16),np.linspace(b,3*b,16))
xy=np.reshape([x,y],(2,len(x.flat))).T
#print(xy)
probes_many=create_batched_probes(probe,xy)

# GENERATE THE POTENTIAL (ENSURE 01_POTENTIAL.PY PASSES BEFORE RUNNING)
positions = trajectory.positions[0]
atom_types=trajectory.atom_types
potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")

# SANITY CHECK THAT OUR CROPPED TRAJECTORY IS CORRECT
#ary=np.asarray(potential.array)
#fig, ax = plt.subplots()
#ax.imshow(np.sum(ary,axis=2), cmap="inferno")
#dx=potential.xs[1]-potential.xs[0] ; dy=potential.ys[1]-potential.ys[0]
#ax.plot(x/dx,y/dy)
#plt.show()
#pr=probes_many.array[5,:,:]
#po=np.sum(potential.array,axis=2)
#fig, ax = plt.subplots()
#ax.imshow(np.absolute(pr)*np.absolute(po), cmap="inferno")
#plt.show()

# TEST PROPAGATION
# Handle device conversion properly for PyTorch tensors
result = Propagate(probes_many,potential)
if hasattr(result, 'cpu'):
    ary = result.cpu().numpy()  # Convert PyTorch tensor to numpy
else:
    ary = np.asarray(result)  # Already numpy array


differ(ary[::2,::2,::2],"outputs/manyprobes-test.npy","EXIT WAVE")

# ASSEMBLE HAADF IMAGE
# Convert PyTorch tensors to numpy arrays for k-space calculations
if hasattr(potential.kxs, 'cpu'):
    kxs = potential.kxs.cpu().numpy()
    kys = potential.kys.cpu().numpy()
else:
    kxs = np.asarray(potential.kxs)
    kys = np.asarray(potential.kys)
q=np.sqrt(kxs[:,None]**2+kys[None,:]**2)
fig, ax = plt.subplots()
fft=np.fft.fft2(ary,axes=(1,2)) ; fft[:,q<2]=0 # mask in reciprocal space (keep only high scattering angles)
#ax.imshow(np.absolute(np.fft.fftshift(fft[0]))**.1, cmap="inferno")
#plt.show()
HAADF=np.sum(np.absolute(fft),axis=(1,2)).reshape((len(x),len(y)))
ax.imshow(HAADF, cmap="inferno")
plt.savefig("outputs/figs/03_manyprobes.png")

