import sys,os
from pyslice.io.loader import Loader
from pyslice.multislice.multislice import Probe,Propagate
from pyslice.multislice.potentials import gridFromTrajectory,Potential
from pyslice.postprocessing.testtools import differ
import numpy as np
#from ..pyslice.tacaw.ms_calculator_npy import gridFromTrajectory
#from pyslice.tacaw.multislice_npy import Probe,Propagate ; import numpy as xp
#from pyslice.tacaw.multislice_torch import Probe,PropagateBatch,create_batched_probes ; import torch as xp
#from pyslice.tacaw.potential import Potential

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}

# LOAD MD OUTPUT
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)

# GENERATE PROBE (ENSURE 00_PROBE.PY PASSES BEFORE RUNNING)
probe=Probe(xs,ys,mrad=5,eV=100e3)

# GENERATE THE POTENTIAL (ENSURE 01_POTENTIAL.PY PASSES BEFORE RUNNING)
positions = trajectory.positions[0]
atom_types=trajectory.atom_types
potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")

# TEST PROPAGATION
# Handle device conversion properly for PyTorch tensors
result = Propagate(probe,potential,onthefly=True)
if hasattr(result, 'cpu'):
    ary = result.cpu().numpy()  # Convert PyTorch tensor to numpy
else:
    ary = np.asarray(result)  # Already numpy array

differ(ary,"outputs/propagate-test.npy","EXIT WAVE")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
#ax.imshow(np.absolute(ary), cmap="inferno")
#plt.show()
ax.imshow(np.absolute(np.fft.fftshift(np.fft.fft2(ary)))**.1, cmap="inferno")
plt.savefig("outputs/figs/02_propagate_otf=True.png")

#result.plot()
