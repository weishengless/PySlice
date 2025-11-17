import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice.io.loader import Loader
from pyslice.multislice.potentials import gridFromTrajectory,Potential
from pyslice.multislice.potentials2 import gridFromTrajectory as gridFromTrajectory2
from pyslice.multislice.potentials2 import Potential as Potential2
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

# TEST GENERATION OF THE POTENTIAL
positions = trajectory.positions[0]
atom_types=trajectory.atom_types
xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)
xs2,ys2,zs2,lx2,ly2,lz2=gridFromTrajectory2(trajectory,sampling=0.1,slice_thickness=0.5)
potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
potential2 = Potential2(xs2, ys2, zs2, positions, atom_types, kind="kirkland")
potential3 = Potential2(xs2, ys2, zs2, positions, atom_types, kind="kirkland", device='cpu')

potential.build()
potential2.build()
potential3.build()

ary=potential.to_cpu()  # Convert to CPU numpy array properly
ary2=potential2.to_cpu()  # Convert to CPU numpy array properly
ary3=potential3.to_cpu()  # Convert to CPU numpy array properly

print(np.max(ary-ary2, axis=(0,1)))
print(np.max(ary-ary3, axis=(0,1)))

differ(ary2[::3,::3,:],"outputs/potentials2-test.npy","POTENTIAL")

potential.plot("outputs/figs/01_potentials2.png")

#import matplotlib.pyplot as plt
#fig, ax = plt.subplots()
#ax.imshow(np.sum(ary,axis=2), cmap="inferno")
#plt.show()
