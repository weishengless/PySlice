import sys,os,glob
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')
from pyslice.io.loader import Loader
from pyslice.multislice.potentials import gridFromTrajectory, Potential
from pyslice.multislice.multislice import probe_grid
from pyslice.postprocessing.haadf_data import HAADFData
from pyslice.postprocessing.testtools import differ
from pyslice.multislice.calculators import MultisliceCalculator
import matplotlib.pyplot as plt
import numpy as np

cif = "inputs/hBN.cif" ; a,b=2.4907733333333337,2.1570729817355123
trajectory=Loader(cif).load()
trajectory = trajectory.tile_positions([5,5,1])
trajectory = trajectory.generate_random_displacements(n_displacements=20,sigma=.1)

positions = trajectory.positions[0]
atom_types=trajectory.atom_types
xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)
potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
potential.plot("out.png")



xy=probe_grid([a,3*a],[b,3*b],14,16) # comment out to default to middle position
#xy = None

calculator=MultisliceCalculator()
calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_positions=xy)
exitwaves = calculator.run()

exitwaves.plot_reciprocal() # can also specify whichProbe and whichTimestep instead of averaging

haadf=HAADFData(exitwaves)
ary=haadf.calculateADF(preview=False) # use preview=True to view the collection angles of the ADF detector in reciprocal space
haadf.plot()


array = exitwaves.array

print("array shape",array.shape)