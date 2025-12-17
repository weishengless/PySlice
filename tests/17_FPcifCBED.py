import sys,os,glob
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice import Loader,probe_grid,gridFromTrajectory,Potential,MultisliceCalculator,HAADFData,differ

import matplotlib.pyplot as plt
import numpy as np

cif = "inputs/hBN.cif" ; a,b=2.4907733333333337,2.1570729817355123
trajectory=Loader(cif).load()
trajectory = trajectory.tile_positions([5,5,1])
trajectory = trajectory.generate_random_displacements(n_displacements=20,sigma=.1,seed=0) # PRO TIP: use a random seed so you get repeatable "random" configurations and don't need to re-multislice on subsequent runs

positions = trajectory.positions[0]
atom_types=trajectory.atom_types
xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)
potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
potential.plot("outputs/figs/17_potential.png")

xy=probe_grid([a,3*a],[b,3*b],14,16) # comment out to default to middle position
#xy = None

calculator=MultisliceCalculator()
calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_positions=xy)
exitwaves = calculator.run()

exitwaves.plot_reciprocal("outputs/figs/17_cbed.png") # can also specify whichProbe and whichTimestep instead of averaging

haadf=HAADFData(exitwaves)
ary=haadf.calculateADF(preview=False) # use preview=True to view the collection angles of the ADF detector in reciprocal space
haadf.plot("outputs/figs/17_haadf.png")


array = exitwaves.array

print("array shape",array.shape)