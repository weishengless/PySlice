import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')
from pyslice.multislice.multislice import Probe
from pyslice.postprocessing.testtools import differ
import numpy as np

# Generate a few dummy probes
xs=np.linspace(0,50,501)
ys=np.linspace(0,49,491)

mrads=[1,3,5,15,30]
ary=np.zeros((5,501,491),dtype=complex)
for i,mrad in enumerate(mrads):
	probe=Probe(xs,ys,mrad=mrad,eV=100e3,preview=False,gaussianVOA=0)
	probe.plot("outputs/figs/00_probe_"+str(i)+".png")
	if hasattr(probe, 'to_cpu'):
		ary[i] = probe.to_cpu()
	else:
		ary[i] = np.asarray(probe.array)
ary=np.reshape(ary,(501*5,491))

differ(ary,"outputs/probe-test.npy","PROBE")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(np.absolute(ary.T)**.25, cmap="inferno")
plt.savefig("outputs/figs/00_probe.png")
#plt.show()
