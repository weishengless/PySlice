import sys,os
sys.path.insert(1,"../../")
from src.io.loader import Loader
from src.multislice.multislice import Probe,Propagate
from src.multislice.potentials import gridFromTrajectory,Potential
from src.multislice.calculators import MultisliceCalculator
from src.postprocessing.testtools import differ
import numpy as np
import matplotlib.pyplot as plt
import os,shutil

# PROBE
xs=np.linspace(0,50,501)
ys=np.linspace(0,49,491)
probe=Probe(xs,ys,mrad=30,eV=100e3)

probe.aberrate({"C30":500000,"C12":10})

# FOCUS TABLEAU
nx,ny = 6,5
z1=-1000 ; zf=0
dz = (zf-z1)/(nx*ny-1)
#zs = np.linspace(-1000,100,nx*ny) ; dz=zs[1]-zs[0]

fig,axs = plt.subplots(ny,nx)
ct=0 ; z=0
for j in range(ny):
	for i in range(nx):
		if ct==0:
			probe.defocus(z1) ; z=z1
		else:
			probe.defocus(dz) ; z+=dz
		axs[j,i].set_title("z="+str(np.round(z))+"$\AA$")
		axs[j,i].imshow(np.absolute(probe.array))
		ct+=1

plt.show()