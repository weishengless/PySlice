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
xs=np.linspace(0,70,701)
ys=np.linspace(0,69,691)
probe=Probe(xs,ys,mrad=30,eV=100e3,gaussianVOA=.1)
# does it matter whether you aberrate the focused or defocused probe? i don't think so??

s=.02 ; l=probe.wavelength
jellyfish = { "C10":-75*2*s/l**2, "C30":5*4*s/l**4, 
	"C12":(2*2*s/l**2,.07), "C32":(2.5*4*s/l**4,.1), 
	"C21":(-13*3*s/l**3,.2), "C23":(0*3*s/l**3,.1) }
# Andy has: r = np.sqrt(x*x + y*y) ; theta = np.arctan2(x,y)
# rand += -r*r*75 + 5*r**4 					# if r^2 n=1 if r^4 n=3, if no cos then m=0
# rand += r*r*2.0*np.cos(2*(theta+0.07))	# r^2 means n=1, cos2 means m=2
# 2.5*np.cos(2*(theta+0.1))“*1*r**4			# r^4 means n=3, cos2 means m=2
# rand +=  -13*np.cos((theta+0.2))*1*r**3	# r^3 means n=2, cos1 means m=1
#  -0*np.cos(3*(theta-0.1))*1*r**3			# r^3 means n=2, cos2 means m=3
# And recall: χ(k,ϕ) = π/2/λ 1/(n+1) C ( k λ )^(n+1) cos(m*(ϕ-ϕa)) for Cnm
dummy = {"C30":1000000, # spherical for caustics
	"C23":(500,np.pi/3)}

#probe.plot()
probe.aberrate(jellyfish) # stig
	#"C21":10 }) # coma
#probe.plot()



# FOCUS TABLEAU
nx,ny = 7,6
z1=1600 ; zf=2600
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
		axs[j,i].imshow(np.absolute(probe.array)**2)
		ct+=1

plt.show()