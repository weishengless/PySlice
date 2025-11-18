import sys,os
from datetime import datetime
import numpy as np

nx,ny,nz=10,10,10	# how many unit cells in x,y,z

a,b,c=5.43729,5.43729,5.43729 # unit cell definition, Å,°

positionsInUnitCell=[[0.,0.,0.],[.5,.5,0.],[.5,0.,.5],[0.,.5,.5],[.25,.25,.25],[.75,.75,.25],[.75,.25,.75],[.25,.75,.75]]
UCID=[1,2,3,4,1,2,3,4] ; CCID=[1,1,1,1,2,2,2,2]
masses=[28.0855]

# set up atom field (cycle through unit cells)
nA=nx*ny*nz*len(positionsInUnitCell)
atoms=np.zeros((nA,4)) # list of atoms, including x,y,z position, and type
CCs=np.zeros(nA,dtype=int) ; UCIDs=np.zeros(nA,dtype=int) # crystal coordinate of each atom ("which atom is this within the unit cell") and unit cell ID ("which unit cell")
ct=-1
for i in range(nx):
	for j in range(ny):
		for k in range(nz):
			for n in range(len(positionsInUnitCell)):
				ct+=1
				x,y,z=positionsInUnitCell[n]	# UC positions
				x*=a ; y*=b ; z*=c
				x+=i*a ; y+=j*b ; z+=k*c	# offset by whichever unit cell we're in
				
				atoms[ct,:]=[x,y,z,1]
				#cc=n+1 ; uc=ct//len(positionsInUnitCell)+1 # WRONG. there technically aren't 8 atoms per UC, there are 2! using 8 will give unnecessary folding
				ucid = ct//len(positionsInUnitCell) # WHICH (macro) UC are we in?
				cc=CCID[n] ; ucid=ucid*4+UCID[n]
				CCs[ct]=cc ; UCIDs[ct]=ucid # USED TO CREATE LATTICE MAPPING FILE FOR pSED ("lattice.dat")
# update the type for all atoms where z position is above the interface

lx=a*nx ; ly=b*ny ; lz=c*nz

# LAMMPS INPUT FILE
#[ "0.0 "+str(n*l)+" "+c+"lo "+c+"hi" for n,l,c in zip([nx,ny,nz],[a,b,c],["x","y","z"])] +\
lines=[ "########### "+sys.argv[0]+" "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+" ###########",
	str(len(atoms))+" atoms" , "" , "1 atom types", "" ] +\
[ "0.0 "+str(lx)+" xlo xhi" , "0.0 "+str(ly)+" ylo yhi" , "0.0 "+str(lz)+" zlo zhi" ] +\
[ "" , "Masses" , "" ] +\
[ str(n+1)+" "+str(m) for n,m in enumerate(masses) ] +\
[ "Atoms" , "" ] +\
[ str(n+1)+" 1 "+str(int(a[3]))+" "+str(float(a[0]))+" "+str(float(a[1]))+" "+str(float(a[2])) for n,a in enumerate(atoms) ]

with open("silicon.positions",'w') as f:
	for l in lines:
		f.write(l+"\n")


# XYZ FILE, SIMPLY ATOMICTYPE X Y Z
#[ "0.0 "+str(n*l)+" "+c+"lo "+c+"hi" for n,l,c in zip([nx,ny,nz],[a,b,c],["x","y","z"])] +\
lines=[ str(len(atoms)) , "" ] +\
[ "Si\t"+str(float(a[0]))+"\t"+str(float(a[1]))+"\t"+str(float(a[2])) for n,a in enumerate(atoms) ]

with open("silicon.xyz",'w') as f:
	for l in lines:
		f.write(l+"\n")

# CONVERT XYZ TO CIF
from ase.io import read
atoms = read("silicon.xyz")

from ase.io import write
write("silicon.cif",atoms)
