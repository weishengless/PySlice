import sys,os
sys.path.insert(1,"../../")
from src.io.loader import Loader
from src.multislice.multislice import probe_grid
from src.multislice.calculators import SEDCalculator
from src.multislice.sed import SED
from src.postprocessing.tacaw_data import TACAWData
from src.postprocessing.testtools import differ
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

# SET UP OUT OWN RECIPROCAL SPACE GRID FOR SED
#lx,ly,lz = np.diag(trajectory.box_matrix)
#nx,ny = int(np.round(lx/a)) , int(np.round(ly/b))
#print(nx,ny)
#kxs=np.linspace(0,4*np.pi/a,nx)
#kys=np.linspace(0,4*np.pi/b,ny)

#kvec = np.zeros((len(kxs),len(kys),3))
#kvec[:,:,0] += kxs[:,None]
#kvec[:,:,1] += kys[None,:]

#avg = trajectory.get_mean_positions()
#disp = trajectory.get_distplacements()
#print(np.shape(avg),np.shape(disp))

# RUN SED INSTEAD OF MULTISLICE
#Zx,ws = SED(avg,disp,kvec=kvec,v_xyz=0)
#Zy,ws = SED(avg,disp,kvec=kvec,v_xyz=1)
#Zz,ws = SED(avg,disp,kvec=kvec,v_xyz=2)

#ws/=dt

calculator=SEDCalculator()
calculator.setup(trajectory,abc=[a,b,1])
calculator.run()

#Zx=calculator.Zx
#Zy=calculator.Zy
#Zz=calculator.Zz
#ws=calculator.ws
#kxs=calculator.kxs
#kys=calculator.kys

#Zx=np.reshape(Zx,(len(ws),nx,ny))
#Zy=np.reshape(Zy,(len(ws),nx,ny))
#Zz=np.reshape(Zz,(len(ws),nx,ny))

calculator.plot(30)

