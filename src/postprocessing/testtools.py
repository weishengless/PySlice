import os
import numpy as np

def differ(ary,filename,label):
	if not os.path.exists(filename):
		print("diff npy file does not exist. creating anew")
		np.save(filename,ary)
	else:
		previous=np.load(filename)
		F , D = np.absolute(ary) , np.absolute(previous)
		dz=np.sum( (F-D)**2 ) / np.sum( F**2 ) # a scaling-resistant values-near-zero-resistance residual function
		if dz>1e-6:
			print("ERROR! "+label+" DOES NOT MATCH PREVIOUS RUN",dz*100,"%")
