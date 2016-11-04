import scipy.integrate
import numpy as np

def wmin(u,v):
	return np.cos(np.arccos(u) - np.arccos(v))

out = scipy.integrate.tplquad(lambda x,y,z: 0.25, 
                         -1.0, 1.0,
                         lambda u: u, lambda u: 1.0,
                         wmin, lambda u,v,: 1.0)

print out