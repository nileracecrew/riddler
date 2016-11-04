import scipy.integrate
import numpy as np

def p(k):
	"""k is ratio of guardian speed to alien speed."""
	def f(u):
		return 0.5*(1.0 + np.cos(0.5 * k * np.arccos(2*u-1)))

	out = scipy.integrate.quad(f, (np.cos(2.0 * np.pi/k) + 1)/2, 1.0)
	return out[0] 

"""
# wrong because cos(full angle) is what is uniformly distributed, not cos(half angle)
def p2(k):
	def f(u):
		return 0.5*(1.0 + np.cos(k * np.arccos(u)))

	out = scipy.integrate.quad(f, np.cos(np.pi/k), 1.0)
	return out[0] 
"""

def p2(k):
	"""k is ratio of guardian speed to alien speed."""
	def f(u):
		return 0.5*(1.0 + np.cos(0.5 * k * np.arccos(u)))

	out = scipy.integrate.quad(f, np.cos(2.0 * np.pi/k), 1.0)
	return out[0] / 2.0

def p3(k):
	"""k is ratio of guardian speed to alien speed."""
	def f(u):
		return 0.5*(1.0 + np.cos(0.5 * k * u)) * np.sin(u)

	out = scipy.integrate.quad(f, 0, 2.0 * np.pi / k)
	return out[0] / 2.0


def p4(k):
	"""k is ratio of guardian speed to alien speed."""
	def f(u):
		return 0.5*(1.0 + np.cos(u)) * np.sin(2*u/k) * 2 / k

	out = scipy.integrate.quad(f, 0, np.pi)
	return out[0] / 2.0


def p5(k):
	"""k is ratio of guardian speed to alien speed."""
	def f(u):
		return 0.5*(1.0 - np.cos(0.5 * k * u)) * np.sin(u)

	out = scipy.integrate.quad(f, 0, 2.0 * np.pi / k)
	out2 = scipy.integrate.quad(lambda u: np.sin(u), 2.0 * np.pi / k, np.pi)
	return (out[0] + out2[0]) / 2.0



"""
# naive integration
def p2(k):
	def f(theta):
		return 0.5*(1.0 + np.cos(theta))

	out = scipy.integrate.quad(f, 0.0, np.pi/k)
	return out[0]
"""

print 1.0 - p(20.0)
print 1.0 - p2(20.0)
print 1.0 - p3(20.0)
print 1.0 - p4(20.0)
print p5(20.0)


