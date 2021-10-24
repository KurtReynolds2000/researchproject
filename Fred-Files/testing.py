import numpy as np
from Functions import Rosenbrock

bounds = np.array([[1,2,3,4],[3,4,5,6]])
store = np.array([2,3,4,5])
ran = len(bounds)

array = np.array([bounds[i,0] <= store[i] <= bounds[i,1] for i in [0,1]])
print(array.all())
