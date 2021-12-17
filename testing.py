
import numpy as np
import matplotlib.pyplot as plt
import Functions as fun
from Coded_Algorithms import hybrid_algorithms as hyb
from Coded_Algorithms import Algorithms as alg
from scipy.optimize import differential_evolution

bounds = np.asarray([[-5,5]]*10)
dimension = len(bounds)
max_eval = 100000

function = fun.Rosenbrock
np.set_printoptions(precision=5)


[obj_class, obj_track, obj_eval, timing] = hyb.mayfly_alg(function,bounds,max_eval,max_eval,seed=123)
best_eval = obj_class.eval
best_val = obj_class.xarray
print("{:.10f}".format(best_eval), best_val,obj_class.message)
print(obj_track[::5])
print(obj_eval[::5])
plt.plot(obj_eval, obj_track)
plt.yscale('log')
plt.xlabel("Objective Function Evaluation")
plt.ylim((10e-4,10e4))
plt.ylabel("Objective Function")