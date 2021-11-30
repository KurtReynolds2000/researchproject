import numpy as np
import matplotlib.pyplot as plt
import Functions as fun
import Algorithms as alg
import hybrid_algorithms as hyb
from scipy.optimize import differential_evolution

bounds = np.asarray([[-10,10]]*10)
dimension = len(bounds)
max_eval = 100000

function = fun.Rastrigin
np.set_printoptions(precision=4)


[obj_class, obj_track, obj_eval, timing] = hyb.mayfly_alg(function, bounds, max_eval, max_eval)
best_eval = obj_class.eval
best_val = obj_class.xarray
print("{:.5f}".format(best_eval), best_val,obj_class.message)
plt.plot(obj_eval, obj_track)
plt.yscale('log')
plt.xlabel("Objective Function Evaluation")
plt.ylim((10e-4,10e4))
plt.ylabel("Objective Function")
plt.show()
