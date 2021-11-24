import numpy as np
import matplotlib.pyplot as plt

import Algorithms as alg
import Functions as fun


bounds = np.asarray([[-5, 5]]*16)
# parameters = (0.4, 1, 0.5)  # for rastrigin
function = fun.Ackley
dimension = len(bounds)
np.set_printoptions(precision=4)


[obj_class, obj_track, obj_eval, timing] = alg.sim_annealing(function, bounds, 100000, 100000,step=0.5)
best_eval = obj_class.eval
best_val = obj_class.xarray
print("{:.5f}".format(best_eval), best_val)
plt.plot(obj_eval, obj_track)
plt.yscale('log')
plt.xlabel("Objective Function Evaluation")
plt.xlim((0,100000))
plt.ylim((10e-4,10e2))
plt.ylabel("Objective Function")

[obj_class, obj_track, obj_eval, timing] = alg.particle_swarm(function, bounds, 100000, 100000)
best_eval = obj_class.eval
best_val = obj_class.xarray
print("{:.5f}".format(best_eval), best_val)
plt.plot(obj_eval, obj_track)


[obj_class, obj_track, obj_eval, timing] = alg.artificial_bee(function, bounds, 100000, 100000,n_bees=500,limit=10)
best_eval = obj_class.eval
best_val = obj_class.xarray
print("{:.5f}".format(best_eval), best_val)
plt.plot(obj_eval, obj_track)


[obj_class, obj_track, obj_eval, timing] = alg.firefly_alg(function, bounds, 100000, 100000)
best_eval = obj_class.eval
best_val = obj_class.xarray
print("{:.5f}".format(best_eval), best_val)
plt.plot(obj_eval, obj_track)

[obj_class, obj_track, obj_eval, timing] = alg.diff_evolution(function, bounds, 100000, 100000)
best_eval = obj_class.eval
best_val = obj_class.xarray
print("{:.5f}".format(best_eval), best_val)
plt.plot(obj_eval, obj_track)

[obj_class, obj_track, obj_eval, timing] = alg.dh_simplex(function, bounds, 100000, 100000)
best_eval = obj_class.eval
best_val = obj_class.xarray
print("{:.5f}".format(best_eval), best_val)
plt.plot(obj_eval, obj_track)

plt.legend(["SA", "PSO","ABC", "FA", "DE", "DSMPLEX"])
plt.title(f' Objective Function vs Time (s) for {dimension} Dimensions')
plt.show()
