import numpy as np
import time
import matplotlib.pyplot as plt

import Algorithms as alg
import Functions as fun


n = 200
M = 300
t_range = [1, 0.05]
bounds = np.asarray([[-10, 10]]*2)
step = 0.5
# parameters = (0.4, 1, 0.5)  # for rastrigin
parameters = (0.5, 0.3, 0.5)
error = 1e-5
no_particles = 200
function = fun.Rosenbrock
dimension = len(bounds)
np.set_printoptions(precision=4)


[best_eval, best_val, obj_track, obj_eval, timing] = alg.sim_annealing(function, n, M, bounds, 1, t_range, step)
print("{:.5f}".format(best_eval), best_val)
plt.plot(obj_eval, obj_track)
plt.yscale('log')
plt.xlabel("Time (s)")
plt.xlim((0,200))
plt.ylabel("Objective Function")

[best_eval, best_val, obj_track, obj_eval, timing] = alg.particle_swarm(function, 100, error, bounds, no_particles, parameters)
print("{:.5f}".format(best_eval), best_val)
plt.plot(obj_eval, obj_track)


[best_eval, best_val, obj_track, obj_eval, timing] = alg.artificial_bee(function, bounds, 500, 100, 10)
print("{:.5f}".format(best_eval), best_val)
plt.plot(obj_eval, obj_track)


[best_eval, best_val, obj_track, obj_eval, timing] = alg.firefly_alg(function, bounds, 300,10,0.2,2,1)
print("{:.5f}".format(best_eval), best_val)
plt.plot(obj_eval, obj_track)

[best_eval, best_val, obj_track, obj_eval, timing] = alg.diff_evolution(function, bounds, 30,200)
print("{:.5f}".format(best_eval), best_val)
plt.plot(obj_eval, obj_track)

[best_eval, best_val, obj_track, obj_eval, timing] = alg.dh_simplex(function, bounds, 500)
print("{:.5f}".format(best_eval), best_val)
plt.plot(obj_eval, obj_track)

plt.legend(["Simulated Annealing", "Particle Swarm","Artificial Bee Colony", "Firefly Algorithm", "Differential Evolution", "Downhill Simplex"])
plt.title(f' Objective Function vs Time (s) for {dimension} Dimensions')
plt.show()
