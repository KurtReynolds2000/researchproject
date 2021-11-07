import numpy as np
import time
import matplotlib.pyplot as plt

import Algorithms as alg
import Functions as fun


n = 500
M = 600
t_range = [1, 0.05]
bounds = np.asarray([[-10, 10]]*10)
step = 1
# parameters = (0.4, 1, 0.5)  # for rastrigin
parameters = (0.5, 0.3, 0.5)
error = 1e-5
no_particles = 5000
function = fun.Rosenbrock
dimension = len(bounds)
np.set_printoptions(precision=4)


[best_eval, best_val, obj_track, timing] = alg.sim_annealing(function, n, M, bounds, 1, t_range, step)
print("{:.5f}".format(best_eval), best_val)
plt.plot(timing, obj_track)
plt.yscale('log')
plt.xlabel("Time (s)")
plt.ylabel("Objective Function")

[best_eval, best_val, obj_track, timing] = alg.particle_swarm(function, 400, error, bounds, no_particles, parameters)
print("{:.5f}".format(best_eval), best_val)
plt.plot(timing, obj_track)


[best_eval, best_val, obj_track, timing] = alg.artificial_bee(function, 500, bounds, 600, 10)
print("{:.5f}".format(best_eval), best_val)
plt.plot(timing, obj_track)


[best_eval, best_val, obj_track, timing] = alg.firefly_alg(function, bounds, 200,60,0.2,2,1)
print("{:.5f}".format(best_eval), best_val)
plt.plot(timing, obj_track)

"""
[obj_track, timing, best, best_index] = alg.genetic(function,bounds,30,30,1000)
print("{:.5f}".format(best), best_index)
plt.plot(timing, obj_track)
"""

plt.legend(["Simulated Annealing", "Particle Swarm","Artificial Bee Colony", "Firefly Algorithm", "Genetic Algorithm"])
plt.title(f' Objective Function vs Time (s) for {dimension} Dimensions')
plt.show()
