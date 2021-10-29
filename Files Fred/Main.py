import numpy as np
import time
import matplotlib.pyplot as plt

import Algorithms as alg
import Functions as fun


n = 500
M = 400
t_range = [1, 0.05]
bounds = np.asarray([[-5.12, 5.12]]*10)
step = 0.5
parameters = (0.4, 1, 0.5)  # for rastringin
# parameters = (0.5, 0.9, 0.5)
error = 1e-5
no_particles = 1000
function = fun.Rastringin
dimension = len(bounds)
np.set_printoptions(precision=4)


[best_eval, best_val, obj_track, timing] = alg.sim_annealing(function, n, M, bounds, 1, t_range, step)
print("{:.5f}".format(best_eval), best_val)
plt.plot(timing, obj_track)
plt.yscale('log')
plt.xlabel("Time (s)")
plt.ylabel("Objective Function")

[best_eval, best_val, obj_track, timing] = alg.particle_swarm(function, 700, error, bounds, no_particles, parameters)
print("{:.5f}".format(best_eval), best_val)
plt.plot(timing, obj_track)


[best_eval, best_val, obj_track, timing] = alg.artificial_bee(function, 900, bounds, 20, 600)
print("{:.5f}".format(best_eval), best_val)
plt.plot(timing, obj_track)


[best_eval, best_val, obj_track, timing] = alg.firefly_alg(function, bounds, 60000,20,2)
print("{:.5f}".format(best_eval), best_val)
plt.plot(timing, obj_track)

plt.legend(["Simulated Annealing", "Particle Swarm","Artificial Bee Colony", "Firefly Algorithm"])
plt.title(f' Objective Function vs Time (s) for {dimension} Dimensions')
plt.show()
