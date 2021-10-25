import numpy as np
import time
import matplotlib.pyplot as plt

import Algorithms as alg
import Functions as fun

n = 600
M = 300
t_range = [1, 0.05]
bounds = np.asarray([[-5, 5]]*50)
step = 0.5
parameters = (0.4, 1, 0.5)
error = 1e-5
no_particles = 700


[best_eval, best_val, obj_track, timing] = alg.sim_annealing(
    fun.Rastringin, n, M, bounds, 1, t_range, step)
print(best_val, best_eval)
plt.plot(timing, obj_track)
plt.yscale('log')
plt.xlabel("Time (s)")
plt.ylabel("Objective Function")

[best_eval, best_val, obj_track, timing] = alg.particle_swarm(
    fun.Rastringin, 800, error, bounds, no_particles, parameters)
print(best_val, best_eval)
plt.plot(timing, obj_track)


[best_eval, best_val, obj_track, timing] = alg.artifical_bee(
    fun.Rastringin, 1200, bounds, 10, 160)
print(best_val, best_eval)
plt.plot(timing, obj_track)
plt.legend(["Simulated Annealing", "Particle Swarm", "Artifical Bee Colony"])
plt.show()
