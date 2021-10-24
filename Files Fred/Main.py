from math import inf
import numpy as np
from numpy import asarray
import time
import matplotlib.pyplot as plt

import Algorithms as alg
import Functions as fun

n = 100
M = 200
t_range = [1,0.05]
bounds = np.asarray([[-5,5],[-5,5]])
step = 0.5
parameters = (0.5,1.1,0.5)
error = 1e-3
no_particles = 400


[best_eval,best_val,obj_track,timing] = alg.sim_annealing(fun.Rastringin,n,M,bounds,1,t_range,step)
print(best_val,best_eval)
plt.plot(timing,obj_track)
plt.yscale('log')
plt.xlabel("Time (s)")
plt.ylabel("Objective Function")

[best_eval,best_val,obj_track,timing] = alg.particle_swarm(fun.Rastringin,30,error,bounds,no_particles,parameters)
print(best_val,best_eval)
plt.plot(timing,obj_track)


[best_eval,best_val,obj_track,timing] = alg.artifical_bee(fun.Rastringin,500,bounds,20,100)
print(best_val,best_eval)
plt.plot(timing,obj_track)
plt.show()