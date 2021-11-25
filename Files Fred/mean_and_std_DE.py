import numpy as np
import time
import matplotlib.pyplot as plt
import statistics
import Algorithms as alg
import Functions as fun


n = 200
M = 300
t_range = [1, 0.05]
bounds = np.asarray([[-10, 10]]*2)
step = 0.05
# parameters = (0.4, 1, 0.5)  # for rastrigin
parameters = (0.5, 0.3, 0.5)
error = 1e-5
no_particles = 200
function = fun.Rosenbrock
dimension = len(bounds)
np.set_printoptions(precision=4)
de_evals = 50000
runs = 10
de_timings, de_objs = np.ndarray([runs,de_evals]), np.ndarray([runs,de_evals])
sa_timings, sa_objs = np.ndarray([runs,n*M]), np.ndarray([runs,n*M])

for run in range(0,runs):
    [best_eval, best_val, obj_track, obj_eval, timing] = alg.diff_evolution(function, bounds, 100, de_evals)
    de_objs[run,:] = obj_track
    de_timings[run,:] = timing

    [best_eval, best_val, obj_track, obj_eval, timing] = alg.sim_annealing(function, n, M, bounds, 1, t_range, step)
    sa_objs[run,:] = obj_track
    sa_timings[run,:] = timing

de_std = np.ndarray([len(de_objs[0,:])])
de_timing = np.ndarray([len(de_objs[0,:])])
de_mean = np.ndarray([len(de_objs[0,:])])
sa_std = np.ndarray([len(sa_objs[0,:])])
sa_timing = np.ndarray([len(sa_objs[0,:])])
sa_mean = np.ndarray([len(sa_objs[0,:])])

for x in range(1,len(de_objs[0,:])+1):
    de_mean[x-1] = np.mean(de_objs[:,x-1])
    de_timing[x-1] = np.mean(de_timings[:,x-1])
    de_std[x-1] = np.std(de_objs[:,x-1])

for x in range(1,len(sa_objs[0,:])+1):
    sa_mean[x-1] = np.mean(sa_objs[:,x-1])
    sa_timing[x-1] = np.mean(sa_timings[:,x-1])
    sa_std[x-1] = np.std(sa_objs[:,x-1])

plt.plot(de_mean,de_timing)
plt.plot(sa_mean,sa_timing)
plt.yscale('log')
plt.xlabel("Time")
plt.xlim((0,10))
plt.fill_between(de_mean, -de_std, de_std, alpha=0.2)
plt.fill_between(sa_mean, -sa_std, sa_std, alpha=0.2)
plt.ylim((10e-15, 10e3))
plt.ylabel("Objective Function")
plt.legend(["DE","SA"])
plt.title(f' Objective Function vs Time for {dimension} Dimensions')
plt.show()