import numpy as np
import time
import matplotlib.pyplot as plt
import statistics
import Algorithms as alg
import Functions as fun


bounds = np.asarray([[-5,5]]*15)
dimension = len(bounds)
de_evals = 1000

runs = 10
de_timings, de_objs = np.ndarray([runs,20]), np.ndarray([runs,20])



for run in range(runs):
    [obj_class, obj_track, obj_eval, timing] = alg.diff_evolution(fun.Goldstein, bounds, 2000, de_evals,50,0.9,0.8,1e-20)
    de_objs[run,:] = obj_track
    de_timings[run,:] = timing
    print("running")


de_std = np.ndarray([len(de_objs[0,:])])
de_timing = np.ndarray([len(de_objs[0,:])])
de_mean = np.ndarray([len(de_objs[0,:])])

de_mean = np.apply_along_axis(np.mean,0,de_objs)
de_timing = np.apply_along_axis(np.mean,0,de_timings)
de_std = np.apply_along_axis(np.std,0,de_objs)

plt.plot(de_timing,de_mean)
#plt.plot(sa_mean,sa_timing)
plt.yscale('log')
plt.xlabel("Time")
#plt.xlim((0,10))
plt.fill_between(de_timing, de_mean-de_std, de_mean+de_std, alpha=0.2)
#plt.fill_between(sa_mean, -sa_std, sa_std, alpha=0.2)
plt.ylim((0.8, 10e2))
plt.ylabel("Objective Function")
plt.legend(["DE","SA"])
plt.title(f' Objective Function vs Time for {dimension} Dimensions')
plt.show()