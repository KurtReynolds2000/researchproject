import numpy as np
import matplotlib.pyplot as plt
import Functions as fun
import Algorithms as alg

bounds = np.asarray([[-10, 10]]*5)

[obj_track, timing, best, best_index] = alg.genetic(fun.Rosenbrock,bounds,75,100,1000)
print("{:.5f}".format(best), best_index)
plt.yscale('log')
plt.xlabel("Time (s)")
plt.ylabel("Objective Function")
plt.plot(timing, obj_track)
plt.show()
