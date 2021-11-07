import numpy as np
import matplotlib.pyplot as plt
import Functions as fun
import Algorithms as alg


bounds = np.asarray([[-10, 10]]*5)


[best_eval, best_val, obj_track, timing] = alg.artificial_bee(fun.Rosenbrock,500,bounds,60,10)
print("{:.5f}".format(best_eval), best_val)
plt.yscale('log')
plt.xlabel("Time (s)")
plt.ylabel("Objective Function")
plt.plot(timing, obj_track)
plt.show()