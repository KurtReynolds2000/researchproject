import numpy as np
from numpy.random import default_rng
import Functions as fun
import Algorithms as alg
function = fun.Rosenbrock
bounds = np.asarray([[-10, 10]]*2)

best, best_source, obj_track, timing = alg.firefly_alg(
    function, bounds, 100, 300)

print(best, best_source)
