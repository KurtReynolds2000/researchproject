import numpy as np
import time
import math as mt
import random
import Alg_conditions as AC
from Algorithms import *


def hybrid_genetic(function , bounds, max_iter, max_eval, n_pop=50, n_selection = 3, f_mut = 0.1, tol=1e-80,eta=5, seed = None):
    """
    Steady State Genetic Algorithm with downhill simplex and DC Replacement method
    SBX crossover and BGA mutation
    """
    # Mutation BGA function
    def mutation(x,bounds,f_mut=f_mut):
        alpha = np.linspace(0,1,16)
        if np.random.rand() <= f_mut:
            x += np.random.choice([-1,1])* 0.05*(bounds[:,1]-bounds[:,0])*np.sum([random.choice(alpha)*2**-k for k in range(16)])
            x = np.clip(x,bounds[:,0], bounds[:,1])
            return x
        else:
            return x
    
    # Replacement helper function to handle SSGA replacement
    def euclidean_distance(children,parents):
        distances = np.empty((2,2))
        for i,child in enumerate(children):
            for j,parent in enumerate(parents):
                distances[i,j]=np.sum(np.square(parent-child))**.5
        
        return distances
        
        
    # Initialise class 
    obj_class = AC.opt_solver(len(bounds))
    obj_class.set_seed(seed)
    try:
        int(max_iter + max_eval)
    except:
        raise ValueError("Ensure that arguments provided for max. iterations and function evaluations are integers")

    error = 3 # Used to print termination criteria

    # Storing solutions
    dim = len(bounds)
    best_coords = np.empty(dim)
    best_eval = float('inf')
    object_track = list()
    obj_counter = 0
    obj_counter_track = list()
    timing = list()
    

    # Initialise data
    pop = np.array([bounds[:, 0] + np.random.rand(len(bounds))* (bounds[:, 1] - bounds[:, 0]) for _ in range(n_pop)])
    pop_eval = np.apply_along_axis(function,1,pop)
    time_start = time.time()

    i = 0

    # Start SSGA algorithm
    while i < max_iter:
        idx_p1 = np.random.randint(0,n_pop)
        
        for idx1 in random.sample(range(0,n_pop),n_selection):
            if pop_eval[idx1] < pop_eval[idx_p1]:
                idx_p1 = idx1
        
        idx_p2 = np.random.randint(0,n_pop)
        while idx_p2 == idx_p1:
            idx_p2 = np.random.randint(0,n_pop)

        p1,p2= pop[idx_p1],pop[idx_p2]
        p_eval1,p_eval2 = pop_eval[idx_p1],pop_eval[idx_p2]
        
        # SBX operator
        beta = np.array([(2*mu)**(1/(eta+1)) if mu < 0.5 else (1/(2*(1-mu)))**(1/(eta+1)) for mu in np.random.rand(dim)])
        c1 = 0.5*((p1+p2)-beta*abs(p2-p1))
        c2 = 0.5*((p1+p2)+beta*abs(p2-p1))

        # Perform mutation on children
        c1 = mutation(c1,bounds)
        c2 = mutation(c2,bounds)
        c = [c1,c2]

        # Evaluate children
        eval_c1,eval_c2 = function(c1),function(c2)
        eval_c = [eval_c1,eval_c2]
        loc_index = np.argmin(eval_c)

        obj_counter += 2

        # Select parent to replace
        distances = euclidean_distance(c,[p1,p2])
        if np.sum([distances[0,0],distances[1,1]]) <= np.sum([distances[0,1],distances[1,0]]):
            if eval_c1 < p_eval1:
                pop[idx_p1] = c1
                pop_eval[idx_p1] = eval_c1

            if eval_c2 < p_eval2:
                pop[idx_p2] = c2
                pop_eval[idx_p2] = eval_c2
        else:
            if eval_c1 < p_eval2:
                pop[idx_p2] = c1
                pop_eval[idx_p2] = eval_c1
            
            if eval_c2 < p_eval1:
                pop[idx_p1] = c2
                pop_eval[idx_p1] = eval_c2

        # Perform local search on child if necessary
        if  eval_c[loc_index] <= min(pop_eval):
            local_result, local_point,counter = dh_simplex(function,c[loc_index],bounds,100,max_eval,seed=seed)
            obj_counter += counter
            pop[loc_index] = local_point
            pop_eval[loc_index] = local_result
            if local_result < best_eval:
                best_eval, best_coords = local_result, local_point
                print(pop_eval)

        # Mutate worst member of population
        mut_index = np.argmax(pop_eval)
        pop[mut_index] = mutation(pop[mut_index],bounds,1)
        pop_eval[mut_index] = function(pop[mut_index])

        # Store solution
        object_track.append(best_eval)
        obj_counter_track.append(obj_counter)
        timing.append(time.time()-time_start)

        i += 1
        print(best_eval)

        # Check for convergence or if max feval has been exceeded
        if AC.opt_converge(pop,tol):
            error = 1
            break
        elif max_eval <= obj_counter:
            error = 2
            break

    obj_class.xarray = best_coords
    obj_class.set_message(error)
    obj_class.eval = best_eval
    obj_class.n_iter = i
    obj_class.n_feval = obj_counter

    return (obj_class, object_track, obj_counter_track, timing)