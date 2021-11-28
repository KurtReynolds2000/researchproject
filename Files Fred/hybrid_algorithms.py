import numpy as np
import time
import math as mt
import random
import Alg_conditions as AC
from Algorithms import dh_simplex


def hybrid_genetic(function , bounds, max_iter, max_eval, n_pop=50, n_selection = 2, eta =  3, f_mut = 0.1, tol=1e-80, seed = None):
    """
    Steady State Genetic Algorithm with downhill simplex and CD/RW Replacement method
    SBX crossover and BGA mutation
    """
    # Mutation BGA function
    def mutation(x,bounds):
        alpha = np.linspace(0,1,16)
        store = np.random.rand(*x.shape) <= f_mut
        indeces = np.argwhere(store)
        x[indeces] += np.random.choice([-1,1])* random.uniform(0,0.5)*np.sum([random.choice(alpha)*2**-k for k in range(16)])
        x = np.clip(x,bounds[:,0], bounds[:,1]) # Ensuring that children are not out of bounds
        return x
    
    # Replacement helper function to handle SSGA replacement
    def replacement(child,child_eval,pop,pop_eval,min_cont):
        index = np.argmin(child_eval>pop_eval)
        test_cont = min_cont[index:]
        # Check if child is at least better than worst member of population
        if index==0 and child_eval>max(pop_eval):
            return None,None
        
        # Finding chromosome to replace
        min_idx = np.argmin(test_cont)
        pop_idx = min_idx + index
        child_cont = np.array([float('inf')]*len(pop))
        for count,coords in enumerate(pop):
            if count == pop_idx:
                pass
            child_cont[count] = np.sum(np.square(coords-child))**.5
        
        if min(child_cont) > test_cont[min_idx]:
            return pop_idx, min(child_cont)
        else:
            return len(pop)-1, min(child_cont) # Replace the worst solution in population

    # Calculating the diversity of each chromosome
    def contribution_diversity(pop_coords):
        min_distance = np.array([float('inf')]*len(pop_coords))
        for i,coords in enumerate(pop_coords):
            for j,x in enumerate(pop_coords):
                if i == j:
                    pass
                else:
                    store = np.sum(np.square(x-coords))**0.5
                    min_distance[i] = min(store,min_distance[i])
        return min_distance

    
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
    pop_min_cont = contribution_diversity(pop)
    time_start = time.time()

    i = 0

    # Start SSGA algorithm
    while i < max_iter:

        sort_order = np.argsort(pop_eval)
        pop = pop[sort_order]
        pop_eval = pop_eval[sort_order]
        pop_min_cont = pop_min_cont[sort_order]
        idx_p1 = np.random.randint(0,n_pop)
        
        for idx1 in random.sample(range(0,n_pop),n_selection):
            if pop_eval[idx1] < pop_eval[idx_p1]:
                idx_p1 = idx1
        
        idx_p2 = np.random.randint(0,n_pop)
        while idx_p2 == idx_p1:
            idx_p2 = np.random.randint(0,n_pop)

        p1,p2= pop[idx_p1],pop[idx_p2]
        
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
        arg = np.argmin(eval_c)
        c,eval_c = c[arg],eval_c[arg]

        obj_counter += 2

        # Select parent to replace
        index,child_cont = replacement(c,eval_c,pop,pop_eval,pop_min_cont)
        if index == None:
            pass
        else:
            pop[index] = c
            pop_eval[index] = eval_c
            pop_min_cont[index] = child_cont
            for count,mem in enumerate(pop):
                if count == index:
                    pass
                else:
                    store = np.sum(np.square(c-mem))**0.5
                    pop_min_cont[count] = min(pop_min_cont[count],store)

            # Perform local search on child if necessary
            if  eval_c <= pop_eval[0]:
                local_result, local_point = dh_simplex(function,c,bounds,150,max_eval,seed=seed)
                pop[index] = local_point
                pop_eval[index] = local_result
                print(eval_c,local_result)
                best_eval, best_coords = local_result, local_point
    
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