import numpy as np
import time
import math as mt
from ypstruct import structure
import random
from numpy.random import default_rng
from scipy.stats import qmc
import Alg_conditions as AC
from Algorithms import *


def hybrid_genetic(function , bounds, max_iter, max_eval, n_pop=50, n_selection = 3, f_mut = 0.01, tol=1e-80,eta=3, seed = None):
    """
    Steady State Genetic Algorithm with downhill simplex and DC Replacement method
    SBX crossover and BGA mutation
    """
    # Mutation BGA function
    def mutation(x,bounds,f_mut=f_mut):
        alpha = np.linspace(0,1,16)
        if np.random.rand() <= f_mut:
            x += np.random.choice([-1,1])* 0.1*(bounds[:,1]-bounds[:,0])*np.sum([random.choice(alpha)*2**-k for k in range(16)])
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
    best_eval = min(pop_eval)
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
            local_result, local_point,counter = dh_simplex(function,c[loc_index],bounds,100,max_eval)
            obj_counter += counter
            pop[loc_index] = local_point
            pop_eval[loc_index] = local_result
            best_eval, best_coords = local_result, local_point

        # Mutate worst member of population
        mut_index = np.argmax(pop_eval)
        pop[mut_index] = mutation(pop[mut_index],bounds,1)
        pop_eval[mut_index] = function(pop[mut_index])

        # Store solution
        object_track.append(best_eval)
        obj_counter_track.append(obj_counter)
        timing.append(time.time()-time_start)

        i += 1

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


def mayfly_alg(function, bounds, max_iter, max_eval, n_fem=26, n_mal=26, f_mut = 0.01, tol=1e-80, seed=None):

    """
    The mayfly algorithm is a hybrid algorithm derived from PSO, firefly and genetic algorithms
    """

    rng = default_rng(seed)
    # Function for crossover
    def crossover(x1,x2,bounds):
        L = rng.random(len(x1))
        off1 = L * x1 + (1-L) * x2
        off2 = L * x2 + (1-L) * x1
        off1 = np.clip(off1,bounds[:,0],bounds[:,1])
        off2 = np.clip(off2,bounds[:,0],bounds[:,1])
        return off1,off2

    def mutation(x,bounds,f_mut = f_mut):
        y = x.copy()
        size_x = len(bounds)
        nmu = int(np.ceil(f_mut*size_x))
        indeces = random.sample(range(size_x),nmu)
        sigma = 0.1 * (bounds[:,1]-bounds[:,0])
        y[indeces] = x[indeces] + sigma[indeces]*rng.normal(0,1,size=len(indeces))
        y = np.clip(y,bounds[:,0],bounds[:,1])
        return y


    # Initialise class 
    obj_class = AC.opt_solver(len(bounds))
    obj_class.set_seed(seed)
    try:
        int(max_iter + max_eval)
    except:
        raise ValueError("Ensure that arguments provided for max. iterations and function evaluations are integers")

    error = 3 # Used to print termination criteria

    dim = len(bounds)
    object_track = list()
    obj_counter = 0
    obj_counter_track = list()
    timing = list()

    # Initialise input parameters to Mayfly
    parameters = structure()
    parameters.weight = 0.75
    parameters.gdamp = 0.999
    parameters.pcoeff = 1
    parameters.g1coeff = 1.3
    parameters.g2coeff = 1.5
    parameters.beta = 2
    parameters.dance = 1.5
    parameters.rflight = 1.5
    parameters.dance_damp = 0.99
    parameters.rflight_damp = 0.99

    # Mating parameters
    mating = structure()
    mating.n_children = n_mal
    mating.n_male = n_mal
    mating.n_female = n_fem
    mating.n_mutant = int(np.round(0.2*n_mal))
    mating.f_mut = f_mut

    velo_max = 0.1 * (bounds[:,1]-bounds[:,0])
    velo_min = -velo_max

    # Global Parameters
    global_best = structure()
    global_best.Position = np.empty(dim)
    global_best.Cost = float('inf')

    # Initialise problem parameters
    empty_mayfly = structure()
    empty_mayfly.Position = np.ndarray
    empty_mayfly.Cost = np.ndarray
    empty_mayfly.Velo = np.ndarray
    empty_mayfly.Best_Position = np.ndarray
    empty_mayfly.Best_Cost = np.ndarray

    male_mayfly = empty_mayfly.deepcopy()
    male_mayfly = male_mayfly.repeat(n_mal)
    for i in range(mating.n_male):
        male_mayfly[i].Position = bounds[:,0] + rng.random(dim)*(bounds[:,1]-bounds[:,0])
        male_mayfly[i].Cost = function(male_mayfly[i].Position)
        male_mayfly[i].Velo = np.zeros(dim)
        male_mayfly[i].Best_Position = male_mayfly[i].Position
        male_mayfly[i].Best_Cost = male_mayfly[i].Cost
        obj_counter += 1

        if male_mayfly[i].Best_Cost < global_best.Cost:
            global_best.Cost = male_mayfly[i].Best_Cost
            global_best.Position = male_mayfly[i].Best_Position

    female_mayfly = empty_mayfly.deepcopy()
    female_mayfly = female_mayfly.repeat(n_fem)

    for i in range(mating.n_female):
        female_mayfly[i].Position = bounds[:,0] + rng.random(dim)*(bounds[:,1]-bounds[:,0])
        female_mayfly[i].Cost = function(female_mayfly[i].Position)
        female_mayfly[i].Velo = np.zeros(dim)
        obj_counter += 1

    # Start main loop 

    i = 0
    time_start = time.time()

    while i < max_iter:
        # Update female population
        for j in range(mating.n_female):
            uni = rng.uniform(-1,1,size=dim)
            dist = np.sqrt(np.sum(np.square(male_mayfly[j].Position-female_mayfly[j].Position)))
            if female_mayfly[j].Cost > male_mayfly[j].Cost:
                female_mayfly[j].Velo = parameters.weight * female_mayfly[j].Velo + parameters.g2coeff * np.exp(-parameters.beta*dist**2)*(male_mayfly[j].Position-female_mayfly[j].Position)
            else:
                female_mayfly[j].Velo = parameters.weight*female_mayfly[j].Velo + uni * parameters.rflight

            # Apply velocity limits
            female_mayfly[j].Velo = np.clip(female_mayfly[j].Velo, velo_min,velo_max)
            
            # Update position
            female_mayfly[j].Position = female_mayfly[j].Position + female_mayfly[j].Velo
            female_mayfly[j].Position = np.clip(female_mayfly[j].Position,bounds[:,0],bounds[:,1])

            female_mayfly[j].Cost = function(female_mayfly[j].Position)
            obj_counter += 1

        for j in range(mating.n_male):
            # Update males
            rpbest = np.sqrt(np.sum(np.square(male_mayfly[j].Best_Position - male_mayfly[j].Position)))
            rgbest = np.sqrt(np.sum(np.square(global_best.Position - male_mayfly[j].Position)))
            uni = rng.uniform(-1,1,size=dim)

            # Update velocity
            if male_mayfly[j].Cost > global_best.Cost:
                male_mayfly[j].Velo = parameters.weight * male_mayfly[j].Velo + parameters.pcoeff * np.exp(-parameters.beta*rpbest**2) * (male_mayfly[j].Best_Position - male_mayfly[j].Position) + parameters.g1coeff * np.exp(-parameters.beta*rgbest**2) * (global_best.Position-male_mayfly[j].Position)
            else:
                male_mayfly[j].Velo = parameters.weight*male_mayfly[j].Velo + uni * parameters.dance

            # Apply velocity limits
            male_mayfly[j].Velo = np.clip(male_mayfly[j].Velo, velo_min,velo_max)

            # Update position
            male_mayfly[j].Position = male_mayfly[j].Position+male_mayfly[j].Velo
            male_mayfly[j].Position = np.clip(male_mayfly[j].Position,bounds[:,0],bounds[:,1])

            male_mayfly[j].Cost = function(male_mayfly[j].Position)
            obj_counter += 1

            # Update personal Best
            if male_mayfly[j].Cost < male_mayfly[j].Best_Cost:
                male_mayfly[j].Best_Position = male_mayfly[j].Position
                male_mayfly[j].Best_Cost = male_mayfly[j].Cost
                # Update global beset
                if male_mayfly[j].Best_Cost < global_best.Cost:
                    global_best.Cost = male_mayfly[j].Best_Cost
                    global_best.Position = male_mayfly[j].Best_Position

        # Sort both female and male population
        male_mayfly = sorted(male_mayfly,key = lambda x: x.Cost)
        female_mayfly = sorted(female_mayfly,key = lambda x: x.Cost)

        # Mate mayflies
        offspring_may = empty_mayfly.deepcopy()
        offspring_may = offspring_may.repeat(int(2*mating.n_children))
        h = int(mating.n_children)
        for k in range(0,mating.n_children):
            p1 = male_mayfly[k]
            p2 = female_mayfly[k]

            # Crossover operation
            offspring_may[k].Position, offspring_may[h].Position = crossover(p1.Position,p2.Position,bounds)
            offspring_may[k].Cost = function(offspring_may[k].Position)
            obj_counter += 1
            if offspring_may[k].Cost < global_best.Cost:
                global_best.Cost = offspring_may[k].Cost
                global_best.Position = offspring_may[k].Position
            offspring_may[h].Cost = function(offspring_may[h].Position)
            obj_counter += 1
            if offspring_may[h].Cost < global_best.Cost:
                global_best.Cost = offspring_may[h].Cost
                global_best.Position = offspring_may[h].Position
            
            offspring_may[k].Best_Position = offspring_may[k].Position
            offspring_may[k].Best_Cost = offspring_may[k].Cost
            offspring_may[k].Velo = np.zeros(dim)
            offspring_may[h].Best_Position = offspring_may[h].Position
            offspring_may[h].Best_Cost = offspring_may[h].Cost
            offspring_may[h].Velo = np.zeros(dim)
            h += 1

        mutated_may = empty_mayfly.deepcopy()
        mutated_may = mutated_may.repeat(mating.n_mutant)
        for j in range(mating.n_mutant):
            # Select random offspring to be mutated
            rand_idx = np.random.randint(0,mating.n_children)
            p = offspring_may[rand_idx]
            mutated_may[j].Position = mutation(p.Position,bounds)
            mutated_may[j].Cost = function(mutated_may[j].Position)
            obj_counter += 1
            if mutated_may[j].Cost < global_best.Cost:
                global_best.Cost = mutated_may[j].Cost
                global_best.Position = mutated_may[j].Position
            
            mutated_may[j].Best_Position = mutated_may[j].Position
            mutated_may[j].Best_Cost = mutated_may[j].Cost
            mutated_may[j].Velo = np.zeros(dim)
        
        # Merge populations
        split1 = round(mating.n_children/2)
        split2 = round(mating.n_mutant/2)
        new_mayflies = offspring_may[:split1] + mutated_may[:split2]
        male_mayfly = male_mayfly + new_mayflies
        male_mayfly = sorted(male_mayfly, key= lambda x: x.Cost)
        male_mayfly = male_mayfly[:mating.n_male]
        new_mayflies = offspring_may[split1:] + mutated_may[:split2]
        female_mayfly = female_mayfly + new_mayflies
        female_mayfly = sorted(female_mayfly, key= lambda x: x.Cost)
        female_mayfly = female_mayfly[:mating.n_female]

        # Apply damping parameters
        parameters.dance *= parameters.dance_damp
        parameters.rflight *= parameters.rflight_damp
        parameters.weight *= parameters.gdamp

        # Store solution
        object_track.append(global_best.Cost)
        obj_counter_track.append(obj_counter)
        timing.append(time.time()-time_start)

        i += 1

        # Check for convergence or if max feval has been exceeded
        pop = np.array([male_mayfly[k].Cost for k in range(mating.n_male)])
        if AC.opt_converge(pop,tol):
            error = 1
            break
        elif max_eval <= obj_counter:
            error = 2
            break

    obj_class.xarray = global_best.Position
    obj_class.set_message(error)
    obj_class.eval = global_best.Cost
    obj_class.n_iter = i
    obj_class.n_feval = obj_counter
    
    return (obj_class, object_track, obj_counter_track, timing)


def hybrid_differential(function , bounds, max_iter, max_eval, n_pop=100, repl = 0.85, f_mut = 0.01, tol=1e-80, seed = None):
    """
    Differential evolution with Downhill Simplex and BGA mutation
    """
    # Mutation BGA function
    def mutation(x,bounds,f_mut=f_mut):
        alpha = np.linspace(0,1,16)
        if np.random.rand() <= f_mut:
            x += np.random.choice([-1,1])* 0.1*(bounds[:,1]-bounds[:,0])*np.sum([random.choice(alpha)*2**-k for k in range(16)])
            x = np.clip(x,bounds[:,0], bounds[:,1])
            return x
        else:
            return x
    
    # Replacement helper function to handle SSGA replacement
    def crossover(bounds,pop,uni):
        new_array = np.array(list(range(len(pop))))
        np.random.shuffle(new_array)
        no_1,no_2= new_array[:2]
        no_3,no_4 = new_array[2:4]
        off1 = pop[0] + uni*(pop[no_1]-pop[no_2])
        off2 = pop[0] + uni* (pop[no_3]-pop[no_4])
        off1,off2 = np.clip(off1,bounds[:,0],bounds[:,1]),np.clip(off2,bounds[:,0],bounds[:,1])
        return off1,off2
        
        
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
    sampler = qmc.Halton(d=dim,scramble=False)
    sample = sampler.random(n=n_pop)
    pop = np.copy(sample)
    pop = qmc.scale(pop, bounds[:,0], bounds[:,1])
    pop_eval = np.apply_along_axis(function,1,pop)
    time_start = time.time()

    i = 0

    # Start SSGA algorithm
    while i < max_iter:
        sort_order = np.argsort(pop_eval)
        pop_eval = pop_eval[sort_order]
        pop = pop[sort_order]
        cross_size = np.random.uniform(0.5,1)
        best_eval_last = best_eval

        for j in range(int(n_pop)):
            idx_p1 = j
            
            idx_p2 = np.random.randint(0,n_pop)
            while idx_p2 == idx_p1:
                idx_p2 = np.random.randint(0,n_pop)

            p1,p2= pop[idx_p1],pop[idx_p2]
            p_eval1,p_eval2 = pop_eval[idx_p1],pop_eval[idx_p2]
            
            # Crossover
            c1,c2 = crossover(bounds,pop,cross_size)

            # Perform mutation on children
            c1 = mutation(c1,bounds)
            c2 = mutation(c2,bounds)
            
            # Recombination
            mu = np.random.uniform(0,1,dim)
            child1 = np.array([c1[k] if mu[k]<repl else p1[k] for k in range(dim)])
            child2 = np.array([c2[k] if mu[k]<repl else p2[k] for k in range(dim)])
            c = [child1,child2]

            # Evaluate children
            eval_c1,eval_c2 = function(child1),function(child2)
            eval_c = [eval_c1,eval_c2]
            loc_index = np.argmin(eval_c)

            obj_counter += 2

            # Select parent to replace
            if eval_c1 < p_eval1:
                pop[idx_p1] = child1
                pop_eval[idx_p1] = eval_c1

            if eval_c2 < p_eval2:
                pop[idx_p2] = child2
                pop_eval[idx_p2] = eval_c2
            
            if eval_c[loc_index] < best_eval:
                best_coords= c[loc_index]
                best_eval = eval_c[loc_index]
                best_index = loc_index
            
        # Perform local search on child if necessary
        criterium = abs((best_eval-best_eval_last)/best_eval)
        if  criterium >= 1e-2:
            local_result, local_point,counter = dh_simplex(function,best_coords,bounds,50)
            obj_counter += counter
            if local_result < best_eval:
                best_eval, best_coords = local_result, local_point

        # Store solution
        object_track.append(best_eval)
        obj_counter_track.append(obj_counter)
        timing.append(time.time()-time_start)

        i += 1

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