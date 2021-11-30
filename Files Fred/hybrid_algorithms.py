import numpy as np
import time
import math as mt
from ypstruct import structure
import random
import Alg_conditions as AC
from Algorithms import *


def hybrid_genetic(function , bounds, max_iter, max_eval, n_pop=50, n_selection = 3, f_mut = 0.01, tol=1e-80,eta=1, seed = None):
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


def mayfly_alg(function, bounds, max_iter, max_eval, n_fem=20, n_mal=20, f_mut = 0.01, tol=1e-80, seed=None):
    """
    The mayfly algorithm is a hybrid algorithm derived from PSO, firefly and genetic algorithms
    """

    # Function for crossover
    def crossover(x1,x2,bounds):
        L = np.random.random(len(x1))
        off1 = L * x1 + (1-L) * x2
        off2 = L * x2 + (1-L) * x1
        off1 = np.clip(off1,bounds[:,0],bounds[:,1])
        off2 = np.clip(off2,bounds[:,0],bounds[:,1])
        return off1,off2

    def mutation(x,bounds,f_mut = f_mut):
        y = x
        size_x = len(bounds)
        nmu = int(np.ceil(f_mut*size_x))
        indeces = random.sample(range(size_x),nmu)
        sigma = 0.1 * (bounds[:,1]-bounds[:,0])
        y[indeces] = x[indeces] + sigma[indeces]*np.random.normal(0,1,size=len(indeces))
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
    parameters.weight = 0.8
    parameters.gdamp = 1
    parameters.pcoeff = 1
    parameters.g1coeff = 1.5
    parameters.g2coeff = 1.5
    parameters.beta = 2
    parameters.dance = 0.1
    parameters.rflight = 0.1
    parameters.dance_damp = 1
    parameters.rflight_damp = 1

    # Mating parameters
    mating = structure()
    mating.n_children = n_mal
    mating.n_male = n_mal
    mating.n_female = n_fem
    mating.n_mutant = int(np.round(0.05*n_mal))
    mating.f_mut = f_mut

    velo_max = 0.1 * (bounds[:,1]-bounds[:,0])
    velo_min = -velo_max

    # Global Parameters
    global_best = structure()
    global_best.Position = np.empty(dim)
    global_best.Cost = float('inf')

    # Initialise problem parameters
    empty_mayfly = structure()
    empty_mayfly.Position = []
    empty_mayfly.Cost = []
    empty_mayfly.Velo = []
    empty_mayfly.Best_Position = []
    empty_mayfly.Best_Cost = []

    male_mayfly = empty_mayfly.deepcopy()
    male_mayfly = male_mayfly.repeat(n_mal)
    for i in range(mating.n_male):
        male_mayfly[i].Position = bounds[:,0] + np.random.rand(dim)*(bounds[:,1]-bounds[:,0])
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
        female_mayfly[i].Position = bounds[:,0] + np.random.rand(dim)*(bounds[:,1]-bounds[:,0])
        female_mayfly[i].Cost = function(female_mayfly[i].Position)
        female_mayfly[i].Velo = np.zeros(dim)
        obj_counter += 1

    # Start main loop 

    i = 0
    time_start = time.time()

    while i < max_iter:
        # Update female population
        for j in range(mating.n_female):
            uni = np.random.uniform(-1,1,size=dim)
            dist = (male_mayfly[j].Position-female_mayfly[j].Position)
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
            rpbest = (male_mayfly[j].Best_Position - male_mayfly[j].Position)
            rgbest = (global_best.Position - male_mayfly[j].Position)
            uni = np.random.uniform(-1,1,size=dim)

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
        offspring_may = offspring_may.repeat(mating.n_children)
        k = 0
        for j in range(0,round(mating.n_children/2)):
            p1 = male_mayfly[j]
            p2 = female_mayfly[j]

            # Crossover operation
            offspring_may[k].Position, offspring_may[k+1].Position = crossover(p1.Position,p2.Position,bounds)
            offspring_may[k].Cost = function(offspring_may[k].Position)
            obj_counter += 1
            if offspring_may[k].Cost < global_best.Cost:
                global_best.Cost = offspring_may[k].Cost
                global_best.Position = offspring_may[k].Position
            offspring_may[k+1].Cost = function(offspring_may[k+1].Position)
            obj_counter += 1
            if offspring_may[k+1].Cost < global_best.Cost:
                global_best.Cost = offspring_may[k+1].Cost
                global_best.Position = offspring_may[k+1].Position
            
            offspring_may[k].Best_Position = offspring_may[k].Position
            offspring_may[k].Best_Cost = offspring_may[k].Cost
            offspring_may[k].Velo = np.zeros(dim)
            offspring_may[k+1].Best_Position = offspring_may[k+1].Position
            offspring_may[k+1].Best_Cost = offspring_may[k+1].Cost
            offspring_may[k+1].Velo = np.zeros(dim)
            k += 2
        mutated_may = empty_mayfly.deepcopy()
        mutated_may = mutated_may.repeat(mating.n_mutant)
        for j in range(mating.n_mutant):
            # Select random offspring to be mutated
            rand_idx = np.random.randint(0,mating.n_children-1)
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
        offspring_may = offspring_may + mutated_may
        split = round(mating.n_children/2)
        new_mayflies = offspring_may[:split+1]
        male_mayfly = male_mayfly + new_mayflies
        male_mayfly = sorted(male_mayfly, key= lambda x: x.Cost)
        male_mayfly = male_mayfly[:mating.n_male]
        new_mayflies = offspring_may[split+1:]
        female_mayfly = female_mayfly + new_mayflies
        female_mayfly = sorted(female_mayfly, key= lambda x: x.Cost)
        female_mayfly = female_mayfly[:mating.n_female]

        # Apply damping parameters
        parameters.dance *= parameters.dance_damp
        parameters.rflight *= parameters.rflight_damp
        parameters.pcoeff *= parameters.gdamp

        # Store solution
        object_track.append(global_best.Cost)
        obj_counter_track.append(obj_counter)
        timing.append(time.time()-time_start)

        i += 1
        print(global_best.Cost)

        # Check for convergence or if max feval has been exceeded
        pop = np.array([list(male_mayfly[k].Position) for k in range(mating.n_male)])
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
    


