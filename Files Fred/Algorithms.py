import numpy as np
import time
from math import exp
import math as mt
import random


def sim_annealing(function, n_iter, M, bounds, t_schedule, t_range, step):
    """
    This is the algorithm for performing simulated annealing including several cooling schedules
    """

    # Define certain cooling schedules
    def cool_straight(t_i, t_f, M, i):
        return (t_f-t_i)/(M-1) * (i-1) + t_i

    def cool_geo(t_i, t_f, M, i):
        return t_i*(t_f/t_i)**((i-1)/(M-1))

    def cool_rec(t_i, t_f, M, i):
        return t_f*t_i*(M-1)/((t_f*M-t_i)+(t_i-t_f)*i)

    schedules = {1: cool_straight, 2: cool_geo, 3: cool_rec}
    t_select = schedules[t_schedule]

    # Initialise variables

    temp = sum(t_range)/len(t_range)
    best_val = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    best_eval = function(best_val)
    curr_val, curr_eval = best_val, best_eval
    t_init, t_final = t_range[0], t_range[1]
    obj_track = list()
    obj_counter = 0
    obj_counter_track = list()
    timing = list()
    t_start = time.time()

    for j in range(M):    # Outer loop iteration

        for i in range(n_iter):
            config_val = curr_val + np.random.randn(len(bounds)) * step
            config_eval = function(config_val)
            obj_counter += 1
            diff = config_eval - curr_eval

            if config_eval < best_eval:
                best_eval, best_val = config_eval, config_val
                curr_eval, curr_val = config_eval, config_val
            else:

                if np.random.rand() < exp(-diff / temp):
                    curr_eval, curr_val = config_eval, config_val
                else:
                    pass

            timing.append(time.time()-t_start)
            obj_track.append(best_eval)
            obj_counter_track.append(obj_counter)

        temp = t_select(t_init, t_final, M, j)

    return [best_eval, best_val, obj_track, obj_counter_track, timing]


def particle_swarm(function, n_iter, error, bounds, n_particles, parameter):
    """
    This is an algorithm for the particle swarm optimisation
    """
    # Initialise parameters for algorithm
    w, c_1, c_2 = parameter[0], parameter[1], parameter[2]
    # Initialise individual particle properties
    part_position = np.zeros((n_particles, len(bounds)), dtype=np.float32)
    part_velocity = part_position
    obj_track = list()
    obj_counter = 0
    obj_counter_track = list()
    timing = list()

    for particle in range(n_particles):
        part_position[particle] = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    part_bposition = part_position
    part_beval = np.array([float('inf') for _ in range(n_particles)])

    group_position = np.array([float('inf') for _ in range(len(bounds))])
    group_eval = float('inf')
    i = 0
    t_start = time.time()

    while i < n_iter:

        for k in range(n_particles):
            eval_candidate = function(part_position[k])
            obj_counter += 1

            if eval_candidate < part_beval[k]:
                part_beval[k] = eval_candidate
                part_bposition[k] = part_position[k]

            if eval_candidate < group_eval:
                group_eval = eval_candidate
                group_position = part_position[k]

            part_velocity[k] = w * part_velocity[k] + c_1 * np.random.random() * (part_bposition[k]-part_position[k]) + c_2 * np.random.random() * (group_position - part_position[k])
            part_position[k] += part_velocity[k]

        if sum(abs(group_eval - part_beval)) < error:
            break

        i = i + 1
        timing.append(time.time()-t_start)
        obj_track.append(group_eval)
        obj_counter_track.append(obj_counter)

    return [group_eval, group_position, obj_track, obj_counter_track, timing]


def artificial_bee(function, bounds, n_iter, n_bees, limit):
    """
    This function represents the artifical bee colony opimisation algorithm
    """

    # Helper function for calculating fitness
    def fitness(store, func):

        trial_source = store
        trial_eval = func(trial_source)
        if trial_eval >= 0:
            trial_fit = 1/(1+trial_eval)
        else:
            trial_fit = 1 + abs(trial_eval)
        return (trial_fit, trial_eval, trial_source)

    # Helper function for calculating pertubation
    def pertubation(source, n_food, n_iter, bounds):
        index_food = int(np.random.uniform(0,n_food-1))
        while index_food == n_iter:
            index_food = int(np.random.uniform(0,n_food-1))
        store = source[n_iter] + np.random.uniform(-1, 1) * (source[index_food]-source[n_iter])
        store = np.clip(store, bounds[:,0], bounds[:,1])

        return store

    # Storing solutions
    best_source = np.empty(len(bounds))
    best_eval = float('inf')
    object_track = list()
    obj_counter = 0
    obj_counter_track = list()
    timing = list()

    # Initialising parameters
    n_food = mt.floor(n_bees / 2)
    food_source = np.array([bounds[:, 0] + np.random.rand(len(bounds))* (bounds[:, 1] - bounds[:, 0]) for _ in range(n_food)])
    food_eval = [function(food_source[i]) for i in range(n_food)]
    food_fit = np.empty(n_food)
    counter = np.empty(n_food)

    trial_source = np.empty(len(bounds))

    # Calculating initial fitness
    for i in range(len(food_eval)):
        food_fit[i] = 1/(1+food_eval[i]) if food_eval[i] >= 0 else 1 + abs(food_eval[i])

    time_start = time.time()
    i = 0

    while i <= n_iter:

        # Employer bee phase
        for j in range(n_food):
            trial_store = pertubation(food_source, n_food, j, bounds)
            trial_fit, trial_eval, trial_source = fitness(trial_store, function)
            obj_counter += 1

            if trial_fit > food_fit[j]:
                food_eval[j] = trial_eval
                food_fit[j] = trial_fit
                food_source[j] = trial_source
                counter[j] = 0
            else:
                counter[j] += 1

        # Calculating probablities
        trial_prob = [1-food_fit[k]/sum(food_fit) for k in range(n_food)]
        trial_prob = [i/sum(trial_prob) for i in trial_prob]
        indeces = np.random.choice([i for i in range(n_food)], size=n_food, p=trial_prob)

        # Onlooker bee phase
        for h in range(n_food):
            bee = indeces[h]
            trial_store = pertubation(food_source, n_food, bee, bounds)
            trial_fit, trial_eval, trial_source = fitness(trial_store, function)
            obj_counter += 1

            if trial_fit > food_fit[bee]:
                food_eval[bee] = trial_eval
                food_fit[bee] = trial_fit
                food_source[bee] = trial_source
                counter[bee] = 0
            else:
                counter[bee] += 1

        # Storing the best value so far
        if min(food_eval) < best_eval:
            best_eval = min(food_eval)
            best_index = np.argmin(food_eval)
            best_source = food_source[best_index]

        # Scout phase
        scout_bees = [i for i, v in enumerate(counter) if v > limit]
        for k in scout_bees:
    
            if counter[k] > limit:
                food_source[k] = bounds[:,0] + np.random.rand(len(bounds)) * (bounds[:,1]-bounds[:,0])
                food_eval[k] = function(food_source[k])
                obj_counter += 1
                food_fit[k] = 1/(1+food_eval[k]) if food_eval[k] >= 0 else 1 + abs(food_eval[k])
                counter[k] = 0

        timing.append(time.time() - time_start)
        object_track.append(best_eval)
        obj_counter_track.append(obj_counter)
        i += 1

    return (best_eval, best_source, object_track, obj_counter_track, timing)


def firefly_alg(function, bounds, max_eval,pop_size=10, alpha=1.0, betamin=1.0, gamma=0.01,error=1e-5,seed=None):
    """
    This is a function which follows the firefly algorithm
    """
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    obj_track = list()
    obj_counter = 0
    obj_counter_track = list()
    timing = list()

    fireflies = rng.uniform(lb, ub, (pop_size, dim))
    new_fireflies = np.empty((pop_size,dim))
    intensity = np.apply_along_axis(function, 1, fireflies)
    new_intensity = np.empty(pop_size)
    store_val = np.empty(dim)
    pop_double = np.empty((2*pop_size,dim))
    int_double = np.empty(2*pop_size)
    sort_order = int_double
    
    best = np.min(intensity)
    best_source = np.zeros(len(bounds))

    new_alpha = alpha
    dmax = (ub-lb)*np.sqrt(dim)
    delta = 0.05 * (ub-lb)
    time_start = time.time()
    k = 0

    while k <= max_eval:
        for i in range(pop_size):
            new_intensity[i] = float('inf')
            for j in range(pop_size):
                if intensity[i] > intensity[j]:
                    r = np.sum(np.square(fireflies[i] - fireflies[j]))/dmax
                    beta = betamin * np.exp(-gamma * r**2)* np.random.random(dim)
                    steps = new_alpha * (np.random.uniform(-1,1,size=dim)) * delta
                    store_val = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + steps
                    store_val = np.clip(store_val, lb, ub)
                    store_eval = function(store_val)
                    obj_counter += 1

                    if store_eval < new_intensity[i]:
                        new_intensity[i] = store_eval
                        new_fireflies[i] = store_val
                        if store_eval < best:
                             best = store_eval
                             best_source = store_val
                    
                    obj_track.append(best)
                    timing.append(time.time()-time_start)
                    obj_counter_track.append(obj_counter)

        k += 1
        new_alpha *= 0.98
        pop_double = np.concatenate((fireflies,new_fireflies))
        int_double = np.concatenate((intensity,new_intensity))
        sort_order = np.argsort(int_double)
        pop_double = pop_double[sort_order]
        int_double = int_double[sort_order]

        intensity = int_double[0:pop_size]
        fireflies = pop_double[0:pop_size]

        if sum(abs(best-intensity)) < error:
            break
    return (best, best_source, obj_track, obj_counter_track, timing)


def diff_evolution(function,bounds,n_pop,n_iter,crossover=0.9, weight=0.8):
    """
    This is an algorithm representing differential evolution
    Crossover is a probability, weight should in range [0,2]
    Classical values are CR = 0.9, Weight = 0.8 and n_pop = 10 * n
    """
    
    # Storing solutions
    dim = len(bounds)
    best_coords = np.empty(dim)
    best_eval = float('inf')
    object_track = list()
    obj_counter = 0
    obj_counter_track = list()
    timing = list()
    
    #Parameter initialisation
    agent_coords = np.array([bounds[:, 0] + np.random.rand(len(bounds))* (bounds[:, 1] - bounds[:, 0]) for _ in range(n_pop)])
    agent_fit = np.apply_along_axis(function,1,agent_coords)
    agent_store = np.empty(dim)
    index_list = np.empty(3)
    a = np.empty(dim)
    b = np.empty(dim)
    c = np.empty(dim)
    i = 0
    time_start = time.time()

    # Start of algorithm
    while i < n_iter:
        for k in range(n_pop): 
            
            for j in range(3):
               index_list[j] = int(np.random.randint(0,n_pop-1))
               while index_list[j] == k or index_list[j] in index_list[:j]:
                   index_list[j] = int(np.random.randint(0,n_pop-1))

            d,e,f = int(index_list[0]),int(index_list[1]),int(index_list[2])
            a,b,c = agent_coords[d],agent_coords[e],agent_coords[f]
            random_index = np.random.uniform(0,n_pop-1)
            
            for j in range(dim):
                random_no = np.random.rand()
                if random_no < crossover or j == random_index:
                    agent_store[j] = a[j] + weight* (b[j] - c[j])
                else:
                    agent_store[j] = agent_coords[k,j]

            agent_eval = function(agent_store)
            obj_counter += 1

            if agent_eval < agent_fit[k]:
                agent_fit[k] = agent_eval
                agent_coords[k] = agent_store
                
                if agent_eval < best_eval:
                    best_coords = agent_store
                    best_eval = agent_eval

        object_track.append(best_eval)
        timing.append(time.time()- time_start)
        obj_counter_track.append(obj_counter)

        i += 1

    return (best_eval, best_coords, object_track, obj_counter_track, timing)


def dh_simplex(function,bounds,n_iter,c_reflct=1,c_exp=2,c_cont=.5,c_shrnk=.5,error_tol=1e-6):
    """
    This algorithm represents the downhill simplex algorithm
    c_reflct, c_exp, c_cont, c_shrink are the reflection, expansion, contraction and shrink coefficients
    Typical values include the following: c_reflect = 1, c_exp = 2, c_cont = 0.5, c_shrnk = 0.5
    error_tol is a stopping criteria which is compared to standard deviation of function values of simplex
    """
     
     # Centroid function
    def centroid(arr,dim):
        arr = arr[:-1]
        length = arr.shape[0]
        sum_p = np.array([np.sum(arr[:,k]) for k in range(dim)])
        return sum_p/length
     
    # Storing solutions
    dim = len(bounds)
    n_points = dim + 1
    best_coords = np.empty(dim)
    best_eval = float('inf')
    object_track = list()
    obj_counter = 0
    obj_counter_track = list()
    timing = list()
    
    #Parameter initialisation
    point_coords = np.array([bounds[:, 0] + np.random.rand(len(bounds))* (bounds[:, 1] - bounds[:, 0]) for _ in range(n_points)])
    points_fit = np.apply_along_axis(function,1,point_coords)
    centroid_coor = np.empty(dim)
    x_reflct = np.empty(dim)
    x_exp = np.empty(dim)
    x_cont = np.empty(dim)
    sort_order = np.empty(dim+1)
    i = 0
    time_start = time.time()
    # Start of algorithm

    while i < n_iter:
        # Step 1: Order points according to values of indeces
        sort_order = np.argsort(points_fit)
        points_fit = points_fit[sort_order]
        point_coords = point_coords[sort_order]

        # Calculate standard deviation of function values for simplex
        deviation = np.std(points_fit)
        if deviation < error_tol:
            break

        # Step 2: Calculate the centroid
        centroid_coor = centroid(point_coords,dim)

        # Step 3: Reflection / Expansion / Contraction / Shrinking
        x_reflct = centroid_coor + c_reflct* (centroid_coor - point_coords[-1])
        reflct_eval = function(x_reflct)
        obj_counter += 1

        if points_fit[0] <= reflct_eval <= points_fit[-2]: # Reflection
           points_fit[-1] = reflct_eval
           point_coords[-1] = x_reflct
        elif reflct_eval < points_fit[0]: # Expansion
            x_exp = centroid_coor + c_exp* (x_reflct-centroid_coor)
            exp_eval = function(x_exp)
            obj_counter += 1
            if exp_eval < reflct_eval:
                points_fit[-1] = exp_eval
                point_coords[-1] = x_exp
            else:
                points_fit[-1] = reflct_eval
                point_coords[-1] = x_reflct
        else: # Contraction
           x_cont = centroid_coor + c_cont*(point_coords[-1]-centroid_coor)
           cont_eval = function(x_cont)
           obj_counter += 1
           if cont_eval < points_fit[-1]:
                points_fit[-1] = cont_eval
                point_coords[-1] = x_cont
           else: # Shrinking
               best_point = point_coords[0]
               for j in range(1,dim):
                  point_coords[j] = best_point + c_shrnk* (point_coords[j]-best_point)
                  points_fit[j] = function(point_coords[j])
                  obj_counter += 1
        
        # Get the best point
        if points_fit[0] < best_eval:
            best_eval = points_fit[0]
            best_coords = point_coords[0]
        
        object_track.append(best_eval)
        timing.append(time.time()-time_start)
        obj_counter_track.append(obj_counter)
        i += 1

    return (best_eval, best_coords, object_track, obj_counter_track, timing)


def genetic(function,bounds,survival_percentage,no_points,no_iterations):

    #Initialising variables
    boundlength = len(bounds)
    dimensions = list(range(1,boundlength+1))
    particles = list(range(1,no_points+1))
    pos = np.zeros((len(particles),len(dimensions)+1))
    timing = list()
    obj_track = list()
    t_start = time.time()

    # First set of points
    for particle in particles:
        for dimension in dimensions:
            pos[particle-1,dimension-1] = random.randrange(round(bounds[dimension-1,0]),round(bounds[dimension-1,1]))
        pos[particle-1,-1] = function(pos[particle-1,0:-1])
    pos = np.sort(pos,axis=0)

    # Producing new points and iterating
    iterations = list(range(1,no_iterations))
    for iteration in iterations:
        no_survivors = round(survival_percentage/100*len(pos))
        pos = pos[0:no_survivors,:]

        children = np.zeros((round(len(pos)*100/survival_percentage-no_survivors),len(dimensions)+1))
        for child in range(round(len(pos)*100/survival_percentage)-no_survivors): 
            for dimension in dimensions:
                children[child-1,dimension-1] = np.clip(0.01*random.randrange(-90,110)*random.choice(pos[0:no_survivors,dimension-1]),bounds[dimension-1,0],bounds[dimension-1,1])

            children[child-1,-1] = function(children[child-1,0:-1])

        pos = np.vstack([children,pos])
        pos = np.sort(pos,axis=0,)
        best = pos[0,-1]
        best_index = pos[0,0:-1]
        timing.append(time.time()-t_start)
        obj_track.append(best)
    return (obj_track, timing, best, best_index)