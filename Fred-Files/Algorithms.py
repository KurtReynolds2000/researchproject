import numpy as np
import time
from math import inf,exp
import math as mt


def sim_annealing(function, n_iter, M, bounds, t_schedule, t_range, step):
    """
    This is the algorithm for performing simulated annealing including several cooling schedules
    """

    # Define certain cooling schedules
    def cool_straight(t_i,t_f,M,i):
        return (t_f-t_i)/(M-1) * (i-1) + t_i
    
    def cool_geo(t_i,t_f,M,i):
        return t_i*(t_f/t_i)**((i-1)/(M-1))
    
    def cool_rec(t_i,t_f,M,i):
        return t_f*t_i*(M-1)/((t_f*M-t_i)+(t_i-t_f)*i)

    schedules = {1:cool_straight,2:cool_geo,3:cool_rec}
    t_select = schedules[t_schedule]


    # Initialise variables

    temp = sum(t_range)/len(t_range)
    best_val = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    best_eval = function(best_val)
    curr_val, curr_eval = best_val, best_eval
    t_init, t_final = t_range[0], t_range[1]
    obj_track = list()
    timing = list()
    t_start = time.time()

    for j in range(M):    # Outer loop iteration

        for i in range(n_iter): 
            config_val = curr_val + np.random.randn(len(bounds)) * step
            config_eval = function(config_val)
            diff = config_eval - curr_eval

            if config_eval < best_eval:
                best_eval,best_val = config_eval,config_val
                curr_eval,curr_val = config_eval,config_val
            else:

                if np.random.rand() < exp(-diff / temp):
                    curr_eval,curr_val = config_eval,config_val
                else:
                    pass

            timing.append(time.time()-t_start)
            obj_track.append(best_eval)
        
        temp = t_select(t_init,t_final,M,j)

    return [best_eval,best_val,obj_track,timing]

def genetic(function,placeholder):
    pass

def particle_swarm(function,n_iter,error,bounds,n_particles,parameter):
    """
    This is an algorithm for the particle swarm optimisation
    """
    # Initialise parameters for algorithm
    w,c_1,c_2 = parameter[0],parameter[1],parameter[2]
    # Initialise individual particle properties
    part_position = np.zeros((n_particles,len(bounds)), dtype = np.float32)
    part_velocity = part_position
    obj_track = list()
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

            if eval_candidate < part_beval[k]:
                part_beval[k] = eval_candidate
                part_bposition[k] = part_position[k]

            if eval_candidate < group_eval:
                group_eval = eval_candidate
                group_position = part_position[k]
            
            part_velocity[k] = w* part_velocity[k]+ c_1 * np.random.random()* (part_bposition[k]-part_position[k])+ c_2 * np.random.random()* (group_position- part_position[k])
            part_position[k] += part_velocity[k]

        if sum(abs(group_eval- part_beval)) < error:
            break

        i = i + 1
        timing.append(time.time()-t_start)
        obj_track.append(group_eval)

    return [group_eval,group_position,obj_track,timing]

def artifical_bee(function,n_iter,bounds,n_bees,limit):
    """
    This function represents the artifical bee colony opimisation algorithm
    """
    
    # Helper function for calculating fitness
    def fitness(store,func):

        trial_source = store
        trial_eval = func(trial_source)
        if trial_eval >= 0:
            trial_fit = 1/(1+trial_eval)
        else:
            trial_fit = 1 + abs(trial_eval)
        return (trial_fit,trial_eval,trial_source)

    # Helper function for calculating pertubation
    def pertubation(source,n_food,n_iter,bounds):
        store = np.zeros(len(bounds))

        for k in range(len(bounds)):

            index_food = np.random.choice([i for i in range(n_food) if i != j ])
            store[k] = source[n_iter,k] + np.random.uniform(-1,1)* (source[index_food,k]-source[n_iter,k])

        return store

    # Storing solutions
    best_source = np.zeros(len(bounds))
    best_eval = float('inf')
    object_track = list()
    timing = list()
    
    # Initialising parameters  
    n_food = mt.floor(n_bees / 2)
    food_source = np.array([bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0]) for i in range(n_food)])
    food_eval = [function(food_source[i]) for i in range(n_food)]
    food_fit = np.zeros(n_food)
    counter = np.zeros(n_food)

    trial_source = np.zeros(len(bounds))

    # Calculating initial fitness
    for i in range(len(food_eval)):
        food_fit[i] = 1/(1+food_eval[i]) if food_eval[i] >= 0 else 1 + abs(food_eval[i])

    time_start = time.time()

    for i in range(n_iter):

        # Employer bee phase
        for j in range(n_food):
            trial_store = pertubation(food_source,n_food,j,bounds)
            test_array = np.array([bounds[i,0] <= trial_store[i] <= bounds[i,1] for i in range(len(bounds))])

            if test_array.all():
                trial_fit,trial_eval,trial_source = fitness(trial_store,function)

                if trial_fit > food_fit[j]:
                    food_eval[j] = trial_eval
                    food_fit[j] = trial_fit
                    food_source[j] = trial_source
                    counter[j] = 0
                else:
                    counter[j] += 1
            
        # Calculating probablities
        trial_prob = [food_fit[k]/sum(food_fit) for k in range(n_food)]
        indeces = np.random.choice([i for i in range(n_food)],size = n_food,p = trial_prob)

        # Onlooker bee phase
        for h in range(n_food):
            bee = indeces[h]
            trial_store = pertubation(food_source,n_food,bee,bounds)
            test_array = np.array([bounds[i,0] <= trial_store[i] <= bounds[i,1] for i in range(len(bounds))])

            if test_array.all():
                trial_fit,trial_eval,trial_source = fitness(trial_store,function)

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
        for k in range(n_food):
            if counter[k] > limit:
                food_source[k] = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
                food_eval[k] = function(food_source[k])
                food_fit[k] = 1/(1+food_eval[k]) if food_eval[k] >= 0 else 1 + abs(food_eval[k])
                counter[k] = 0
            else:
                pass

        timing.append(time.time() - time_start)
        object_track.append(best_eval)

    return (best_eval,best_source,object_track, timing)