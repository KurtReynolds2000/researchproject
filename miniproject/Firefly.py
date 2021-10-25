#Firefly that minimises obj function. No graphing yet

import random
import numpy as np
import time
import matplotlib.pyplot as plt

def Rosenbrock(x):
    z = (100 * ((x[1] - (x[0]**2))**2)) + ((1 - x[0])**2)
    return z

def firefly(no_particles,no_iterations,lowerbound,upperbound,beta,gamma,function):
    
    #Initialising variables
    start = time.time()
    time_axis = []
    iteration_axis = []
    obj_axis = []
    dimensions = list(range(1,len(lowerbound)+1))
    particles = list(range(1,no_particles+1))
    pos = np.zeros((len(particles),len(dimensions)))
    for dimension in dimensions:
        pos[0,dimension-1] = random.randrange(lowerbound[dimension-1],upperbound[dimension-1])
    
    # Initialising particle position
    for particle in particles:
        for dimension in dimensions:
            pos[particle-1,dimension-1] = random.randrange(lowerbound[dimension-1],upperbound[dimension-1])
        
    # Iteration
    iterations = list(range(1,no_iterations))
    for iteration in iterations:
        for particlei in particles:
            for particlej in particles:
                for dimension in dimensions:
                    if function(pos[particlei-1]) < function(pos[particlej-1]):
                        pos[particlei-1,dimension-1] += beta*np.exp(-gamma*(pos[particlej-1,dimension-1]-pos[particlei-1,dimension-1])**2)*(pos[particlei-1,dimension-1]-pos[particlej-1,dimension-1])
        time_axis.append(time.time() - start)
        iteration_axis.append(iteration)
        obj_axis.append(max(function(pos)))
        print(max(obj_axis))
    
                        
firefly(50,100,[-100,-100],[100,100],1,.2,Rosenbrock)