import random
import numpy as np

def Rosenbrock(x):
    z = (100 * ((x[1] - (x[0]**2))**2)) + ((1 - x[0])**2)
    return z

def particleswarm(no_particles,no_iterations,lowerbound,upperbound,cogco,socco,w,function):
    
    #Initialising variables
    dimensions = list(range(1,len(lowerbound)+1))
    particles = list(range(1,no_particles+1))
    pos = np.zeros((len(particles),len(dimensions)))
    for dimension in dimensions:
        pos[0,dimension-1] = random.randrange(lowerbound[dimension-1],upperbound[dimension-1])
    velocity = np.zeros((len(particles),len(dimensions)))
    localbest = np.zeros((len(particles),len(dimensions)))
    globalbest = pos[0,:]
    

    # Initialising particle velocities and position
    for particle in particles:
        for dimension in dimensions:
            pos[particle-1,dimension-1] = random.randrange(lowerbound[dimension-1],upperbound[dimension-1])
            velocity[particle-1,dimension-1] = random.randrange(-abs(upperbound[dimension-1]-lowerbound[dimension-1]),abs(upperbound[dimension-1]-lowerbound[dimension-1]))

        localbest[particle-1] = pos[particle-1]

        if function(localbest[particle-1,:]) < function(globalbest):
            globalbest = localbest[particle-1,:]
        
    # Iteration
    iterations = list(range(1,no_iterations))
    for iteration in iterations:
        for particle in particles:
            for dimension in dimensions:
                rp = random.randrange(0,1)
                rg = random.randrange(0,1)
                velocity[particle-1,dimension-1] = w*velocity[particle-1,dimension-1] + cogco*rp*(localbest[particle-1,dimension-1] - pos[particle-1,dimension-1]) + socco*rg*(globalbest[dimension-1] - pos[particle-1,dimension-1])
                pos[particle-1,dimension-1] += velocity[particle-1,dimension-1]
                if function(pos[particle-1]) < function(localbest[particle-1]):
                    localbest[particle-1] = pos[particle-1]
                    if function(localbest[particle-1]) < function(globalbest):
                        globalbest = localbest[particle-1]
                        print(function(globalbest))

particleswarm(100,10,[-10,-10],[10,10],1,1,.5,Rosenbrock)