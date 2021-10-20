import random
import numpy as np

def Rosenbrock(x):
    z = (100 * ((x[1] - (x[0]**2))**2)) + ((1 - x[0])**2)
    return z

def particleswarm(no_particles,no_iterations,lowerbound,upperbound,cogco,socco,w,function):
    
    particles = list(range(1,no_particles))
    pos = np.array()
    for dimension in lowerbound:
        pos[0,dimension] = random.randrange(lowerbound[dimension],upperbound[dimension])
    velocity = []
    localbest = []
    globalbest = function(pos[0])


    for particle in particles:
        for dimension in lowerbound:
            pos[particle,dimension] = random.randrange(lowerbound[dimension],upperbound[dimension])
            velocity[particle,dimension] = random.randrange(-abs(upperbound[dimension]-lowerbound[dimension]),abs(upperbound[dimension]-lowerbound[dimension]))

        localbest[particle] = pos[particle]
        if function(localbest[particle]) < globalbest:
            globalbest = localbest[particle]
        
    iterations = list(range(1,no_iterations))
    for iteration in iterations:
        for particle in particles:
            for dimension in lowerbound:
                rp = random.randrange(0,1)
                rg = random.randrange(0,1)
                velocity[particle,dimension] = w*velocity[particle,dimension] + cogco*rp*(localbest[particle] - pos[particle]) + socco*rg*(globalbest - pos[particle])
                pos[particle,dimension] += velocity[particle,dimension]
                if function(pos[particle]) < function(localbest[particle]):
                    localbest[particle] = pos[particle]
                    if function(localbest[particle]) < globalbest:
                        globalbest = localbest[particle]
                        return globalbest