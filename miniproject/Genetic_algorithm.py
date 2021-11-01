import random
import numpy as np

def Rosenbrock(x):
    z = (100 * ((x[1] - (x[0]**2))**2)) + ((1 - x[0])**2)
    return z

def genetic(no_points,no_iterations,lowerbound,upperbound,survival_percentage,function):
    
    #Initialising variables
    dimensions = list(range(1,len(lowerbound)+1))
    particles = list(range(1,no_points+1))
    pos = np.zeros((len(particles),len(dimensions)+1))

    # First set of points
    for particle in particles:
        for dimension in dimensions:
            pos[particle-1,dimension-1] = random.randrange(lowerbound[dimension-1],upperbound[dimension-1])
        pos[particle-1,-1] = function(pos[particle-1,0:-1])
    
    # Producing new points and iterating
    iterations = list(range(1,no_iterations))
    for iteration in iterations:
        pos = np.sort(pos,axis=0)
        no_survivors = round(survival_percentage/100*len(pos))
        pos = pos[0:no_survivors,:]

        children = np.zeros((round(len(pos)*100/survival_percentage-no_survivors),len(dimensions)+1))
        for child in range(round(len(pos)*100/survival_percentage)-no_survivors): 
            for dimension in dimensions:
                children[child-1,dimension-1] = 0.01*random.randrange(95,105)*random.choice(pos[0:no_survivors,dimension-1])

            children[child-1,-1] = function(children[child-1,0:-1])
    
        pos = np.vstack([pos, children])
    return pos[0,-1]

genetic(20,2,[-10,-10],[10,10],25,Rosenbrock)