import math as mt


def Rosenbrock(x):
    """
    x within range [-Inf, Inf] for all n dimensions
    Optimum is f(x) = 0 for x = 1
    """

    z = sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    return z


def Rastrigin(x):
    """
    x within range of [-5.12,5.12] for all n dimensions
    Optimum is f(x) = 0 for x = 0
    """

    z = 10*len(x)
    for i in range(len(x)):
        z += x[i]**2 - (10 * mt.cos(2 * mt.pi * x[i]))

    return z


def Ackley(x):
    """
    x within range [-5,5] for all n dimensions
    Optimum is f(x) = 0 for x = 0
    """

    a = 20
    b = 0.2
    c = 2 * mt.pi
    d = len(x)

    sum_1 = 0
    sum_2 = 0

    for i in range(d):
        x_i = x[i]
        sum_1 += x_i**2
        sum_2 += mt.cos(c * x_i)

    term_1 = -a * mt.exp(-b * mt.sqrt(sum_1/d))
    term_2 = -mt.exp(sum_2/d)

    return term_1 + term_2 + a + mt.exp(1)


def Beale(x):
    """
    x within range [-4.5,4.5]
    Optimum is f(x) = 0 for x = [3,0.5] 
    """

    term_1 = (1.5-x[0] + x[0] * x[1])**2
    term_2 = (2.25 - x[0] + x[0] * x[1]**2)**2
    term_3 = (2.625-x[0] + x[0] * x[1]**3)**2

    return term_1 + term_2 + term_3


def Goldstein(x):
    """
    x within range [-2,2]
    Optimum is f(x) = 3 for x = [0,-1]
    """

    X = x[0]
    Y = x[1]

    term_1 = (1+(X+Y+1)**2*(19-14*X+3*X**2-14*Y+6*X*Y+3*Y**2))
    term_2 = (30+(2*X-3*Y)**2*(18-32*X+12*X**2+48*Y-36*X*Y+27*Y**2))

    return term_1*term_2

def Michalewitz(x):
    m = 10
    z = 0
    for i in range(len(x)):
        z += mt.sin(x[i])*mt.sin((i+1)*x[i]**2/mt.pi)**(2*m)
    return -z    


def Easom(x):
    x1 = x[0]
    x2 = x[1]
    term1 = -mt.cos(x1)*mt.cos(x2)
    term2 = mt.exp(-(x1-mt.pi)**2 - (x2-mt.pi)**2)
    return term1*term2