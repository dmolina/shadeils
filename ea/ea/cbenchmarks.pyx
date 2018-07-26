#cython: boundscheck=False,wraparound=False
import numpy as np
import math
from libc.math cimport abs, floor, sin, cos, sqrt, pow, exp
from libc.stdlib cimport malloc, free

"""
This module includes the benchmarks used in the comparisons. 
The tests are in test_benchmarks.py, using doctests by simplificy.
"""
import cython

cdef double pi = math.pi
#cdef double nan = math.nan

def get_info(fun,dim):
    """
    Return the lower bound of the function

    """
    bounds = [100, 10, 100, 100, 30, 100, 1.28, 500, 5.12, 32, 600, 50, 50]
    objective_value = 0

    if fun == 8:
        objective_value = -12569.5*dim/30.0

    return {'lower': -bounds[fun-1], 'upper': bounds[fun-1], 'threshold': 1e-8, 'best': objective_value}

def get_function(int fun):
    """
    Evaluate the solution
    @param fun function value (1-13)
    @param x solution to evaluate
    @return the obtained fitness
    """
    functions = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13]
    return functions[fun-1]

cdef double sphere(double[::1]x):
    """
    function 1: sphere function: (x*x).sum()
    @param x solution
    """
    cdef:
        unsigned int i, n
        double sum
        
    n = x.shape[0]
    sum = 0.0
    
    for i in range(n):
        sum = sum + x[i]*x[i]
        
    return sum

def f1(double[::1]x):
    return sphere(x)

cpdef double f2(double[::1]x):
    """
    function 2: Schwefel's Problem 2.22
    @param x solution
    #   f = np.abs(x).sum()
    #   g = np.abs(x).prod()
    #   return f + g
    """
    cdef:
        double f, g
        double abs_i
        unsigned i, n

    n = x.shape[0]
    f, g = 0, 1

    for i in range(n):
        abs_i = abs(x[i])
        f += abs_i
        g *= abs_i

    return f + g

cpdef double f3(double[::1] x):
    """
    function 3: Schwefel's Problem 1.2
    @param x solution
    # np.power(np.add.accumulate(x), 2).sum()
    """
    cdef:
        unsigned i, n
        double sum, acc

    n = x.shape[0]
    sum, acc = 0, 0

    for i in range(n):
        acc += x[i]
        sum += acc*acc

    return sum

cpdef double f4(double[::1] x):
    """
    function 4: Schwefel problem 2.21
    @param x solution
    #np.abs(x).max()
    """
    cdef:
        unsigned i, n
        double max_i,abs_i

    n = x.shape[0]
    max_i = 0

    for i in range(n):
        abs_i = abs(x[i])

        if (abs_i > max_i):
            max_i = abs_i

    return max_i

cpdef double f5(double[::1] x):
    """
    function 5: Generalized Rosenbrock's Function
    @param x solution
    """
    cdef:
        unsigned i, n
        double a, b, total

    n = x.shape[0]
    total = 0

    for i in range(n-1):
        a = x[i+1]-x[i]*x[i]
        b = x[i]-1
        total += 100*a*a+b*b

    return total 

cpdef double f6(double [::1] x):
    """
    function 6: Step Function
    @param x solution
    #  np.power(np.floor(x+0.5), 2).sum()
    """
    cdef:
        unsigned i, n
        double elem, fit

    n = x.shape[0]
    fit = 0

    for i in range(n):
        elem = floor(x[i]+0.5)
        fit += elem*elem
    
    return fit

cpdef double f7(double[::1] x):
    """
    function 7: Quartic Function with Noise
    @param x solution
    #(np.arange(1, dim+1)*np.power(x, 4) + np.random.uniform(0, 1, dim)).sum()
    """
    cdef:
        unsigned i, dim
        double fit, xi
        double[::1] random

    dim = x.shape[0]
    fit = 0

    for i in range(dim):
        xi = x[i]
        fit += (i+1)*xi*xi*xi*xi

    fit += np.random.uniform(0, 1)
    return fit

cpdef double f8(double[::1] x):
    """
    function 8: Generalized Schwefel's Problem 2.26
    @param x solution
    #(x*np.sin(np.sqrt(np.abs(x)))).sum()
    """
    cdef: 
        unsigned i, dim
        double fit

    dim = x.shape[0]
    fit = 0

    for i in range(dim):
        fit += x[i]*sin(sqrt(abs(x[i])))

    return -fit

cpdef double f9(double[::1] x):
    """
    function 9: Generalized Rastrigin's Function
    @param x solution
    #(x**2-10*np.cos(2*pi*x)+10).sum()
    """
    cdef:
        unsigned i, n = x.shape[0]
        double fit = 0

    for i in range(n):
        fit += x[i]*x[i] - 10*cos(2*pi*x[i])+10

    return fit

cpdef double f10(double[::1] x):
    """
    function 10: Ackley's Function
    @param x solution
    """
    cdef:
        double a, b, fit
        unsigned i, dim

    a = sphere(x)
    dim = x.shape[0]
    b = 0

    for i in range(dim):
        b += cos(2*pi*x[i])

    fit = -20 * exp(-0.2*sqrt((1.0/dim)*a)) - exp((1.0/dim)*b) + 20 + exp(1)
    return fit

cpdef double f11(double[::1] x):
    """
    function 11: Generalized Griewank Function
    @param x solution
    """
    cdef:
        double a, b, fit
        unsigned i, dim = x.shape[0]
        
    a = sphere(x)
    b = 1

    for i in range(dim):
        b *= cos(x[i]/sqrt(i+1))

    fit = 1.0/4000*a - b + 1
    return fit

cdef double u(double x,double a, double k, unsigned m):
    cdef:
        double result

    if x > a:
        result = k*pow(x-a,m)
    elif -a <= x <= a:
        result = 0
    elif x < -a:
        result = k*pow(-x-a,m)
    else:
        print "Error"

    return result

cpdef double f12(double[::1] x):
    """
    function 12: Penalized Functions f12
    @param x solution
    """
    cdef:
        unsigned i, dim = x.shape[0]
        double *y = <double *>malloc(dim*sizeof(double))
        double f, g

    for i in range(dim):
        y[i] = 1 + 0.25*(x[i]+1)

    f = 0
    g = 0

    for i in range(dim-1):
        f += (y[i]-1)*(y[i]-1) * (1 + 10*pow(sin(pi*y[i+1]), 2))
  
    f += (10 * pow(sin(pi*y[0]), 2)) + pow(y[dim-1]- 1, 2)
    f = f * pi/30
  
    for i in range(dim):
        g += u(x[i], 5., 100., 4)

    free(y)

    return f + g

cpdef double f13(double[::1] x):
    """
    function 13: Penalized Functions f13
    @param x solution
    """
    cdef:
        unsigned i, dim = x.shape[0]
        double f, g

    f = 0
    g = 0

    for i in range(dim-1):
        f = f + (pow(x[i]-1, 2) * (1 + pow(sin(3*pi*x[i+1]), 2)))

    f = f + pow(sin(3*pi*x[0]), 2) + (pow(x[dim-1] - 1, 2) * (1 + sin(pow(2*pi*x[dim-1], 2))))
    f = f * 0.1

    for i in xrange(dim):
        g += u(x[i], 5., 100., 4)
   
    return f + g
