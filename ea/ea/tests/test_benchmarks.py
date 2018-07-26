"""
This file is for testing the benchmarks only. 
It uses the doctest utility for simplicity.

For each function it checks its optima, and results know for simple vectors.
"""
from numpy import ones, zeros, arange
from math import sin, cos
import ea.benchmarks as b

def test_f1():
    """
    Testing the f1 (sphere functions)
    >>> b.f1(zeros(10))
    0.0
    >>> b.f1(ones(10))
    10.0
    >>> b.f1(ones(5))
    5.0
    >>> b.f1(arange(0.,10.))
    285.0
    >>> b.f1(0.5*ones(5))
    1.25
    """
    pass

def test_f2():
    """
    Schwefel's problem 2.22
    #   f = abs(x).sum()
    #   g = abs(x).prod()
    #   return f + g
    >>> b.f2(ones(10))
    11.0
    >>> b.f2(-ones(10))
    11.0
    >>> b.f2(zeros(10))
    0.0
    >>> b.f2(arange(0.,10.))
    45.0
    """
    pass

def test_f3():
    """
    function 3: Schwefel's Problem 1.2f
    @param x solution
    # power(add.accumulate(x), 2).sum()
    >>> b.f3(ones(5))
    55.0
    >>> b.f3(zeros(10))
    0.0
    >>> b.f3(arange(5.))
    146.0
    """
    pass

def test_f4():
    """
    Schwefel problem 2.21
    abs(x).max()
    >>> b.f4(ones(10))
    1.0
    >>> b.f4(-1*ones(10))
    1.0
    >>> b.f4(arange(0., 11.))
    10.0
    >>> b.f4(zeros(10))
    0.0
    >>> b.f4(arange(-10., 1.))
    10.0
    """
    pass

def test_f5():
    """
    Generalized Rosenbrock's Function
    >>> b.f5(ones(10))
    0.0
    >>> b.f5(zeros(10))
    9.0
    >>> b.f5(arange(0.,2.))
    101.0
    >>> b.f5(arange(0.,3.))
    201.0
    >>> b.f5(arange(0.,4.))
    302.0
    """
    pass

def test_f6():
    """
    >>> b.f6(ones(10))
    10.0
    >>> b.f6(zeros(10))
    0.0
    >>> b.f6(0.4+ones(10))
    10.0
    >>> b.f6(0.5+ones(10))
    40.0
    """
    pass

def test_f7():
    """
    Testing f7: Quartic Function with Noise
    This function can not tested by doctests because the error <= 1.0
    """
    sols = [ones(10), zeros(10), arange(0., 3.)]
    expected = [55., 1., 50.]

    for i, sol in enumerate(sols):
        fit = b.f7(sol)
        assert abs(fit-expected[i])<=1.0

def test_f8():
    """
    Generalized Schwefel's Problem 2.26
    #(x*np.sin(np.sqrt(np.abs(x)))).sum()
    >>> b.f8(zeros(10))<1e-15
    True
    >>> b.f8(ones(10))==-10*sin(1)
    True
    >>> b.f8(-1*ones(10))==10*sin(1)
    True
    >>> b.f8(arange(0.,3.))
    -2.8170028767933677
    >>> abs(b.f8(420.9687*ones(30))+12569.486618164879)<1e-15
    True
    """
    pass

def test_f9():
    """
    function 9: Generalized Rastrigin's Function
    #(x**2-10*np.cos(2*pi*x)+10).sum()
    >>> b.f9(zeros(10))
    0.0
    >>> b.f9(ones(10))
    10.0
    >>> b.f9(ones(5))
    5.0
    >>> b.f9(arange(0.,10.))
    285.0
    >>> b.f9(0.5*ones(5))
    101.25
    """
    pass

def test_f10():
    """
    function 10: Ackley's function
    >>> b.f10(ones(10))
    3.6253849384403627
    >>> b.f10(arange(0., 5.))
    7.7462216643521575
    >>> b.f10(zeros(30))<1e-15
    True
    >>> b.f10(zeros(10))<1e-15 
    True
    """
    pass

def test_f11():
    """
    function 11: Generalized Griewank Function
    >>> b.f11(zeros(30))
    0.0
    >>> b.f11(zeros(10))
    0.0
    >>> abs(b.f11(ones(10))-0.8067591547236139)<1e-8
    True
    >>> b.f11(arange(0.,11.))
    1.097157467637113
    >>> b.f11(arange(0.,2.))
    0.24000540292436978
    """
    pass

def test_f12():
    """
    function 12: Penalized Functions f12
    >>> b.f12(zeros(30))
    1.668971097219577
    >>> abs(b.f12(ones(30))-9.42477796076938)<1e-15
    True
    >>> b.f12(zeros(10))
    0.8835729338221292
    >>> abs(b.f12(ones(10))-3.6651914291880918)<1e-15
    True
    >>> abs(b.f12(-1*ones(30)))<1e-15
    True
    """
    pass

def test_f13():
    """
    function 13: Penalized Functions f13
    >>> b.f13(zeros(30))
    3.0
    >>> b.f13(zeros(10))
    1.0
    >>> abs(b.f13(arange(0., 2.))-0.1)<1e-15
    True
    >>> b.f13(ones(10))<1e-15
    True
    >>> b.f13(ones(30))<1e-15
    True
    """
    pass
