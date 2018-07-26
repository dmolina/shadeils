from math import sqrt, exp, pi, sin
import numpy as np


def get_info(fun, dim):
    """
    Return the lower bound of the function

    """
    bounds = [100, 10, 100, 100, 30, 100, 1.28, 500, 5.12, 32, 600, 50, 50]
    objective_value = 0

    if fun == 8:
        objective_value = -12569.5 * dim / 30.0

    return {
        'lower': -bounds[fun - 1],
        'upper': bounds[fun - 1],
        'threshold': objective_value,
        'best': 1e-8
    }


def get_function(fun):
    """
    Evaluate the solution
    @param fun function value (1-13)
    @param x solution to evaluate
    @return the obtained fitness
    """
    functions = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13]
    return functions[fun - 1]


def f1(x):
    """
    function 1: Sphere
    @param x solution
    """
    return (x**2).sum()


def f2(x):
    """
    function 2: Schwefel's Problem 2.22
    @param x solution
    """
    f = np.abs(x).sum()
    g = np.abs(x).prod()
    return f + g


def f3(x):
    """
    function 3: Schwefel's Problem 1.2
    @param x solution
    """
    #    f = 0
    #    size = np.size(x)
    #
    #    for i in xrange(size):
    #       f = f + np.sum(x[:i+1])**2
    f = np.power(np.add.accumulate(x), 2).sum()
    return f


def f4(x):
    """
    function 4: Schwefel problem 2.21
    @param x solution
    """
    return np.abs(x).max()


def f5(x):
    """
    function 5: Generalized Rosenbrock's Function
    @param x solution
    """
    size = np.size(x)
    total = 0

    for i in xrange(size - 1):
        total += 100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2

    return total


def f6(x):
    """
    function 6: Step Function
    @param x solution
    """
    return np.power(np.floor(x + 0.5), 2).sum()


def f7(x):
    """
    function 7: Quartic Function with Noise
    @param x solution
    """
    dim = np.size(x)
    expr = np.arange(1, dim + 1) * np.power(x, 4)
    return expr.sum() + np.random.uniform(0, 1)


def f8(x):
    """
    function 8: Generalized Schwefel's Problem 2.26
    @param x solution
    """
    return -(x * np.sin(np.sqrt(np.abs(x)))).sum()


def f9(x):
    """
    function 9: Generalized Rastrigin's Function
    @param x solution
    """
    expr = x**2 - 10 * np.cos(2 * pi * x) + 10
    return expr.sum()


def f10(x):
    """
    function 10: Ackley's Function
    @param x solution
    """
    a = (x**2).sum()
    b = np.cos(2 * pi * x).sum()
    dim = np.size(x)
    f = -20 * exp(-0.2 * sqrt(1.0 / dim * a)) - exp(
        (1.0 / dim) * b) + 20 + exp(1)
    return f


def f11(x):
    """
    function 11: Generalized Griewank Function
    @param x solution
    """
    a = (x**2).sum()
    dim = np.size(x)
    b = np.cos(x / np.sqrt(np.arange(1, dim + 1))).prod()
    return 1.0 / 4000 * a - b + 1


def u(x, a, k, m):
    if x > a:
        result = k * pow(x - a, m)
    elif -a <= x <= a:
        result = 0
    elif x < -a:
        result = k * pow(-x - a, m)
    else:
        print("Error")

    return result


def f12(x):
    """
    function 12: Penalized Functions f12
    @param x solution
    """
    dim = np.size(x)
    y = 1 + (0.25 * (x + 1))
    f = 0
    g = 0

    for i in range(dim - 1):
        f = f + (pow(y[i] - 1, 2) * (1 + 10 * pow(sin(pi * y[i + 1]), 2)))

    f = f + (10 * pow(sin(pi * y[0]), 2)) + pow(y[dim - 1] - 1, 2)
    f = f * pi / 30

    for i in range(dim):
        g += u(x[i], 5., 100., 4)

    return f + g


def f13(x):
    """
    function 13: Penalized Functions f13
    @param x solution
    """
    dim = np.size(x)
    f = 0
    g = 0

    for i in range(dim - 1):
        f = f + (pow(x[i] - 1, 2) * (1 + pow(sin(3 * pi * x[i + 1]), 2)))

    f = f + pow(sin(3 * pi * x[0]),
                2) + (pow(x[dim - 1] - 1, 2) *
                      (1 + sin(pow(2 * pi * x[dim - 1], 2))))
    f = f * 0.1

    for i in xrange(dim):
        xi = x[i]

        if xi > 5:
            g = g + 100 * (xi - 5)**4
        elif xi <= 5 and xi >= -5:
            g = g + 0
        elif xi < -5:
            g = g + pow(100 * ((-1) * xi - 5), 4)

    return f + g
