
import numpy as np
from collections import namedtuple
import os

EAresult = namedtuple("EAresult", "fitness solution evaluations")

def random_population(domain, dimension, size):
    """
    Return an initial population using a uniform random generator
    """
    assert domain[0] < domain[1]
    uniform = np.random.uniform(domain[0], domain[1], dimension*size)
    return uniform.reshape((size, dimension))

def clip(domain, solution):
    """
    Returns the solution clippd between the values of the domain.

    Params
    ------
    domain vector with the lower and upper values.
    """
    assert domain[0] < domain[1]
    return np.clip(solution, domain[0], domain[1])

def get_experiments_file(name_output, replace=False, times=1):
    """
    Return

    Params
    ------
    name_output name of file output.
    replace boolean value that indicates if the file output should be replaced.
    times number of lines that should be in the output (they are maintained if
          replace is False).
    """
    if name_output is None:
        fid = None
    else:
        # if it replaced it only return the last value
        if not replace and os.path.isfile(name_output):
            fin = open(name_output, 'rb')
            lines = fin.readlines()

            if len(lines) >= times:
                (bestSolutionFitness, bestSol, bestEval, evaluations) = lines[-1].split(',')
                return EAresult(fitness=bestSolutionFitness, solution=bestSol, evaluations=evaluations), None

        if replace:
            fid = open(name_output, 'w')
        else:
            fid = open(name_output, 'a')

    return None, fid

def random_indexes(n, size, ignore=[]):
    """
    Returns a group of n indexes between 0 and size, avoiding ignore indexes.

    Params
    ------
    n number of indexes.
    size size of the vectors.
    ignore indexes to ignore.

    >>> random_indexes(1, 1)
    0
    >>> random_indexes(1, 2, [0])
    1
    >>> random_indexes(1, 2, [1])
    0
    >>> random_indexes(1, 3, [0, 1])
    2
    >>> random_indexes(1, 3, [0, 2])
    1
    >>> random_indexes(1, 3, [1, 2])
    0
    """
    indexes = [pos for pos in range(size) if pos not in ignore]

    assert len(indexes) >= n
    np.random.shuffle(indexes)

    if n == 1:
        return indexes[0]
    else:
        return indexes[:n]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
