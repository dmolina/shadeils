from ea import DE, DEcrossover
from math import sqrt
import tempfile

import pytest

from mock import Mock

from numpy.random import rand, seed

from numpy import copy, all

def sphere(x):
    return sqrt((x*x).sum())

@pytest.fixture
def output():
    tmpfile = tempfile.NamedTemporaryFile("w", delete=True)
    return tmpfile.name

@pytest.fixture
def funinfo():
    return {'lower': -5, 'upper': 5, 'threshold': 0, 'best': 0}

@pytest.mark.parametrize("dim", 
[2]#, 5, 10]
)
def test_DE(output, funinfo, dim):
    evals = dim*1000
    (fitness, best, evals) = DE.DE(sphere, funinfo, dim, evals, name_output=output, debug=False)
    assert sphere(best) == fitness

@pytest.mark.parametrize("dim", 
[2, 5]
)
def test_evals(output, funinfo, dim):
    fun_fitness = Mock()
    fun_fitness.side_effect = sphere
    evals = dim*1000
    assert fun_fitness.call_count == 0
    (fitness, best, realevals) = DE.DE(fun_fitness, funinfo, dim, evals, name_output=output, run=1, debug=False)
    assert sphere(best) == fitness
    assert realevals - evals < 60
    assert realevals == fun_fitness.call_count

@pytest.mark.de
@pytest.mark.parametrize("dim", 
[2, 5]
)
def test_evals(output, funinfo, dim):
    evals = dim*100
    popsize = 10
    population = (rand(dim*popsize)*10-5).reshape((popsize, dim))
    previous_population = copy(population)
    result = DE.DE(sphere, funinfo, dim, evals, name_output=output, run=1, debug=False, population=population)
    assert sphere(result.solution) == result.fitness
    
    # Check the population improves
    for i in range(popsize):
        assert not all(population[i] == previous_population[i])
        assert all(previous_population[i] == previous_population[i])
        assert sphere(population[i]) <= sphere(previous_population[i])

@pytest.mark.de
@pytest.mark.parametrize("dim", 
[2, 5]
)
def test_evals_continue(output, funinfo, dim):
    evals = dim*100
    popsize = 10
    first_population = (rand(dim*popsize)*10-5).reshape((popsize, dim))
    myseed = 12345679
    seed(myseed)
    population = copy(first_population)
    total_evals = 0
    result = DE.DE(sphere, funinfo, dim, evals, name_output=output, run=1, debug=False, population=population)
    assert sphere(result.solution) == result.fitness
    total_evals += result.evaluations
    second_population = copy(population)
    result = DE.DE(sphere, funinfo, dim, evals, name_output=output, run=1, debug=False, population=population)
    total_evals += result.evaluations

 # Check the population improves
    for i in range(popsize):
        assert not all(first_population[i] == second_population[i])
        assert not all(population[i] == second_population[i])
        assert sphere(first_population[i]) >= sphere(second_population[i])
        assert sphere(second_population[i]) >= sphere(population[i])

    # Check again with the double of evaluations
    final_population = copy(population)
    population = copy(first_population)
    seed(myseed)
    result = DE.DE(sphere, funinfo, dim, total_evals-popsize, name_output=output, run=1, debug=False, population=population)
    assert sphere(result.solution) == result.fitness

    for i in range(popsize):
        assert all(population[i] == final_population[i])
