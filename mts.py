"""
Implements the MTS-LS1 indicated in MTS 
http://sci2s.ugr.es/EAMHCO/pdfs/contributionsCEC08/tseng08mts.pdf 
Lin-Yu Tseng; Chun Chen, "Multiple trajectory search for Large Scale Global Optimization," Evolutionary Computation, 2008. CEC 2008. (IEEE World Congress on Computational Intelligence). IEEE Congress on , vol., no., pp.3052,3059, 1-6 June 2008
doi: 10.1109/CEC.2008.4631210
and used by MOS
"""
from numpy import clip, zeros, flatnonzero, copy

from numpy.random import permutation

from ea.DE import EAresult

from functools import partial

def _mtsls_improve_dim(function, sol, best_fitness, i, check, SR):
    newsol = copy(sol)
    newsol[i] -= SR[i]
    newsol = check(newsol)
    fitness_newsol = function(newsol)
    evals = 1

    if fitness_newsol < best_fitness:
            best_fitness = fitness_newsol
            sol = newsol
    elif fitness_newsol > best_fitness:
            newsol = copy(sol)
            newsol[i] += 0.5*SR[i]
            newsol = check(newsol)
            fitness_newsol = function(newsol)
            evals += 1

            if (fitness_newsol < best_fitness):
                best_fitness = fitness_newsol
                sol = newsol

    return EAresult(solution=sol, fitness=best_fitness, evaluations=evals)

def mtsls(function, sol, fitness, lower, upper, maxevals, SR):
    """
    Implements the MTS LS

    parameters
    :function: to optimize.
    :sol: solution to improves.
    :lower: lower domain.
    :upper: upper domain.
    :maxevals: maximum evaluations.
    :SR: step size.
    :improved: indicate the ratio of previous improvement.
    """
    dim = len(sol)

    improved_dim = zeros(dim, dtype=bool)
    check = partial(clip, a_min=lower, a_max=upper)
    current_best = EAresult(solution=sol, fitness=fitness, evaluations=0)
    totalevals = 0

    improvement = zeros(dim)

    if totalevals < maxevals:
        dim_sorted = permutation(dim)

        for i in dim_sorted:
            result = _mtsls_improve_dim(function, current_best.solution, current_best.fitness, i, check, SR)
            totalevals += result.evaluations
            improve = max(current_best.fitness - result.fitness, 0)
            improvement[i] = improve

            if improve:
                improved_dim[i] = True
                current_best = result
            else:
                SR[i] /= 2
  

        dim_sorted = improvement.argsort()[::-1]
        d = 0

    while totalevals < maxevals:
        i = dim_sorted[d]
        result = _mtsls_improve_dim(function, current_best.solution, current_best.fitness, i, check, SR)
        totalevals += result.evaluations
        improve = max(current_best.fitness - result.fitness, 0)
        improvement[i] = improve
        next_d = (d+1)%dim
        next_i = dim_sorted[next_d]

        if improve:
            improved_dim[i] = True
            current_best = result

            if improvement[i] < improvement[next_i]:
                dim_sorted = improvement.argsort()[::-1]
        else:
            SR[i] /= 2
            d = next_d

     # Check lower value
    initial_SR = 0.2*(upper-lower)
    SR[SR < 1e-15] = initial_SR

    return current_best, SR
