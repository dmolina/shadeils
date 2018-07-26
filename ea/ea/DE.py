"""
This program contains the DE versions to test. 

It uses a simple benchmark of 13 functions, implemented in Python, from the
article.

[1] Xin Yao, Yong Liu (1999). Evolutionary Programming Made Faster. IEEE
Transaction on Evolutionary Computation, 3:2, 82-102.
"""
import numpy as np
# To ignore numpy errors:
#     pylint: disable=E1101
from ea import DEcrossover
from ea.parameters import get_parameter, get_value, parameter_result
import types
import six

from collections import namedtuple
EAresult = namedtuple("EAresult", "fitness solution evaluations")

import os


def random_population(lower, upper, dimension, size):
    return np.random.uniform(lower, upper, dimension * size).reshape(
        (dimension, size))


def check_file(name_output, replace=True):
    if name_output is None:
        fid = None
    else:
        # if it replaced it only return the last value
        if not replace and os.path.isfile(name_output):
            fin = open(name_output, 'rb')
            lines = fin.readlines()

            if lines:
                (bestSolutionFitness, bestSol, bestEval,
                 evaluations) = lines[-1].split(',')
                return EAresult(
                    fitness=bestSolutionFitness,
                    solution=bestSol,
                    evaluations=evaluations)

        fid = open(name_output, 'w')

    return (fid, )


def DE(fun,
       run_info,
       dimension,
       max_evals,
       name_output=None,
       run=25,
       replace=True,
       debug=True,
       F=0.6,
       CR=0.9,
       popsize=60,
       crossoverFunction=DEcrossover.classicalBinFunction,
       population=None,
       initial_solution=None):
    """
    It applies the DE elements.

    Params
    ------
    fun function of the problem to optimize.

    run_info is a dictionary with the following information:
         lower: double lower bounds
         upper: double upper bounds
         threshold: minimum optima
         
    dimension of the problem.
    max_evals maximum_evaluations_numbers.
    name_output name of the output file
    run number of evaluations
    replace replace the file
    debug show the debug info if it is true
    F mutation value (a fixed F value, or 'n', meaning normal(0.5, 0.3), or 'a' automatic with a LP. Default 0.9.
    CR mutation value (a fixed F value, or 'n', meaning normal(0.9, 0.1), or 'a' automatic with a LP. Default 0.9.
    PS population size
    crossoverFunction crossover function to use
    """
    assert isinstance(dimension, int), 'dimension should be integer'
    assert (dimension > 0), 'dimension must be positive'

    for attr in ['lower', 'upper', 'threshold', 'best']:
        assert attr in run_info.keys(
        ), "'{}' info not provided for benchmark".format(attr)

    # Added in a array the max evaluations
    if not isinstance(max_evals, list):
        max_evals = [max_evals]

    lower = run_info['lower']
    upper = run_info['upper']
    threshold = run_info['threshold']
    fun_best = run_info['best']

    PS = popsize

    normal_values = {'F': (0.5, 0.3), 'CR': (0.9, 0.1)}

    maxEval = max_evals[-1]

    if name_output is None:
        fid = None
    else:
        # if it replaced it only return the last value
        if not replace and os.path.isfile(name_output):
            fin = open(name_output, 'rb')
            lines = fin.readlines()

            if lines:
                (bestSolutionFitness, bestSol, bestEval,
                 evaluations) = lines[-1].split(',')
                return EAresult(
                    fitness=bestSolutionFitness,
                    solution=bestSol,
                    evaluations=evaluations)

        fid = open(name_output, 'w')

    if isinstance(crossoverFunction, types.FunctionType):
        crossover = DEcrossover.SimpleCrossover(crossoverFunction)
    else:
        crossover = crossoverFunction

    currentEval = 0
    bestSolutionFitness = np.Inf

    for numrun in range(1, run + 1):
        check_evals = max_evals[:]
        check_eval = check_evals.pop(0)
        valueMutation = get_parameter('F', F, normal_values['F'], PS)
        probabilityRecombination = get_parameter('CR', CR, normal_values['CR'],
                                                 PS)

        bestSolution = np.zeros(dimension)
        bestSolutionFitness = np.Inf
        numEvalFound = 0
        sizePopulation = PS
        crossover.initrun(numrun, (lower, upper), maxEval, PS)

        currentEval = 0

        # Start generating the initial population
        i = 0
        indexBest = 0

        if population is not None:
            sizePopulation = population.shape[0]
        else:
            population = np.zeros((sizePopulation, dimension))

            for i in range(sizePopulation):
                population[i, :] = np.random.uniform(lower, upper, dimension)

        if initial_solution is not None:
            population[0, :] = initial_solution

        populationFitness = np.zeros(sizePopulation)

        for i in range(sizePopulation):
            populationFitness[i] = fun(population[i, :])
            currentEval += 1

            if bestSolutionFitness > populationFitness[i]:
                bestSolutionFitness = populationFitness[i]

                bestSolution[:] = population[i, :]

                indexBest = i
                numEvalFound = currentEval

                msg = "Best solution Find: %e at %d" % (bestSolutionFitness,
                                                        currentEval)
                dprint(msg, debug)

        while not shouldEnd(currentEval, maxEval, bestSolutionFitness,
                            fun_best, threshold):
            # Mutate the current population
            trialVector = np.zeros((sizePopulation, dimension))
            trialVectorFitness = np.zeros(sizePopulation)

            for i in range(sizePopulation):
                noisyVector = crossover.apply(population, i, indexBest,
                                              get_value(valueMutation))
                noisyVector = np.clip(noisyVector, lower, upper)

                # Obtain the next solution considering probabilityRecombination
                probability = get_value(probabilityRecombination)
                changed = (np.random.rand(dimension) < probability)
                trialVector[
                    i] = noisyVector * changed + population[i] * np.invert(
                        changed)
                trialVectorFitness[i] = fun(trialVector[i])
                currentEval += 1
                successful = trialVectorFitness[i] < populationFitness[i]
                improvement = populationFitness[i] - trialVectorFitness[i]
                crossover.set_previous_improvement(improvement)
                parameter_result(valueMutation, successful)
                parameter_result(probabilityRecombination, successful)

                if successful:
                    population[i, :] = trialVector[i, :]
                    populationFitness[i] = trialVectorFitness[i]

                if populationFitness[i] < bestSolutionFitness:

                    bestSolution[:] = population[i, :]
                    bestSolutionFitness = populationFitness[i]

                    indexBest = i
                    numEvalFound = currentEval
                    position = currentEval - sizePopulation + indexBest
                    dprint("Best solution Find: %e at %d" %
                           (bestSolutionFitness, position), debug)

            if fid is not None and currentEval >= check_eval:
                fid.write("[%.0e]: %e,%d\n" %
                          (check_eval, abs(bestSolutionFitness - threshold),
                           numEvalFound))
                fid.flush()

                if check_evals:
                    check_eval = check_evals.pop(0)

            # Generation

        # Show the best solution ever
        msg = "The best solution is: %e in %d evaluation %s" % (
            bestSolutionFitness, numEvalFound, crossover.stats())

        dprint(msg, debug)

        if fid is not None:
            fid.write("%s\n" % msg)
            fid.write("%e,%s,%d,%d\n" % (abs(bestSolutionFitness - threshold),
                                         ' '.join(map(str, bestSolution)),
                                         numEvalFound, currentEval))
            fid.flush()

    if fid is not None:
        fid.close()

    return EAresult(
        fitness=bestSolutionFitness,
        solution=bestSolution,
        evaluations=currentEval)


def initializeVariables(F_n):
    """
    @brief Initialize all variable that we need
    @param semilla We use semilla to initialize rand
    @param F_n The function that we are checking 
    @return upper The maximum value possible
    @return minimum The minimum value possible
    """
    bounds = [(-100, 100), (-10, 10), (-100, 100), (-100, 100), (-30, 30),
              (-100, 100), (-1.28, 1.28), (-500, 500), (-5.12, 5.12),
              (-32, 32), (-600, 600), (-50, 50), (-50, 50)]

    if F_n == 8:
        objectiveValue = -12569.5
    else:
        objectiveValue = 1e-8

    (lower, upper) = bounds[F_n - 1]
    return (lower, upper, objectiveValue)


def dprint(msg, debug):
    """
    Print the message only if debug is True
    @param msg message
    @param debug debug value
    """
    if debug:
        six.print_(msg)


def shouldEnd(currentEval, maxEval, bestSolutionFitness, objectiveValue,
              threshold):
    """
    Return if the algorithm should end.

    @param currentEval
    @param maxEval 
    @param bestSolutionFitness
    @param objectiveValu
    @param maxEval
    @param bestSolutionFitness
    @param objectiveValue
    @param threshold -- threshold to obtain the objectiveValue
    """
    if currentEval >= maxEval:
        return True
    elif abs(bestSolutionFitness - objectiveValue) <= threshold:
        return True
    else:
        return False
