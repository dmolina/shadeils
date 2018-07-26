"""
This program contains the SHADE algorithm, proposed in [1].

It is a DE that adapts its F and CR parameters, it uses a memory of solutions,
and a new crossover operator.

Tanabe, R.; Fukunaga, A., "Success-history based parameter adaptation
for Differential Evolution," Evolutionary Computation (CEC), 2013 IEEE
Congress on , vol., no., pp.71,78, 20-23 June 2013
doi:10.1109/CEC.2013.655755510.1109/CEC.2013.6557555
"""
import numpy as np

from DE import EAresult, random_population, get_experiments_file, random_indexes
import math

def improve(fun, run_info, dimension, check_evals, name_output=None, replace=True, popsize=100, H=100, population=None, population_fitness=None, initial_solution=None, MemF=None, MemCR=None):
    """
    It applies the DE elements.

    Params
    ------
    fun function of the problem to optimize.

    run_info is a dictionary with the following information:
         lower: double lower bounds
         upper: double upper bounds
         threshold: minimum optim value
    
    dimension of the problem.
    max_evals maximum_evaluations_numbers.
    name_output name of the output file
    run number of evaluations
    replace replace the file
    debug show the debug info if it is true
    PS population size
    """
    assert isinstance(dimension, int), 'dimension should be integer'
    assert (dimension > 0), 'dimension must be positive'

    final, fid = get_experiments_file(name_output, replace)

    if final is not None:
        return final
    
    for attr in ['lower', 'upper', 'threshold', 'best']:
         assert attr in run_info.keys(), "'{}' info not provided for benchmark".format(attr)

    # Added in a array the max evaluations
    if not isinstance(check_evals, list):
        check_evals = [check_evals]

    domain = (run_info['lower'], run_info['upper'])
    fun_best = run_info['best']
    maxEval = check_evals[-1]
    check_eval = check_evals.pop()

    if population is None:
        population = random_population(domain, dimension, popsize)
    else:
        popsize = population.shape[0]

    if initial_solution is not None:
        population[0] = initial_solution

    if population_fitness is None:
        population_fitness = np.array([fun(ind) for ind in population])
        currentEval = popsize
    else:
        currentEval = 0

    # Init memory with population
    memory = population.tolist()

    # Init F and CR
    memorySize = popsize*2

    if MemF is None:
        MemF = np.ones(H)*0.5

    if MemCR is None:
        MemCR = np.ones(H)*0.5
        
    k = 0
    pmin = 2.0/popsize

    while currentEval < maxEval:
        SCR = []
        SF = []
        F = np.zeros(popsize)
        CR = np.zeros(popsize)
        u = np.zeros((popsize, dimension))
        best_fitness = np.min(population_fitness)
        numEvalFound = currentEval

        for (i, xi) in enumerate(population):
            # Getting F and CR for that solution
            index_H = np.random.randint(0, H)
            meanF = MemF[index_H]
            meanCR = MemCR[index_H]
            Fi = np.random.normal(meanF, 0.1)
            CRi = np.random.normal(meanCR, 0.1)
            p = np.random.rand()*(0.2-pmin)+pmin

            # Get two random values
            r1 = random_indexes(1, popsize, ignore=[i])
            # Get the second from the memory
            r2 = random_indexes(1, len(memory), ignore=[i, r1])
            xr1 = population[r1]
            xr2 = memory[r2]
            # Get one of the p best values
            maxbest = int(p*popsize)
            bests = np.argsort(population_fitness)[:maxbest]
            pbest = np.random.choice(bests)
            xbest = population[pbest]
            # Mutation
            v = xi + Fi*(xbest - xi) + Fi*(xr1-xr2)
            # Special clipping
            v = shade_clip(domain, v, xi)
            # Crossover
            idxchange = np.random.rand(dimension) < CRi
            u[i] = np.copy(xi)
            u[i, idxchange] = v[idxchange]
            F[i] = Fi
            CR[i] = CRi

        # Update population and SF, SCR
        weights = []
        
        for i, fitness in enumerate(population_fitness):
            fitness_u = fun(u[i])

            if math.isnan(fitness_u):
                print(i)
                print(domain)
                print(u[i])
                print(fitness_u)
                
            assert not math.isnan(fitness_u)

            if fitness_u <= fitness:
                # Add to memory
                if fitness_u < fitness:
                    memory.append(population[i])
                    SF.append(F[i])
                    SCR.append(CR[i])
                    weights.append(fitness-fitness_u)

                if (fitness_u < best_fitness):
                    best_fitness = fitness_u
                    numEvalFound = currentEval
                    
                population[i] = u[i]
                population_fitness[i] = fitness_u

        currentEval += popsize
        # Check the memory
        memory = limit_memory(memory, memorySize)
                
        # Update MemCR and MemF
        if len(SCR) > 0 and len(SF) > 0:
            Fnew, CRnew = update_FCR(SF, SCR, weights)
            MemF[k] = Fnew
            MemCR[k] = CRnew
            k = (k + 1) % H

    if fid is not None and currentEval >= check_eval:
        bestFitness = np.min(population_fitness)
        print("bestFitness: {}".format(bestFitness))
        fid.write("[%.0e]: %e,%d\n" %(check_eval, abs(bestFitness-fun_best), numEvalFound))
        fid.flush()

        if check_evals:
            check_eval = check_evals.pop(0)

    if fid is not None:
        fid.close()

    bestIndex = np.argmin(population_fitness)
    print("SHADE Mean[F,CR]: ({0:.2f}, {1:.2f})".format(MemF.mean(), MemCR.mean()))
          
    return EAresult(fitness=population_fitness[bestIndex], solution=population[bestIndex], evaluations=numEvalFound), bestIndex

def limit_memory(memory, memorySize):
    """
    Limit the memory to  the memorySize
    """
    memory = np.array(memory)
    
    if len(memory) > memorySize:
        indexes = np.random.permutation(len(memory))[:memorySize]
        memory = memory[indexes]

    return memory.tolist()

def update_FCR(SF, SCR, improvements):
    """
    Update the new F and CR using the new Fs and CRs, and its improvements
    """
    total = np.sum(improvements)
    assert total > 0
    weights = improvements/total

    Fnew = np.sum(weights*SF*SF)/np.sum(weights*SF)
    Fnew = np.clip(Fnew, 0, 1)
    CRnew = np.sum(weights*SCR)
    CRnew = np.clip(CRnew, 0, 1)

    return Fnew, CRnew

def shade_clip(domain, solution, original):
    lower = domain[0]
    upper = domain[1]
    clip_sol = np.clip(solution, lower, upper)

    if np.all(solution == clip_sol):
        return solution
    
    idx_lowest = (solution < lower)
    solution[idx_lowest] = (original[idx_lowest]+lower)/2.0
    idx_upper = (solution > upper)
    solution[idx_upper] = (original[idx_upper]+upper)/2.0
    return solution
