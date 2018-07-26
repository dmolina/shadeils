from numpy.random import permutation, rand
from numpy import concatenate
from ea.PoolProb import PoolInc
# pylint: disable=invalid-name

class EmptyCrossover(object):
    """
    This class wrap a simple crossover function with empties methods required for DE
    """
    def initrun(self, run, bounds, maxEvals, PS):
        """There is no code at the beginning of each run"""
        pass

    def apply(self, population, i, indexBest, F):
        """
        Applies the crossover function

        :population: from which apply the crossover
        :i: current position.
        :indexBest: index of best solution.
        :F: parameter F
        """
        pass

    def stats(self):
        """There is special statistics"""
        return ""

    def set_previous_improvement(self, account):
        return

class SimpleCrossover(EmptyCrossover):
    """
    This class wrap a simple crossover function, making easier to use directly a
    function with no special data or statistics
    """
    def __init__(self, function):
        self.function = function

    def apply(self, population, i, indexBest, F):
        """
        Applies the crossover function

        :population: from which apply the crossover
        :i: current position.
        :indexBest: index of best solution.
        :F: parameter F
        """
        return self.function(population, i, indexBest, F)

class RhctCrossover(EmptyCrossover):
    """
    This class implements the experimental Rhct (from Miguel Ortiz)
    """
    def __init__(self):
        self.T = 0
        self.T0 = 0.95
        self.Tf = 0.05
        self.Tchange = 0

    def initrun(self, run, bounds, maxEvals, PS):
        """
        Init the crossover information
        """
        if run == 0:
            self.T = self.T0
            self.Tchange = (self.T0 - self.Tf) / (float(maxEvals) / PS)
        else:
            self.T -= self.Tchange

        self.contr = 0
        self.contctb = 0

    def apply(self, population, i, bestIndex, F):
        """
        Implements the strange crossoverRhct
        """
        sizePopulation = population.shape[0]
        (c, a, b) = permutation(sizePopulation)[:3]
        current = population[i]
        best = population[bestIndex]
        F2 = 0.9
        r = rand()

        if r < self.T:
            noisyVector = population[c] + F2 * (population[a] - population[b])
            self.contr += 1
        else:
            noisyVector = current + F * (best - population[i]) +\
                            F * (population[a] - population[b])
            self.contctb += 1

        return noisyVector

    def stats(self):
        """Show the ratio of each crossover application"""
        return "contr=%.2f  contctb=%.2f" % (self.contr, self.contctb)

class SADECrossover(EmptyCrossover):
    def __init__(self, LP=50):
        crossovers = [classicalBinFunction, classicalTwoBinFunction, classicalBestFunction, currentToRand]
        self.pool = PoolInc(crossovers)
        self.LP = LP
        self.PS = 0
        self.count_calls = 0

    def initrun(self, run, bounds, maxEvals, PS):
        self.PS = PS
        self.count_calls = 0
        self.gene = 0

    def apply(self, population, i, bestIndex, F):
        crossover = self.pool.get_new()
        self.last_crossover = crossover
        return crossover(population, i, bestIndex, F)

    def stats(self):
        cumprob = self.pool.get_prob()
        prob = cumprob - concatenate(([0], cumprob[0:-1]))
        return ' '.join(map(str, prob))
    
    def set_previous_improvement(self, improvement):
        """Update the pool command"""
        self.pool.improvement(self.last_crossover, improvement)
        self.count_calls += 1

        if self.count_calls == self.PS:
            self.count_calls = 0
            self.gene += 1

            if self.gene >= self.LP:
               self.pool.update_prob()

def classicalBinFunction(population, i, bestIndex, F):
    """
    Implements the classical crossover function for DE
    """
    (c, a, b) = permutation(len(population))[:3]
    noisyVector = population[c] + F * (population[a] - population[b])
    return noisyVector

def classicalTwoBinFunction(population, i, bestIndex, F):
    """
    Implements the classical crossover function for DE
    :param population: population
    :param i: current
    :param bestIndex: best global
    :param F: parameter
    """
    size = population.shape[0]

    (c, a, b, r3, r4) = permutation(size)[:5]
    noisyVector = population[c] + F * (population[a] - population[b])  + F * (population[r3] - population[r4])
    return noisyVector

def currentToRand(population, i, bestIndex, F):
    """
    Crossover with the DE/current-to-rand/1
    :param population: of solution
    :param i: current solution
    :param bestIndex: best current solution
    :param F: parameter
    :return: vector results
    """
    size = len(population)
    (r1, r2, r3) = permutation(size)[:3]
    k = rand()
    noisyVector = population[i]+k*(population[r1]-population[i])\
                               +F*(population[r2]-population[r3])

    return noisyVector


def classicalBestFunction(population, i, bestIndex, F):
    """
    Implements the classical DE/best/ mutation
    """
    (a, b) = permutation(len(population))[:2]
    noisyVector = population[bestIndex] + F * (population[a] - population[b])
    return noisyVector

def randToBestFunction(population, i, bestIndex, F):
    """
    Implements the DE/rand-to-best/2/bin

    :param population: of solutions
    :param i: iteration
    :param bestIndex: index of current best
    :param F: parameter F (ratio)
    :return: A vector with
    """
    size = len(population)
    (r1, r2, r3, r4) = permutation(size)[:4]
    noisy_vector = population[i]+F*(population[bestIndex]-population[i])\
                               +F*(population[r1]-population[r2])\
                               +F*(population[r3]-population[r4])
    return noisy_vector

#def SaDE_init(generation, bounds, maxEvals, PS):
#    global pool
#    crossovers = [classicalBinFunction, classicalTwoBinFunction, currentToRand, randToBestFunction]
#    pool = PoolCross(crossovers)
#
#def SaDECrossover(population, i, best, F):
#    global pool
#    crossover = pool.get_new()
#    return crossover(population, i, best, F)
#
#D = 10
#region_size = 0
#
#def initRegion(generation, bounds, maxevals, PS):
#    # total_generations=float(maxevals)/PS
#    global region_size
#    region_size = (bounds[1] - bounds[0]) / D
#
#
#def sameRegion(a, b):
#    return (np.floor(a / region_size) == np.floor(b / region_size)).all()
#
#
#def regionBestFunction(population, i, bestIndex, F):
#    """
#    Implements a new crossover using region
#    """
#    sizePopulation = len(population)
#    (a,b)=permutation(sizePopulation)[:2]
#    while sameRegion(a, b):
#        b = np.random.randint(0, sizePopulation)
#
#    noisyVector = population[bestIndex] + F * (population[a] - population[b])
#    return noisyVector
#
#
#def regionCurrentFunction(population, i, bestIndex, F):
#    """
#    Implements a new crossover using region
#    """
#    sizePopulation = len(population)
#    (a,b)=permutation(sizePopulation)[:2]
#
#    while sameRegion(a, b):
#        b = np.random.randint(0, sizePopulation)
#
#    noisyVector = population[bestIndex] + F * (population[a] - population[b])
#    return noisyVector


#def getCrossoverSimpleBestRegion():
#    """
#    This method returns three function: 
#
#    @initCrossover: function to init the variables
#    @crossover: function to make actually the crossover
#    @stats: functions to show statistics
#    """
#    return (initRegion, regionBestFunction, no_stats_function)
