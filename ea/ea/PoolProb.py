import numpy as np
from numpy.random import rand
"""
This class allow us to have a pool of operation. When we ask the Pool one of
them, the selected operator is decided following a certain probability that it
is adjusted. The idea is to apply more times the operator with a better
improvement
"""


class PoolProb:
    def __init__(self, options):
        """
        Constructor
        :param options:to store (initially the probability is equals)
        :return:
        """
        self.options = options
        self.cumProb = []
        self.improvements = []
        self.count_calls = 0

        if len(options) > 0:
            size = len(options)
            prob = np.ones(size) / float(size)
            self.cumProb = prob.cumsum()
            self.improvements = dict(zip(options, [0] * size))
            self.total = dict(zip(options, [0] * size))

    def get_prob(self):
        return self.cumProb

    def get_new(self):
        """
        Get one of the options, following the probabilities
        :return: one of the stored object
        """
        if not self.options:
            raise Exception("There is no object")

        r = rand()
        position = self.cumProb.searchsorted(r)
        return self.options[position]

    def values(self):
        """
        Return the different values
        :return:
        """
        return self.options[:]

    def is_empty(self):
        counts = self.improvements.values()
        return np.all(counts == 0)

    def improvement(self, object, account, freq_update=0, minimum=0.15):
        """
        Received how much improvement this object has obtained (higher is better), it only update
        the method improvements

        :param object:
        :param account: improvement obtained (higher is better), must be >= 0
        :param freq_update: Frequency of improvements used to update the ranking
        :return: None
        """
        if account < 0:
            return

        if object not in self.improvements:
            raise Exception("Error, object not found in PoolProb")

        self.improvements[object] += account
        self.total[object] += 1
        self.count_calls += 1

        if self.count_calls >= freq_update and np.all(self.total.values() > 0):
            self.update_prob(minimum)
            # Restart the counter
            size = len(self.options)
            self.improvements = dict(zip(self.options, [0] * size))
            self.total = dict(zip(self.options, [0] * size))
            self.count_calls = 0

    def update_prob(self, minimum):
        """
        update the probabilities considering improvements value, following the equation
        prob[i] = Improvements[i]/TotalImprovements

        :return: None
        """
        # Complete the ranking
        if np.any(self.total.values() == 0):
            return

        improvements = np.array(
            self.improvements.values()) / self.total.values()
        nonzero = np.nonzero(~np.isnan(improvements))

        total = float(improvements[nonzero].sum())

        if total == 0:
            return

        prob_local = improvements[nonzero] / total
        dim = len(self.improvements)
        prob = np.zeros(dim)
        prob[nonzero] = prob_local
        # add a minimum value
        prob = np.maximum(minimum, prob)
        # Check that it its sum is 1
        total = float(prob.sum())
        prob = prob / total

        self.cumProb = prob.cumsum()
        # Init again the list and the count


#        self.improvements = dict(zip(self.options, [0] * size))
#        self.count_calls = 0


class PoolInc:
    def __init__(self, options):
        """
        Constructor
        :param options:to store (initially the probability is equals)
        :return:
        """
        self.options = options
        self.cumProb = []
        self.improvements = []
        self.count_calls = 0

        if len(options) > 0:
            size = len(options)
            prob = np.ones(size) / float(size)
            self.cumProb = prob.cumsum()
            self.improvements = dict(zip(options, [0.0] * size))
            self.count_total = dict(zip(options, [0.0] * size))

    def get_prob(self):
        return self.cumProb

    def get_new(self):
        """
        Get one of the options, following the probabilities
        :return: one of the stored object
        """
        if not self.options:
            raise Exception("There is no object")

        r = rand()
        position = self.cumProb.searchsorted(r)
        return self.options[position]

    def values(self):
        """
        Return the different values
        :return:
        """
        return self.options[:]

    def improvement(self, obj, account):
        """
        Received how much improvement this obj has obtained (higher is better), it only update
        the method improvements

        :param obj:
        :param account: improvement obtained (higher is better)
        :param freq_update: Frequency of run used to update the ranking
        :return: None
        """
        if obj not in self.improvements:
            raise Exception("Error, obj not found in PoolProb")

        self.count_total[obj] += 1

        if account > 0:
            self.improvements[obj] += 1

    def update_prob(self):
        """
        update the probabilities considering improvements value, following the equation
        prob[i] = Improvements[i]/TotalImprovements

        :return: None
        """
        size = len(self.options)

        # Complete the ranking
        improvements = np.array(self.improvements.values())
        totals = np.array(self.count_total.values())
        assert (np.all(totals > 0))
        ratio = improvements / totals + 0.01

        total_ratio = float(ratio.sum())
        prob = ratio / total_ratio

        self.cumProb = prob.cumsum()
