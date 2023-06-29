from decision_tree.criteria import *
import numpy as np
from typing import Callable

criteria = gini_index

class Splitter():

    def __init__(self, data: np.dtype, max_features: int = None, max_samples: int = None, sample_weight: list = None, criterion: Callable = None) -> None:
        self.features = data[:, :-1] # all but the last column for each data input
        self.outcomes = data[:, -1] # the last column only

        self.n_features = len(self.features[0])

        self.max_features = max_features
        self.max_samples = max_samples
        if criterion:
            self.criteria = criterion
        else:
            self.criteria = criteria
        self.constant_features = np.empty(len(self.features))
    
    def test_split(self, index, threshold):
        start, end = self.start, self.end
        split = [[], []]
        idx = 0
        for i, row in enumerate(self.features[start:end]):
            # if the value of a given row is below the threshold then add it to the left side
            if row[index] < threshold:
                split[0].append(self.outcomes[start + i])
                idx += 1 # Used to find where the split is made
            # else to the right side
            else:
                split[1].append(self.outcomes[start + i])
        crit = 0
        for i in range(len(split)):
            if len(split[i]) == 0:
                continue
            crit += self.criteria(split[i]) * (len(split[i]) / len(self.features[start:end]))
        return crit, (start + idx)
    
    def get_split(self, start, end):
        self.start, self.end = start, end 
        best_index, best_threshold, best_score = np.inf, np.inf, np.inf
        split = 0
        # for all features
        for index in range(self.n_features):

            # For all samples within the start and end
            for row in self.features[start:end]:
                crit, t_split = self.test_split(index, row[index])
                if crit < best_score:
                    best_index, best_threshold, best_score = index, row[index], crit
                    split = t_split
        print('X%d < %.3f Gini=%.3f' % ((best_index), best_threshold, best_score))
        return split, best_threshold, best_index, best_score



