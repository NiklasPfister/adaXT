import numpy as np
from typing import Callable
import numpy as np

class Splitter():
    """
    Splitter function used to create splits of the data
    """
    def __init__(self, data: np.dtype, criterion: Callable) -> None:
        """
        Parameters
        ----------
        data : np.dtype
            the data used for the tree entire tree generation
        criterion : Callable, optional
            Criteria function for calculating information gain,
            if None it uses the specified function in the start of splitter.py
        """
        self.features = data[:, :-1] # all but the last column for each data input
        self.outcomes = data[:, -1] # the last column only

        self.n_features = len(self.features[0])

        self.criteria = criterion
        self.constant_features = np.empty(len(self.features)) #TODO: not yet implemented
    
    def test_split(self, index: int, threshold: float) -> list[list]:
        """
        Creates a split on the given feature index with the given threshold

        Parameters
        ----------
        index : int
            index of the feature to split on
        threshold : float
            the threshold value to split on

        Returns
        -------
        float
            the information gain given the criteria function
        list[list]
            first index is the list of indices split to the left, second index is the list of indices split to the right
        list[float]
            the impurity of the left side followed by impurity of the right side
        """        
        indices = self.indices
        idx_split = [[], []]
        imp = [0, 0]
        for idx in indices:
            # if the value of a given row is below the threshold then add it to the left side
            if self.features[idx, index] < threshold:
                idx_split[0].append(idx)

            # else to the right side
            else:
                idx_split[1].append(idx)
        crit = 0
        for i in range(len(idx_split)):
            n_outcomes = len(idx_split[i]) # number of outcomes in the given side
            # Make sure not to divide by 0 in criteria function
            if n_outcomes == 0:
                continue
            imp[i] = self.criteria(self.outcomes[idx_split[i]]) # calculate the impurity
            crit += self.criteria(self.outcomes[idx_split[i]]) * (n_outcomes / len(self.features[indices])) # weight the impurity
        return crit, idx_split, imp
    
    """ TODO: Currently getting the same splits, however there are differences in the thresholds between our implementation and sklearn.
        Reason for this is not clear atm
    """
    def get_split(self, indices: list[int]) -> tuple:
        """
        gets the best split given the criteria function

        Parameters
        ----------
        indices : list[int]
            indices of all rows to take into account when splitting

        Returns
        -------
        list[list]
            first index is the list of indices split to the left, second index is the list of indices split to the right
        float
            the best threshold value for the split
        int
            the feature index splitting on
        float
            the best score of a split
        list[float]
            list of 2 elements, impurity of left child followed by right child
        """
        self.indices = indices
        best_index, best_threshold, best_score = np.inf, np.inf, np.inf
        split = []
        # for all features
        for index in range(self.n_features):

            # For all samples in the node
            for row in self.features[indices]:
                crit, t_split, imp = self.test_split(index, row[index]) # test the split
                if crit < best_score:
                    best_index, best_threshold, best_score, best_imp = index, row[index], crit, imp # save the best split
                    split = t_split
        return split, best_threshold, best_index, best_score, best_imp # return the best split



