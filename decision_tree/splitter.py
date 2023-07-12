import sys
from typing import List

# On William
from criteria import *

# On Simon
#from decision_tree.criteria import *

from typing import Callable
import numpy as np
import numpy.typing as npt

class Splitter():
    """
    Splitter function used to create splits of the data
    """
    def __init__(self, X: npt.NDArray, Y: npt.NDArray, criterion: Callable) -> None:
        """
        Parameters
        ----------
        data : np.dtype
            the data used for the tree entire tree generation

            maybe this should be split into features and outcome before hand so we make it explicit what the 
            features are, and what the outcomes are?
        criterion : Callable, optional
            Criteria function for calculating information gain,
            if None it uses the specified function in the start of splitter.py
        """
        self.features = X 
        self.outcomes = Y 

        self.n_features = len(self.features[0])

        self.criteria = criterion
        self.constant_features = np.empty(len(self.features)) #TODO: not yet implemented
    
    def test_split(self, index: int, threshold: float) -> tuple:
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
        float
            the mean threshold of the split feature and the closest neighbour with a smaller value.
        """        
        features = self.features
        outcomes = self.outcomes
        criteria = self.criteria
        indices = self.indices
        closets_neighbour = [np.inf, 0]
        idx_split = [[], []]
        imp = [0, 0]
        for idx in indices:
            closest_dist, _ = closets_neighbour
            # if the value of a given row is below the threshold then add it to the left side
            other_val = features[idx, index]
            if other_val < threshold:
                # Calculate the distance between this and the threshold
                distance = threshold - other_val
                if distance < closest_dist:
                    closets_neighbour = [distance, idx] # store the closest neighbour on the left side
                idx_split[0].append(idx)

            # else to the right side
            else:
                # distance = other_val - threshold
                # if distance < closest_dist:
                #     closets_neighbour = [distance, idx]
                idx_split[1].append(idx)
        crit = 0
        for i in range(len(idx_split)):
            n_outcomes = len(idx_split[i]) # number of outcomes in the given side
            # Make sure not to divide by 0 in criteria function
            if n_outcomes == 0:
                continue
            imp[i] = criteria(features[idx_split[i]], outcomes[idx_split[i]]) # calculate the impurity
            crit += imp[i] * (n_outcomes / len(self.features[indices])) # weight the impurity
        _, closest_idx = closets_neighbour
        mean_thresh = np.mean([threshold, features[closest_idx, index]])
        return crit, idx_split, imp, mean_thresh
    

    def get_split(self, indices: List[int]) -> tuple:
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
        best_index, best_threshold, best_score, best_imp = np.inf, np.inf, np.inf, [-1, -1]
        split = []
        # for all features
        for index in range(self.n_features):

            # For all samples in the node
            for row in self.features[indices]:
                crit, t_split, imp, threshold = self.test_split(index, row[index]) # test the split
                if crit < best_score:
                    best_index, best_threshold, best_score, best_imp = index, threshold, crit, imp # save the best split
                    split = t_split
        return split, best_threshold, best_index, best_score, best_imp # return the best split



