from typing import List

from typing import Callable
import numpy as np
import numpy.typing as npt

from .criteria import gini_index

class Splitter:
    """
    Splitter class used to create splits of the data
    """
    def __init__(self, X: npt.NDArray, Y: npt.NDArray, criterion: Callable[[npt.NDArray, npt.NDArray], float], presort: npt.NDArray|None = None) -> None:
        """
        Parameters
        ----------
        X : npt.NDArray
            The input features of the dataset

        Y : npt.NDArray
            The outcomes of the dataset

        criterion : Callable, optional
            Criteria function for calculating information gain,
            if None it uses the specified function in the start of splitter.py
        """
        self.features = X 
        self.outcomes = Y 

        self.n_features = len(self.features[0])
        self.criteria = criterion
        self.pre_sort = presort
        # self.constant_features = np.empty(len(self.features)) #TODO: not yet implemented
            
    def sort_feature(self, indices: List[int], feature: npt.NDArray) -> npt.NDArray:
        """
        Parameters
        ----------
        indices : List[int]
            A list of the indices which are to be sorted over
        
        feature: npt.NDArray
            A list containing the feature values that are to be sorted over
            
        Returns 
        -----------
        List[int]
            A list of the sorted indices 
        """
        return np.array(sorted(indices, key=lambda x: feature[x]), dtype=int)
    
    def test_split(self, left_indices: npt.NDArray, right_indices: npt.NDArray, feature) -> tuple:
        """
        Evaluates a split on two datasets

        Parameters
        ----------
        left_indices : List[int]
            indices of the left dataset

        right_indices : List[int]
            indices of the right dataset

        feature : int
            the current feature that is evaluated

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
        idx_split = [left_indices, right_indices]
        imp = [0.0, 0.0]
        crit = 0

        for i in range(len(idx_split)):
            n_outcomes = len(idx_split[i]) # number of outcomes in the given side
            # Make sure not to divide by 0 in criteria function
            if n_outcomes == 0:
                continue
            imp[i] = criteria(features[idx_split[i]], outcomes[idx_split[i]]) # calculate the impurity
            crit += imp[i] * (n_outcomes / len(indices)) # weight the impurity
        # calculate mean threshold as the mean of the last element in the left dataset and the first element in the right dataset 
        mean_thresh = np.mean([features[left_indices[-1], feature], features[right_indices[0], feature]])
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
        best_feature, best_threshold, best_score, best_imp = np.inf, np.inf, np.inf, [-1, -1]
        split = []

        # for all features
        for feature in range(self.n_features):
            current_feature = self.features[:, feature]
            if type(self.pre_sort) != npt.NDArray:
                sorted_index_list_feature = self.sort_feature(self.indices, current_feature)
            else:
                sorted_index_list_feature = self.pre_sort[self.indices, current_feature] #TODO: possible argsort list
             
            # loop over sorted feature list
            for i in range(len(sorted_index_list_feature) - 1):
                # Skip one iteration of the loop if the current threshold value is the same as the next in the feature list
                # Is this a speedup worth doing, it gives us a 10x speedup? do we know that splitting in two lists like this [0,0,0] and [1,1,1] is always better than [0,0] and [0,1,1,1]
                if current_feature[sorted_index_list_feature[i]] == current_feature[sorted_index_list_feature[i + 1]]:
                    continue 
                # Split the dataset
                left_indicies = sorted_index_list_feature[:i + 1]
                right_indicies = sorted_index_list_feature[i + 1:]

                crit, t_split, imp, threshold = self.test_split(left_indicies, right_indicies, feature) # test the split
                if crit < best_score:
                    # save the best split
                    best_feature, best_threshold, best_score, best_imp = feature, threshold, crit, imp # The index is given as the index of the first element of the right dataset 
                    split = t_split
        return split, best_threshold, best_feature, best_score, best_imp # return the best split

def main():
    lst = np.array([97.3, 28.9, 85.9, 91.9, 20.9, 26.5, 43.1, 88.5, 84.2])
    bin_lst = np.array([0, 0, 1, 0, 1, 0])
    splitter = Splitter(lst, bin_lst, gini_index)
    sorted = splitter.sort_feature(list(range(len(bin_lst))), bin_lst)
    x = 2 + 2

    for i in range(len(sorted)):
        if i < len(sorted) - 1 and bin_lst[sorted[i]] == bin_lst[sorted[i + 1]]:
            print("continuing from", i, "as bin_lst[sorted[i]] is ", bin_lst[sorted[i]], "and plus one is", bin_lst[sorted[i + 1]])
            continue 
        print("using", i, "as bin_lst[sorted[i]] is ", bin_lst[sorted[i]])


if __name__ == "__main__":
    main()