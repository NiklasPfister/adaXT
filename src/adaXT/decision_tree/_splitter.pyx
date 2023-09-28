# cython: profile=True

from typing import List

import numpy as np
cimport numpy as cnp
cnp.import_array()
import numpy.typing as npt
from cython cimport boundscheck, wraparound, profile
from ._func_wrapper cimport FuncWrapper
from libc.stdlib cimport malloc, free
from numpy.math cimport INFINITY

cdef class Splitter:   
    """
    Splitter class used to create splits of the data
    """
    def __init__(self, X: npt.NDArray, Y: npt.NDArray, FuncWrapper criterion, presort: npt.NDArray|None = None) -> None:
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
        self.n_class = len(np.unique(Y))
        self.class_labels = <double *> malloc(sizeof(double) * self.n_class)
        self.n_in_class = <int *> malloc(sizeof(int) * self.n_class)
        # self.constant_features = np.empty(len(self.features)) #TODO: not yet implemented
    
    def free_c_lists(self):
        free(self.class_labels)
        free(self.n_in_class)
    
    cdef (double, double, double, double) test_split(self, int[::1] left_indices, int[::1] right_indices, int feature):
        """
        Evaluates a split on two datasets

        Parameters
        ----------
        left_indices : np.NDArray of int
            indices of the left dataset

        right_indices : np.NDArray of int
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
        
        cdef :
            FuncWrapper criteria = self.criteria
            double left_imp = 0.0
            double right_imp = 0.0
            double crit = 0.0
            int i
        cdef int n_outcomes = left_indices.shape[0]

        # calculate on the left dataset
        if n_outcomes == 0:
            left_imp = 0.0
        else:
            left_imp = criteria.func(self.features, self.outcomes, left_indices, self.class_labels, self.n_in_class)
            crit += left_imp * (n_outcomes / self.n_indices)
        
        # calculate on the right dataset
        n_outcomes = right_indices.shape[0]
        if n_outcomes == 0:
            right_imp = 0.0
        else:
            right_imp = criteria.func(self.features, self.outcomes, right_indices, self.class_labels, self.n_in_class)
            crit += right_imp * (n_outcomes / self.n_indices)
        
        cdef double mean_thresh = (self.features[left_indices[-1], feature] + self.features[right_indices[0], feature]) / 2
        
        return (crit, left_imp, right_imp, mean_thresh)
    
    '''
    cdef int[::1] sort_feature(self, int[:] indices, double[:] feature):
        """
        Parameters
        ----------
        indices : np.NDArray
            A memoryview of the indices which are to be sorted over
        
        feature: np.NDArray
            A memoryview of feature values that are to be sorted
            
        Returns 
        -----------
        np.NDArray
            A memoryview of the sorted indices 
        """

        cdef:
            int x
            double[:] temp

        temp = np.asarray(feature)[indices]
        return np.array(indices.base[np.argsort(temp)], dtype=np.int32) 
    
    cdef int[:] sort_feature(self, int[:] indices, double[:] feature):
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
        return np.array(sorted(indices.base, key=lambda x: feature[x]), dtype=int)   
    '''
    
    cdef int[::1] sort_feature(self, int[:] indices, double[:] feature):
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
    

    def set_pre_sort(self, pre_sort: np.ndarray):
        self.pre_sort = pre_sort
    
    cpdef get_split(self, int[::1] indices):
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
        self.n_indices = len(indices)
        cdef:
            int best_feature = -1
            double best_threshold = INFINITY
            double best_score = INFINITY
        
        best_imp = []
        split = []  
        cdef double[:] current_feature_values
        # declare variables for loop
        cdef int i
        cdef int N_i = self.n_indices - 1
        cdef int[:] sorted_index_list_feature
        # for all features
        for feature in range(self.n_features):
            current_feature_values = self.features[:, feature]
            if self.pre_sort is None:
                sorted_index_list_feature = self.sort_feature(self.indices, current_feature_values)
            else:
                # Create mask to only retrieve the indices in the current node from presort
                mask = np.isin(self.pre_sort[:, feature], self.indices)
                # Use the mask to retrieve values from presort
                sorted_index_list_feature = np.asarray(self.pre_sort[:, feature])[mask]
            
            # loop over sorted feature list
            for i in range(N_i):
                # Skip one iteration of the loop if the current threshold value is the same as the next in the feature list
                if current_feature_values[sorted_index_list_feature[i]] == current_feature_values[sorted_index_list_feature[i + 1]]:
                    continue 
                # Split the dataset
                left_indices = sorted_index_list_feature[:i + 1]
                right_indices = sorted_index_list_feature[i + 1:]
                crit, left_imp, right_imp, threshold = self.test_split(left_indices, right_indices, feature) # test the split

                if crit < best_score:
                    # save the best split
                    best_feature, best_threshold, best_score, best_imp = feature, threshold, crit, [left_imp, right_imp] # The index is given as the index of the first element of the right dataset 
                    split = [left_indices, right_indices]
        return split, best_threshold, best_feature, best_score, best_imp # return the best split