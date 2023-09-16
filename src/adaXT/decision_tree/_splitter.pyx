import numpy as np
from numpy import float64 as DOUBLE
cimport numpy as cnp
from numpy.math cimport INFINITY
from ._func_wrapper cimport FuncWrapper
cnp.import_array()

# cython: profile=true

cdef class Splitter:
    """
    Splitter class used to create splits of the data
    """

    def __init__(self, double[:, ::1] X, double[:] Y, criteria: FuncWrapper):
        
        self.features = X 
        self.outcomes = Y 

        self.n_features = X.shape[0]
        self.criteria = criteria
        self.pre_sort = None
        # self.constant_features = np.empty(len(self.features)) #TODO: not yet implemented

    def set_pre_sort(self, pre_sort: np.ndarray):
        self.pre_sort = pre_sort

    cdef cnp.ndarray sort_feature(self, int[:] indices, double[:] feature):
        cdef double[:] sort_list
        cdef long[:] argsorted
        sort_list = feature.base[indices]
        argsorted = np.argsort(sort_list)

        return indices.base[argsorted]

    cdef (double, double, double, double) test_split(self, int[:] left_indices, int[:] right_indices, int feature):
        cdef:
            double mean_thresh
            FuncWrapper criteria
            double left_imp = 0.0
            double right_imp = 0.0
            double crit = 0.0

        criteria = self.criteria
        features = self.features.base
        outcomes = self.outcomes.base

        cdef int n_outcomes = left_indices.shape[0]

        # calculate on the left dataset
        if n_outcomes == 0:
            left_imp = 0.0
        else:
            left_imp = criteria.func(features[left_indices], outcomes[left_indices])
            crit += left_imp * (n_outcomes / self.n_indices)
        
        # calculate in the right dataset
        n_outcomes = right_indices.shape[0]
        if n_outcomes == 0:
            right_imp = 0.0
        else:
            right_imp = criteria.func(features[right_indices], outcomes[right_indices])
            crit += right_imp * (n_outcomes / self.n_indices)
        
        mean_thresh = (self.features[left_indices[-1], feature] + self.features[right_indices[0], feature]) / 2
        
        return (crit, left_imp, right_imp, mean_thresh)

    cpdef get_split(self, int[:] indices):
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

        features = self.features.base
        n_features = self.n_features
        # for all features
        for feature in range(n_features):
            current_feature_values = features[:, feature]
            if self.pre_sort is None:
                sorted_index_list_feature = self.sort_feature(indices, current_feature_values)
            else:
                pre_sort = self.pre_sort.base
                # Create mask to only retrieve the indices in the current node from presort
                mask = np.isin(pre_sort[:, feature], indices)
                # Use the mask to retrieve values from presort
                sorted_index_list_feature = np.asarray(pre_sort[:, feature])[mask]       

            
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