import numpy as np
cimport numpy as cnp
from ._func_wrapper import FuncWrapper
cimport cython
cnp.import_array()

# cython: profile=true

cdef class Splitter:
    """
    Splitter class used to create splits of the data
    """

    def __init__(self, double[:, ::1] X, double[:] Y, criterion: FuncWrapper, presort: np.ndarray|None = None):
        self.features = X 
        self.outcomes = Y 

        self.n_features = X.shape[0]
        self.criteria = criterion
        self.pre_sort = presort
        # self.constant_features = np.empty(len(self.features)) #TODO: not yet implemented

    cdef cnp.ndarray sort_feature(self, cnp.ndarray[npInt, ndim=1] indices, cnp.ndarray[npFloat, ndim=1] feature):
        cdef npFloat[:] sort_list
        sort_list = feature[indices]
        sort_list = np.argsort(sort_list)

        return indices[sort_list]

    cdef (double, double, double, double) test_split(self, int[:] left_indices, int[:] right_indices, int feature):
        cdef:
            double[2] imp
            int[:] indices 
            double crit, mean_thresh
            int n_total, n_curr, i
            double[:] x, y, curr

        features = self.features.base
        outcomes = self.outcomes.base
        func_wrap = self.criteria
        criteria = func_wrap.func
        
        indices = self.indices
        idx_split = [left_indices, right_indices]

        n_total = len(indices)
        for i in range(2): # Left and right side
            curr = idx_split[i]
            n_curr = len(curr)
            if n_curr == 0:
                continue
            x = features[curr]
            y = outcomes[curr]
            imp[i] = criteria(x, y) # Calculate the impurity of current child
            crit += imp[i] * (n_curr / n_total) # Weight by the amount of datapoints in child

        mean_thresh = np.mean([features[left_indices[-1], feature], features[right_indices[0], feature]])
        
        return (crit, imp[0], imp[1], mean_thresh)

    cpdef get_split(self, int[:] indices):
        self.indices = indices
        self.n_indices = len(indices)
        cdef:
            int best_feature = -1
            double best_threshold = 1000000.0
            double best_score = 1000000.0
        
        best_imp = []
        split = []  
        cdef double[:] current_feature_values
        # declare variables for loop
        cdef int i
        cdef int N_i = self.n_indices - 1

        features = self.features.base
        outcomes = self.features.base
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