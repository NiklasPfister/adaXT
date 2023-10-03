#cython: profile=True

cimport cython
import numpy as np
from numpy import float64 as DOUBLE
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

    def __init__(self, double[:, ::1] X, double[:] Y, criteria: FuncWrapper):
        
        self.features = X 
        self.outcomes = Y 

        self.n_features = X.shape[1]
        self.criteria = criteria
        self.pre_sort = None
        self.n_class = len(np.unique(Y))
        # self.constant_features = np.empty(len(self.features)) #TODO: not yet implemented

    cpdef void make_c_lists(self, int n_class):
        # C lists which are used for more efficient criteria function in case of Classification tree.
        self.class_labels = <double *> malloc(sizeof(double) * self.n_class)
        self.n_in_class = <int *> malloc(sizeof(int) * self.n_class)

    cpdef void free_c_lists(self):
        # if make_c_lists is called once by caller, this should be called once to.
        free(self.class_labels)
        free(self.n_in_class)

    def set_pre_sort(self, pre_sort: np.ndarray):
        #Set the pre_sort value.
        self.pre_sort = pre_sort

    cdef cnp.ndarray sort_feature(self, int[:] indices, double[:] feature):
        # Sort the indicies given a list of feature values
        cdef:
            double[:] temp
        temp = feature.base[indices] 
        return np.array(indices.base[np.argsort(temp)], dtype=np.int32) 

    cdef (double, double, double, double) test_split(self, int[:] left_indices, int[:] right_indices, int feature):
        cdef:
            double mean_thresh
            FuncWrapper criteria
            double left_imp = 0.0
            double right_imp = 0.0
            double crit = 0.0
            double* class_labels
            int* n_in_class
            int n_outcomes = left_indices.shape[0] # initial

        criteria = self.criteria
        features = self.features
        outcomes = self.outcomes
        class_labels = self.class_labels
        n_in_class = self.n_in_class
        
        # calculate on the left dataset
        if n_outcomes == 0:
            left_imp = 0.0
        else:
            left_imp = criteria.func(features, outcomes, left_indices, class_labels, n_in_class)
            crit = left_imp * (n_outcomes/self.n_indices)

        # calculate in the right dataset
        n_outcomes = right_indices.shape[0]
        if n_outcomes == 0:
            right_imp = 0.0
        else:
            right_imp = criteria.func(features, outcomes, right_indices, class_labels, n_in_class)
        
        crit += (right_imp) * (n_outcomes/self.n_indices)
        mean_thresh = (features[left_indices[-1], feature] + features[right_indices[0], feature]) / 2
        
        return (crit, left_imp, right_imp, mean_thresh)

    # Gets split given the indices in the given node
    cpdef get_split(self, int[:] indices):
        self.indices = indices
        self.n_indices = len(indices)
        cdef:
            int N_i = self.n_indices - 1 # number of indices to loop over. Skips last
            double best_threshold = INFINITY
            double best_score = INFINITY
            int best_feature = 0
            double[:] current_feature_values
            int i, feature # variables for loop
            int[:] left_indices, right_indices
            cnp.ndarray[cnp.int32_t, ndim=1] sorted_index_list_feature
        
        # If the classes list is not null, then we have a classification tree, as such allocate memory for lists
        if self.class_labels != NULL:
            self.free_c_lists()
            classes = np.unique(self.outcomes.base[indices])
            self.make_c_lists(len(classes))

        best_imp = []
        split = []  
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