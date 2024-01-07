# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()
from .criteria cimport Criteria

cdef double EPSILON = 2*np.finfo('double').eps
# The rounding error for a criteria function is larger than that in DepthTreeBuilder.
# This is most likely needed due to the fact that the criteria does multiple calculations before returing the critical value,
# where the DepthTreeBuilder is just comparing the impurity (that already has gone through this check).

cdef double INFINITY = np.inf


cdef class Splitter:
    """
    Splitter class used to create splits of the data
    """
    def __init__(self, double[:, ::1] X, double[::1] Y, criteria: Criteria):
        '''
        Class initializer
        ----------------
        Parameters
        ----------
            x: memoryview of NDArray
                The feature values of the dataset
            y: memoryview of NDArray
                The response values of the dataset
            criteria: Criteria
                The criteria class used to find the impurity of a split
        '''
        self.features = X
        self.response = Y
        self.n_features = X.shape[1]
        self.criteria = criteria
        self.n_class = len(np.unique(Y))

    cdef cnp.ndarray sort_feature(self, int[:] indices, double[:] feature):
        """
        Function to sort an array at given indices.

        Parameters
        ----------
        indices : memoryview of NDArray
            A list of the indices which are to be sorted over

        feature: memoryview of NDArray
            A list containing the feature values that are to be sorted over

        Returns
        -----------
        memoryview of NDArray
            A list of the sorted indices
        """

        cdef:
            cnp.ndarray[double, ndim=1] feat_temp = np.asarray(feature)
            cnp.ndarray[int, ndim=1] idx = np.asarray(indices)
            cnp.ndarray[long, ndim=1] temp
        temp = np.argsort(feat_temp[idx])
        return idx[temp]

    cpdef get_split(self, int[:] indices, int[:] feature_indices):
        """
        Function that finds the best split of the dataset
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            Indices constituting the dataset

        Returns
        -----------
        (list, double, int, double, double)
            Returns the best split of the dataset, with the values being:
            (1) a list containing the left and right indices, (2) the best
            threshold for doing the splits, (3) what feature to split on,
            (4) the best criteria score, and (5) the best impurity
        """
        self.indices = indices
        self.n_indices = indices.shape[0]
        cdef:
            # number of indices to loop over. Skips last
            int N_i = self.n_indices - 1
            double best_threshold = INFINITY
            double best_score = INFINITY
            int best_feature = 0
            double[:] current_feature_values
            int i, feature  # variables for loop
            cnp.ndarray[int, ndim=1] sorted_index_list_feature
            int[:] best_sorted
            int best_split_idx
            double best_left_imp, best_right_imp
            double crit

        features = self.features.base
        n_features = self.n_features
        split, best_imp = [], []
        best_right_imp, best_left_imp = 0.0, 0.0
        best_split_idx = -1
        best_sorted = None
        # For all features
        for feature in feature_indices:
            current_feature_values = features[:, feature]
            sorted_index_list_feature = self.sort_feature(
                    indices, current_feature_values
                    )

            # Loop over sorted feature list
            for i in range(N_i):
                # Skip one iteration of the loop if the current
                # threshold value is the same as the next in the feature list
                if (current_feature_values[sorted_index_list_feature[i]] ==
                        current_feature_values[sorted_index_list_feature[i + 1]]):
                    continue
                # test the split
                crit, left_imp, right_imp, threshold = self.criteria.evaluate_split(
                                                        sorted_index_list_feature, i+1,
                                                        feature
                                                        )

                if best_score - crit > EPSILON:  # rounding error
                    # Save the best split
                    # The index is given as the index of the
                    # first element of the right dataset
                    best_feature, best_threshold = feature, threshold
                    best_score = crit
                    best_left_imp = left_imp
                    best_right_imp = right_imp
                    best_split_idx = i + 1
                    best_sorted = sorted_index_list_feature

        # We found a best split
        if best_sorted is not None:
            split = [best_sorted[0:best_split_idx], best_sorted[best_split_idx:self.n_indices]]
            best_imp = [best_left_imp, best_right_imp]
        return split, best_threshold, best_feature, best_score, best_imp
