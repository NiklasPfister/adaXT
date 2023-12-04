# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()
from .criteria cimport Criteria


cdef double EPSILON = 2*np.finfo('double').eps
# The rounding error for a criteria function is larger than that in DepthTreeBuilder.
# This is most likely doe to the fact that the criteria does multiple calculations before returing the critical value,
# where the DepthTreeBuilder is just comparing the impurity (that already has gone through this check).
cdef double INFINITY = 1e20

cdef class Splitter:
    """
    Splitter class used to create splits of the data
    """
    def __init__(self, double[:, ::1] X, double[:] Y, criteria: Criteria):
        '''
        Class initializer
        ----------------
        Parameters
        ----------
            x: memoryview of NDArray
                The feature values of the dataset
            y: memoryview of NDArray
                The response values of the dataset
            criteria: FuncWrapper object
                A FuncWrapper object containing the criteria function
        '''
        self.features = X
        self.response = Y
        self.n_features = X.shape[1]
        self.criteria = criteria
        self.pre_sort = None
        self.n_class = len(np.unique(Y))

    def set_pre_sort(self, pre_sort):
        '''
        Function to set a presort array for the feature values
        ----------------

        Parameters
        ----------
        pre_sort: np.ndarray
            The pre sort array to set
        '''
        self.pre_sort = pre_sort

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
            double[:] temp
        temp = feature.base[indices]
        return np.array(indices.base[np.argsort(temp)], dtype=np.int32)

    # Gets split given the indices in the given node
    cpdef get_split(self, int[:] indices):
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
            cnp.ndarray[cnp.int32_t, ndim=1] sorted_index_list_feature
            double crit

        features = self.features.base
        n_features = self.n_features
        split, best_imp = [], []
        # For all features
        for feature in range(n_features):
            current_feature_values = features[:, feature]
            if self.pre_sort is None:
                sorted_index_list_feature = self.sort_feature(
                    indices, current_feature_values
                    )
            else:
                pre_sort = self.pre_sort.base

                # Mask only current node
                mask = np.isin(pre_sort[:, feature], indices)
                # Use the mask to retrieve values from presort
                sorted_index_list_feature = np.asarray(pre_sort[:, feature])[mask]

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
                    best_score, best_imp = crit, [left_imp, right_imp]
                    split = [sorted_index_list_feature[:i+1], sorted_index_list_feature[i+1:]]

        return split, best_threshold, best_feature, best_score, best_imp
