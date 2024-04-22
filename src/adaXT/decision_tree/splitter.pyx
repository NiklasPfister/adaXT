# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, profile=True
# from libcstdlib cimport qsort_r
import numpy as np
cimport numpy as cnp
cnp.import_array()
from ..criteria.criteria cimport Criteria  # Must be complete path for cimport
from libc.stdlib cimport qsort
from cython cimport view

cdef double EPSILON = 2*np.finfo('double').eps
# The rounding error for a criteria function is set twice as large as in DepthTreeBuilder.
# This is needed due to the fact that the criteria does multiple calculations before returing the critical value,
# where the DepthTreeBuilder is just comparing the impurity (that already has gone through this check).

cdef double INFINITY = np.inf

cdef double[:] current_feature_values

cdef int compare(const void* a, const void* b) noexcept nogil:
    cdef:
        int a1 = (<int *> a)[0]
        int b1 = ( <int *> b )[0]

    if  current_feature_values[a1] >= current_feature_values[b1]:
        return 1
    if current_feature_values[a1] < current_feature_values[b1]:
        return -1

cdef int[::1] sort_feature(int[::1] indices):
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
        int n_obs = indices.shape[0]
        int[::1] ret = indices.copy()
    qsort(&ret[0], n_obs, sizeof(int), compare)
    return ret


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


    cpdef get_split(self, int[::1] indices, int[::1] feature_indices):
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
        global current_feature_values
        self.indices = indices
        self.n_indices = indices.shape[0]
        cdef:
            # number of indices to loop over. Skips last
            int N_i = self.n_indices - 1
            double best_threshold = INFINITY
            double best_score = INFINITY
            int best_feature = 0
            int i, feature  # variables for loop
            int[::1] sorted_index_list_feature
            int[::1] best_sorted
            int best_split_idx
            double crit

        split, best_imp = [], []
        best_split_idx = -1
        best_sorted = None
        # For all features
        for feature in feature_indices:
            current_feature_values = np.asarray(self.features[:, feature])
            sorted_index_list_feature = sort_feature(indices)

            # Loop over sorted feature list
            for i in range(N_i):
                # Skip one iteration of the loop if the current
                # threshold value is the same as the next in the feature list
                if (self.features[sorted_index_list_feature[i], feature] ==
                        self.features[sorted_index_list_feature[i + 1], feature]):
                    continue
                # test the split
                crit, threshold = self.criteria.evaluate_split(
                                                        sorted_index_list_feature, i+1,
                                                        feature
                                                        )
                if best_score > crit:  # rounding error
                    # Save the best split
                    # The index is given as the index of the
                    # first element of the right dataset
                    best_feature, best_threshold = feature, threshold
                    best_score = crit
                    best_split_idx = i + 1
                    best_sorted = sorted_index_list_feature

        # We found a best split
        if best_sorted is not None:
            split = [best_sorted[0:best_split_idx], best_sorted[best_split_idx:self.n_indices]]
            best_imp = [self.criteria.impurity(split[0]), self.criteria.impurity(split[1])]

        return split, best_threshold, best_feature, best_score, best_imp
