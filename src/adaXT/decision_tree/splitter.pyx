# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as cnp
cnp.import_array()
from ..criteria.criteria cimport Criteria  # Must be complete path for cimport
from libc.stdlib cimport qsort

cdef double EPSILON = 2*np.finfo('double').eps
# The rounding error for a criteria function is set twice as large as in DepthTreeBuilder.
# This is needed due to the fact that the criteria does multiple calculations before returing the critical value,
# where the DepthTreeBuilder is just comparing the impurity (that already has gone through this check).

cdef double INFINITY = np.inf

cdef double[:] current_feature_values

cdef inline int compare(const void* a, const void* b) noexcept nogil:
    cdef:
        int a1 = (<int *> a)[0]
        int b1 = (<int *> b)[0]

    if  current_feature_values[a1] >= current_feature_values[b1]:
        return 1
    else:
        return -1

cdef inline int[::1] sort_feature(int[::1] indices):
    """
    Function to sort an array at given indices.

    Parameters
    ----------
    indices : memoryview of NDArray
        A list of the indices which are to be sorted over

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
    def __init__(self, double[:, ::1] X, double[:, ::1] Y, criteria_instance: Criteria):
        self.X = X
        self.Y = Y
        self.n_features = X.shape[1]
        self.criteria_instance = criteria_instance

    cpdef get_split(self, int[::1] indices, int[::1] feature_indices):
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
            current_feature_values = np.asarray(self.X[:, feature])
            sorted_index_list_feature = sort_feature(indices)

            # Loop over sorted feature list
            for i in range(N_i):
                # Skip one iteration of the loop if the current
                # threshold value is the same as the next in the feature list
                if (self.X[sorted_index_list_feature[i], feature] ==
                        self.X[sorted_index_list_feature[i + 1], feature]):
                    continue
                # test the split
                crit, threshold = self.criteria_instance.evaluate_split(
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
            best_imp = [self.criteria_instance.impurity(split[0]), self.criteria_instance.impurity(split[1])]

        return split, best_threshold, best_feature, best_score, best_imp
