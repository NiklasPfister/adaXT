cdef class Criteria:
    cdef:
        double[:, ::1] x
        double[::1] y
        double[::1] sample_weight
        int old_obs
        int old_split
        int old_feature

    cdef double proxy_improvement(self, int[::1] indices, int split_idx)
    """
        This function defaults to calling impurity, but if there is a way
        to calculate a proxy for the impurity, then defining this function,
        will give you the speed up without having to change anything else.
    """
    cdef double update_proxy(self, int[::1] indices, int split_idx)
    """
        This function is a drop in replacement for proxy_improvement. It get's called,
        whenever we are looking for a split within the same node and on the same feature,
        but we have just moved the splitting index a little bit. This means, that indices
        self.indices[self.old_split:split_idx] have moved from the previous right child node
        to the left child node. Can be used to shorten the calculation, but defaults to just
        calling proxy_improvement.
    """

    cpdef double impurity(self, int[::1] indices)
    cdef (double, double) evaluate_split(self, int[::1] indices, int split_idx, int feature)
    """
        Function to evaluate how good a split is
        ----------

        Parameters
        ----------
        indices: int[:]
        the indices of a given node

        split_idx: int
        the index of the split, such that left indices are indices[:split_idx] and right indices are indices[split_idx:]

        feature: int
        The current feature we are working on

        Returns
        -----------
        (double, double)
        The critical value of the given split,
        followed by the mean threshold between the
        split index and the closest neighbour outside.
    """

cdef class Gini_index(Criteria):
    cdef:
        double[::1] class_labels
        double* weight_in_class_left
        double* weight_in_class_right
        double weight_left
        double weight_right
        int num_classes
        bint first_call

    cdef void reset_weight_list(self, double* class_occurences)

    cpdef double impurity(self, int[::1] indices)

    cdef double _gini(self, int[::1] indices, double* class_occurences)
    """
    Function that calculates the gini index of a dataset
    ----------

    Parameters
    ----------
    indices : memoryview of NDArray
        The indices to calculate the gini index for

    class_occurences : double pointer
        A pointer to an double array for the number of elements seen of each class

    Returns
    -----------
    double
        The value of the gini index
    """

    cdef double update_proxy(self, int[::1] indices, int new_split)

    cdef double proxy_improvement(self, int[::1] indices, int split_idx)


cdef class Entropy(Criteria):
    cdef:
        double[::1] class_labels
        double* weight_in_class_left
        double* weight_in_class_right
        double weight_left
        double weight_right
        int num_classes
        bint first_call

    cpdef double impurity(self, int[::1] indices)

    cdef void reset_weight_list(self, double* class_occurences)

    cdef double _entropy(self, int[:] indices, double* class_occurences)
    """
    Function that calculates the entropy index of a dataset
    ----------

    Parameters
    ----------
    indices : memoryview of NDArray
        The indices to calculate

    class_occurences : double pointer
        A pointer to an double array for the number of elements seen of each class

    Returns
    -----------
    double
        The value of the entropy index
    """

    cdef double proxy_improvement(self, int[::1] indices, int split_idx)

    cdef double update_proxy(self, int[::1] indices, int new_split)

cdef class Squared_error(Criteria):
    cdef:
        double left_sum
        double right_sum
        double weight_left, weight_right

    cdef double update_proxy(self, int[::1] indices, int new_split)

    cdef double proxy_improvement(self, int[::1] indices, int split_idx)

    cpdef double impurity(self, int[::1] indices)

    cdef double _squared_error(self, int[::1] indices)
    """
    Function used to calculate the squared error of y[indices]
    ----------

    Parameters
    ----------
    indices : memoryview of NDArray
        The indices to calculate

    left_or_right : int
        An int indicating whether we are calculating on the left or right dataset

    Returns
    -------
    double
        The variance of the response y
    """
