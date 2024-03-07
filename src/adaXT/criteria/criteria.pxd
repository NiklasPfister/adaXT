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
