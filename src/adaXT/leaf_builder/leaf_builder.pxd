cimport numpy as cnp

cdef class LeafBuilder:
    cdef:
        double[::1] y
        double[:, ::1] x

    cpdef object build_leaf(self,
                            int leaf_id,
                            int[::1] indices,
                            int depth,
                            double impurity,
                            double weighted_samples,
                            object parent)


cdef class LeafBuilderClassification(LeafBuilder):
    cdef:
        double[::1] classes
        int n_classes

    cdef double[::1] _get_mean(self, int[::1] indices)

    cpdef object build_leaf(self,
                            int leaf_id,
                            int[::1] indices,
                            int depth,
                            double impurity,
                            double weighted_samples,
                            object parent)


cdef class LeafBuilderRegression(LeafBuilder):

    cdef double _get_mean(self, int[::1] indices)

    cpdef object build_leaf(self,
                            int leaf_id,
                            int[::1] indices,
                            int depth,
                            double impurity,
                            double weighted_samples,
                            object parent)


cdef class LeafBuilderPartialLinear(LeafBuilderRegression):

    cdef (double, double) _custom_mean(self, int[::1] indices)

    cdef (double, double, double) _theta(self, int[::1] indices)

    cpdef object build_leaf(self,
                            int leaf_id,
                            int[::1] indices,
                            int depth,
                            double impurity,
                            double weighted_samples,
                            object parent)


cdef class LeafBuilderPartialQuadratic(LeafBuilderRegression):

    cdef (double, double, double) _custom_mean(self, int[::1] indices)

    cdef (double, double, double, double) _theta(self, int[::1] indices)

    cpdef object build_leaf(self,
                            int leaf_id,
                            int[::1] indices,
                            int depth,
                            double impurity,
                            double weighted_samples,
                            object parent)
