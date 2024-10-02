cimport numpy as cnp

ctypedef cnp.float64_t DOUBLE_t

cdef class LeafBuilder:
    cdef:
        double[:, ::1] Y
        double[:, ::1] X

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

    cdef double[::1] __get_mean(self, int[::1] indices)

    cpdef object build_leaf(self,
                            int leaf_id,
                            int[::1] indices,
                            int depth,
                            double impurity,
                            double weighted_samples,
                            object parent)


cdef class LeafBuilderRegression(LeafBuilder):

    cdef cnp.ndarray[DOUBLE_t, ndim=1] __get_mean(self, int[::1] indices)

    cpdef object build_leaf(self,
                            int leaf_id,
                            int[::1] indices,
                            int depth,
                            double impurity,
                            double weighted_samples,
                            object parent)


cdef class LeafBuilderPartialLinear(LeafBuilderRegression):

    cdef (double, double) __custom_mean(self, int[::1] indices)

    cdef (double, double, double) __theta(self, int[::1] indices)

    cpdef object build_leaf(self,
                            int leaf_id,
                            int[::1] indices,
                            int depth,
                            double impurity,
                            double weighted_samples,
                            object parent)


cdef class LeafBuilderPartialQuadratic(LeafBuilderRegression):

    cdef (double, double, double) __custom_mean(self, int[::1] indices)

    cdef (double, double, double, double) __theta(self, int[::1] indices)

    cpdef object build_leaf(self,
                            int leaf_id,
                            int[::1] indices,
                            int depth,
                            double impurity,
                            double weighted_samples,
                            object parent)
