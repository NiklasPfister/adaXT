from .nodes cimport LeafNode, DecisionNode
cimport numpy as cnp

cdef class LeafBuilder:
    cdef:
        double[::1] y
        double[:, ::1] x

    cpdef LeafNode build_leaf(self,
                              int leaf_id,
                              int[::1] indices,
                              int depth,
                              double impurity,
                              int n_samples,
                              object parent)


cdef class LeafBuilderClassification(LeafBuilder):
    cdef:
        double[::1] classes
        int n_classes

    cdef double[::1] __get_mean(self, int[::1] indices, int n_samples)

    cpdef LeafNode build_leaf(self,
                              int leaf_id,
                              int[::1] indices,
                              int depth,
                              double impurity,
                              int n_samples,
                              object parent)


cdef class LeafBuilderRegression(LeafBuilder):

    cdef double __get_mean(self, int[::1] indices)

    cpdef LeafNode build_leaf(self,
                              int leaf_id,
                              int[::1] indices,
                              int depth,
                              double impurity,
                              int n_samples,
                              object parent)
