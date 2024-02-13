import numpy as np
from .Nodes import DecisionNode, LeafNode

cdef class Predict():
    cdef:
        double[:, ::1] X
        double[::1] Y
        int n_features
        str tree_type
        double[::1] classes
        object root

    cdef double[:, ::1] __check_dimensions(self, double[:, ::1] X)

    cdef int __find_max_index(self, list lst)

    cpdef double[:] predict(self, double[:, ::1] X)

    cpdef double[:, ::1] predict_leaf_matrix(self, double[:, ::1] X, bint scale=*)

    cpdef list predict_proba(self, double[:, ::1] X)
