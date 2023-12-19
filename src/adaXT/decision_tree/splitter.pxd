cimport numpy as cnp
from .criteria cimport Criteria
cnp.import_array()

cdef class Splitter:
    cdef:
        double[:, ::1] features
        double[::1] response
        int n_features
        int[:] indices
        int n_indices
        Criteria criteria
        int n_class
        double* class_labels
        int* n_in_class

    cdef int[:] sort_feature(self, int[:], double[:])

    cpdef get_split(self, int[:])
