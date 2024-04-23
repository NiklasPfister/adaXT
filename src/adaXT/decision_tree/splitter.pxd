cimport numpy as cnp
from ..criteria.criteria cimport Criteria  # Must be complete path for cimport
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

    cpdef get_split(self, int[::1], int[::1])
