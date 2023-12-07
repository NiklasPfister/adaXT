cimport numpy as cnp
from .criteria cimport Criteria
cnp.import_array()

cdef class test_obj:
    cdef:
        double crit
        list[:, :] idx_split
        double[2] imp
        double split_val

cdef class Splitter:
    cdef:
        double[:, ::1] features
        double[:] response
        int n_features
        int[:] indices
        int n_indices
        Criteria criteria
        int n_class
        double* class_labels
        int* n_in_class

    cdef cnp.ndarray sort_feature(self, int[:], double[:])

    cpdef get_split(self, int[:])
