cimport numpy as cnp
from ._func_wrapper cimport FuncWrapper
cnp.import_array()

ctypedef cnp.float64_t npFloat
ctypedef cnp.int_t npInt

cdef class test_obj:
    cdef:
        double crit
        list[:, :] idx_split
        double[2] imp
        double split_val

cdef class Splitter:
    cdef:
        double[:, ::1] features
        double[::1] outcomes
        int n_features
        int[:, ::1] pre_sort
        int[::1] indices
        int n_indices
        FuncWrapper criteria
        int n_class
        double* class_labels
        int* n_in_class

    cdef int[::1] sort_feature(self, int[:], double[:])

    cdef (double, double, double, double) test_split(self, int[::1], int[::1], int)

    cpdef get_split(self, int[::1])
    
    cpdef void make_c_lists(self, int)