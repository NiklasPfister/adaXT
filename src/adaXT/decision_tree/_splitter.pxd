cimport numpy as cnp
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
        double[:] outcomes
        int n_features
        int[:] pre_sort
        int[:] indices
        object criteria

    cdef cnp.ndarray sort_feature(self, cnp.ndarray[npInt], cnp.ndarray[npFloat])

    cdef (double, double, double, double) test_split(self, int[:], int[:], int)

    cpdef get_split(self, int[:])