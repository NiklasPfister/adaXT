cimport numpy as cnp
cnp.import_array()
cdef class Splitter:
    cdef:
        double[:, :] features
        double[:, :] outcomes
        int n_features
        int[:, :] pre_sort

    cdef cnp.ndarray sort_feature(self, cnp.ndarray[cnp.int_t, ndim=1] indices, cnp.ndarray[cnp.float64_t, ndim=1] feature)