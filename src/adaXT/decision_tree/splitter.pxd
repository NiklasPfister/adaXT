cimport numpy as cnp
from ..criteria.criteria cimport Criteria  # Must be complete path for cimport
cnp.import_array()

cdef class Splitter:
    cdef:
        double[:, ::1] X
        double[:, ::1] Y
        int n_features
        int[:] indices
        int n_indices
        Criteria criteria_instance

    cpdef get_split(self, int[::1], int[::1])
