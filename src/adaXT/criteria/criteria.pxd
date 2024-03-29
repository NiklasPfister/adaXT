cdef class Criteria:
    cdef:
        double[:, ::1] x
        double[::1] y
        double[::1] sample_weight
    cpdef double impurity(self, int[:] indices)
    cdef (double, double, double, double) evaluate_split(self, int[:] indices, int split_idx, int feature)
