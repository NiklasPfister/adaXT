cdef class Criterion:
    cdef double impurity(self, double[:, ::1] x, double[:] y, int[:] indices)