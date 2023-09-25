cimport numpy as cnp

ctypedef double (*func_ptr)(double[:, ::1], double[::1], int[::1])

cdef class FuncWrapper:
    cdef func_ptr func
    @staticmethod
    cdef FuncWrapper make_from_ptr(func_ptr f)