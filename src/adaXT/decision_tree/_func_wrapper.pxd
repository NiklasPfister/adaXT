cimport numpy as cnp

ctypedef double (*func_ptr)(cnp.ndarray, cnp.ndarray)

cdef class FuncWrapper:
    cdef func_ptr func
    @staticmethod
    cdef FuncWrapper make_from_ptr(func_ptr f)