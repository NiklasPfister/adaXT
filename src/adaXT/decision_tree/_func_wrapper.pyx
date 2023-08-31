cimport numpy as cn
cdef class FuncWrapper:
    def __cinit__(self):
       self.func = NULL

    @staticmethod
    cdef FuncWrapper make_from_ptr(func_ptr f):
        cdef FuncWrapper out = FuncWrapper()
        out.func = f
        return out
