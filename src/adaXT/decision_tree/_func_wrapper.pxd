cimport numpy as cnp
"""
The funcwrapper class is created such that cdef functions in cython,
can be passed through python code. When a user uses the funcwrapper,
it saves a pointer to the cdef function. That way cython can still
call this function, even though it has been loaded within pure python.
"""

# Function type, this is the type of any criteria function
ctypedef double (*func_ptr)(double[:, ::1], double[:], int[:],  double*, int*)

cdef class FuncWrapper:
    cdef func_ptr func

    @staticmethod
    cdef FuncWrapper make_from_ptr(func_ptr f)