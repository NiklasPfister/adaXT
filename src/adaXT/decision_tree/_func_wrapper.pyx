import numpy as np
from libc.stdlib cimport malloc, free
cdef class FuncWrapper:
    def __cinit__(self):
       self.func = NULL
    
    # Function to calculate the criteria value from a python file
    def crit_func(self, x, y, indices):
        n_tot_class = len(np.unique(y))
        cdef:
            double* class_labels = <double *> malloc(sizeof(double)*n_tot_class)
            int* n_in_class = <int *> malloc(sizeof(int)* n_tot_class)
        res = self.func(x, y, indices, class_labels, n_in_class)
        free(class_labels)
        free(n_in_class)
        return res

    # Set the function value of a FuncWrapper object to a given function
    @staticmethod
    cdef FuncWrapper make_from_ptr(func_ptr f):
        # TODO do error checking on input function
        cdef FuncWrapper out = FuncWrapper()
        out.func = f
        return out
