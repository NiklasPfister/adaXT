

import numpy as np
from cython cimport boundscheck, wraparound
from libc.math cimport fabs as cabs
cimport numpy as cnp
cnp.import_array()
from libc.stdlib cimport malloc, free

from ._func_wrapper cimport FuncWrapper

# int for y
cdef double gini_index(double[:, ::1] x, double[:] y, int[:] indices, int n_tot_class) nogil:
    cdef:
        double sum = 0.0
        int n_obs = indices.shape[0]
        int n_classes = 0
        double* class_labels = <double *> malloc(sizeof(double) * n_tot_class)
        int* n_in_class = <int *> malloc(sizeof(int) * n_tot_class)
        double proportion_cls 
        double epsilon = 2e-30
        int seen
    for i in range(n_obs):
        seen = 0
        for j in range(n_classes):
            if cabs(class_labels[j] - y[indices[i]]) < epsilon:
                n_in_class[j] = n_in_class[j] + 1
                seen = 1
                break
        if (seen == 0):
            class_labels[n_classes] = y[indices[i]]
            n_in_class[n_classes] = 1
            n_classes += 1
    for i in range(n_classes):
        proportion_cls = n_in_class[i] / n_obs
        sum += proportion_cls**2
    free(class_labels)
    free(n_in_class)
    return 1 - sum  

cdef double variance(double[:, ::1] x, double[:] y, int[:] indices, int n_tot_class) nogil:
    """
    Calculates the variance 

    Parameters
    ----------
    x : npt.NDArray
        features, not used in this implementation
    y : npt.NDArray
        1-dimensional outcomes

    Returns
    -------
    double
        variance of the y data
    """
    cdef double cur_sum = 0
    cdef double mu = mean(y, indices)
    cdef int i 
    cdef int n_indices = indices.shape[0]

    with boundscheck(False), wraparound(False):    
        for i in range(n_indices):
            cur_sum += (y[indices[i]] - mu) * (y[indices[i]]- mu)

    return cur_sum / n_indices

cdef double mean(double[:] y, int[:] indices) nogil:
    cdef double sum = 0.0
    cdef int i 
    cdef int length = indices.shape[0]

    with boundscheck(False), wraparound(False): 
        for i in range(length):
            sum += y[indices[i]]
    return sum / length

def gini_index_wrapped():
    return FuncWrapper.make_from_ptr(gini_index)
    
#gini_index_wrapped = FuncWrapper.make_from_ptr(gini_index)
def variance_wrapped():
    return FuncWrapper.make_from_ptr(variance)

