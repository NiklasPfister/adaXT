# cython: profile=True

import numpy as np
from cython cimport boundscheck, wraparound

from ._func_wrapper cimport FuncWrapper

# int for y
cdef double gini_index(double[:, ::1] x, double[:] y):
    """
    Calculates the gini coefficient given outcomes, y.

    Parameters
    ----------
    x : npt.NDArray
        features, not used in this implementation
    y : npt.NDArray
        1-dimensional outcomes

    Returns
    -------
    double  
        The gini coefficient
    """
    cdef double[:] class_labels = np.unique(y)
    
    cdef double sum = 0.0
    cdef int y_len = y.shape[0]
    cdef int n_in_class
    cdef double proportion_cls 
    cdef double cls

    for cls in class_labels:
        n_in_class = count_class_occ(y, cls)
        proportion_cls = n_in_class / y_len
        sum += proportion_cls**2
    return 1 - sum  

cdef int count_class_occ(double[:] lst, double element):
    cdef int count, i, length 
    count = 0
    length = lst.shape[0]
    for i in range(length):
        if lst[i] == element:
            count += 1
    return count

cdef double variance(double[:, ::1] x, double[:] y):
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
    cdef double mu = mean(y)
    cdef int i 
    cdef int y_len = y.shape[0]

    with boundscheck(False), wraparound(False):    
        for i in range(y_len):
            cur_sum += (y[i] - mu)**2

    return cur_sum / y_len

cdef double mean(double[:] lst):
    cdef double sum = 0.0
    cdef int i 
    cdef int length = lst.shape[0]

    with boundscheck(False), wraparound(False): 
        for i in range(length):
            sum += lst[i]
    return sum / length

def gini_index_wrapped():
    return FuncWrapper.make_from_ptr(gini_index)
#gini_index_wrapped = FuncWrapper.make_from_ptr(gini_index)
variance_wrapped = FuncWrapper.make_from_ptr(variance)
