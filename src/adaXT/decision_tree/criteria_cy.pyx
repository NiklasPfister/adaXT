# cython: profile=True

import numpy as np
from cython cimport boundscheck, wraparound
from libcpp.list cimport list as cpplist
cimport numpy as cnp
cnp.import_array()

from ._func_wrapper cimport FuncWrapper

# int for y
cdef double gini_index(double[:, ::1] x, double[:] y, int len) nogil:
    cdef:
        double sum = 0.0
        int y_len = y.shape[0]
        int n_len = 0
        cpplist[double] class_labels
        cpplist[int] n_in_class
        double proportion_cls 
        double epsilon = 2e-30
        int seen
    class_push_back = class_labels.push_back
    n_push_back = n_in_class.push_back
    for i in range(y_len):
        seen = 0
        for j in range(n_len):
            if (class_labels[j] - y[i]) < epsilon:
                n_in_class[i] += 1
                seen = 1
                break
        if (seen == 0):
            class_push_back(y[i])
            n_push_back(0)
            n_len += 1


    for i in range(n_len):
        proportion_cls = n_in_class[i] / y_len
        sum += proportion_cls**2
    return 1 - sum  

cdef double variance(double[:, ::1] x, double[:] y, int len):
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
    cdef int n_indices = y.shape[0]

    with boundscheck(False), wraparound(False):    
        for i in range(n_indices):
            cur_sum += (y[i] - mu) * (y[i]- mu)

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
def variance_wrapped():
    return FuncWrapper.make_from_ptr(variance)


def call_criteria_wrapped(crit: FuncWrapper, x: np.ndarray, y: np.ndarray):
    return crit.func(x, y)
