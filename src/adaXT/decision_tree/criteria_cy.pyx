# cython: profile=True

import numpy as np
from cython cimport boundscheck, wraparound
cimport numpy as cnp
cnp.import_array()
from cpython cimport set, list

from func_wrapper cimport FuncWrapper

gini_index_wrapped = FuncWrapper.make_from_ptr(gini_index)
variance_wrapped = FuncWrapper.make_from_ptr(variance)

@wraparound(False)
@boundscheck(False)
cdef double gini_index(double[:, ::1] x, double[:] y, int[:] indices):
    cdef:
        double gini_coef = 1.0
        int N = indices.shape[0]
        int i, j
        double p_i 
        double[:] seen_already = np.empty(N, dtype=np.double)  
        int n_seen = 0 
        int num_seen = 0
        bint skip_y_i = 0
    
    for i in range(N):
        for j in range(n_seen):
            if(y[indices[i]] == seen_already[j]):
                skip_y_i = 1
                break
        if skip_y_i == 1:
            skip_y_i = 0
            continue
        for j in range(N):
            if(y[indices[i]] == y[indices[j]]):
                num_seen += 1
        p_i = num_seen / N
        gini_coef -= p_i * p_i
        seen_already[n_seen] = y[indices[i]]
        n_seen += 1
        num_seen = 0
    
    return gini_coef

    
'''
cdef double gini_index(double[:, ::1] x, double[:] y, int[:] indices):
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
    cdef double[:] class_labels = unique_double(y, indices)
    
    cdef double sum = 0.0
    cdef int len = indices.shape[0]
    cdef int n_in_class
    cdef double proportion_cls 
    cdef double cls

    for cls in class_labels:
        n_in_class = count_class_occ(y, cls, indices)
        proportion_cls = n_in_class / len
        sum += proportion_cls * proportion_cls
    return 1 - sum
''' 

@wraparound(False)
@boundscheck(False)
cdef double[:] unique_double(double[:] arr, int[:] indices):
    '''cdef int i
    cdef int n = indices.shape[0]
    cdef cnp.ndarray ret_val = np.empty(n, dtype=np.double)

    for i in range(n):
        for j in range(n):
            if arr[indices[i]] == ret
    return np.array(list(s))'''
    return np.unique(arr.base[indices])

# counts how many times an element occurs in a list in the given indices
cdef int count_class_occ(double[:] lst, double element, int[:] indices):
    cdef int count, i, length 
    count = 0
    length = indices.shape[0]
    for i in range(length):
        if lst[indices[i]] == element:
            count += 1
    return count

cdef double variance(double[:, ::1] x, double[:] y, int[:] indices):
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
    cdef int y_len = y.shape[0]
    cdef int n_indices = indices.shape[0]

    with boundscheck(False), wraparound(False):    
        for i in range(n_indices):
            cur_sum += (y[indices[i]] - mu) * (y[indices[i]] - mu)

    return cur_sum / y_len

cdef double mean(double[:] lst, int[:] indices):
    cdef double sum = 0.0
    cdef int i 
    cdef int length = indices.shape[0]

    with boundscheck(False), wraparound(False): 
        for i in range(length):
            sum += lst[indices[i]]
    return sum / length