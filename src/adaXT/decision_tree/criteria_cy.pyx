# cython: profile=True

import numpy as np
from cython cimport boundscheck, wraparound, profile
cimport numpy as cnp
cnp.import_array()

from func_wrapper cimport FuncWrapper

gini_index_wrapped = FuncWrapper.make_from_ptr(gini_index)
variance_wrapped = FuncWrapper.make_from_ptr(variance)

'''
cdef double gini_index(double[:, ::1] x, double[:] y, int[:] indices, double* class_labels, int* n_in_class) nogil:
    cdef:
        double sum = 0.0
        int n_obs = indices.shape[0]
        int n_classes = 0
        double proportion_cls 
        int seen
        double[:] 

    with boundscheck(False), wraparound(False):
        for i in range(n_obs):
            seen = 0
            for j in range(n_classes):
                if class_labels[j] == y[indices[i]]:
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
    return 1 - sum  



@wraparound(False)
@boundscheck(False)
cdef double gini_index(double[:, ::1] x, double[::1] y, int[::1] indices):
    cdef:
        double gini_coef = 1.0
        int N = indices.shape[0]
        int i, j
        double[:] seen_already = np.empty(N, dtype=np.double)
        int[:] num_seen_already = np.zeros(N, dtype=np.int32)
        int num_seen = 0
        bint skip_y_i = 0
        double p_i
    
    for i in range(N):
        for j in range(num_seen):
            if(y[indices[i]] == seen_already[j]):
                skip_y_i = 1
                num_seen_already[j] = num_seen_already[j] + 1
                break
        if skip_y_i:
            skip_y_i = 0
            continue
        seen_already[num_seen] = y[indices[i]]
        num_seen_already[num_seen] = 1
        num_seen = num_seen + 1
    
    for i in range(num_seen):
        p_i = num_seen_already[i] / N
        gini_coef = gini_coef - p_i * p_i
    
    #print(list(y))
    #print(list(num_seen_already))
    #print(list(seen_already), "\n")
    return gini_coef


Sort y and then run through list and see when it changes
@wraparound(False)
@boundscheck(False)
cdef double gini_index(double[:, ::1] x, double[::1] y, int[::1] indices):
    cdef double[::1] sorted_result = np.sort(y.base[indices])
    cdef int num_seen = 1
    cdef int i
    cdef int N = sorted_result.shape[0]
    cdef double gini_coef = 1
    cdef double p_i

    for i in range(N - 1):
        if sorted_result[i] == sorted_result[i + 1]:
            num_seen += 1
        else:
            p_i = num_seen / N
            gini_coef -= p_i * p_i
            num_seen = 1

    p_i = num_seen / N
    gini_coef -= p_i * p_i

    return gini_coef


find unique classes of y and then count occurence of each class and subtract that from gini_index
@wraparound(False)
@boundscheck(False)
cdef double gini_index(double[:, ::1] x, double[::1] y, int[::1] indices):
    cdef:
        double gini_coef = 1.0
        int N = indices.shape[0]
        int i, j
        double[:] seen_already = np.empty(N, dtype=np.double)  
        int n_seen = 0 
        int num_seen = 0
        bint skip_y_i = 0
    
    for i in range(N):
        for j in range(n_seen):
            if y[indices[i]] == seen_already[j]:
                skip_y_i = 1
                break
        if skip_y_i:
            skip_y_i = 0
            continue
        seen_already[n_seen] = y[indices[i]]
        n_seen += 1

    for i in range(n_seen):
        for j in range(N):
            if seen_already[i] == y[indices[j]]:
                num_seen += 1
        gini_coef -= (num_seen / N) * (num_seen / N) 
        num_seen = 0
    
    return gini_coef

'''
#Loop over y and if you encounter an element not seen yet then count occurances and subtract from gini
@wraparound(False)
@boundscheck(False)
cdef double gini_index(double[:, ::1] x, double[::1] y, int[::1] indices):
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


@wraparound(False)
@boundscheck(False)
cdef double variance(double[:, ::1] x, double[::1] y, int[::1] indices):
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

    for i in range(n_indices):
        cur_sum += (y[indices[i]] - mu) * (y[indices[i]] - mu)

    return cur_sum / y_len   

@wraparound(False)
@boundscheck(False)
cdef double mean(double[::1] lst, int[::1] indices):
    cdef double sum = 0.0
    cdef int i 
    cdef int length = indices.shape[0]

    for i in range(length):
        sum += lst[indices[i]]
    return sum / length