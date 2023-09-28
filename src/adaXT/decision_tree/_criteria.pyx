# cython: profile=True, boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport fabs as cabs

from ._func_wrapper cimport FuncWrapper

cdef double gini_index(double[:, ::1] x, double[:] y, int[:] indices, double* class_labels, int* n_in_class):
    cdef:
        double sum = 0.0
        int n_obs = indices.shape[0]
        int n_classes = 0
        double proportion_cls 
        int seen
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
        proportion_cls = (<double> n_in_class[i]) / (<double> n_obs)
        sum += proportion_cls*proportion_cls
    return 1 - sum  

cdef double variance(double[:, ::1] x, double[:] y, int[:] indices, double* class_labels, int* n_in_class):
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

    return (<double> cur_sum) / (<double> y_len)   

cdef double mean(double[:] lst, int[:] indices):
    cdef double sum = 0.0
    cdef int i 
    cdef int length = indices.shape[0]

    for i in range(length):
        sum += lst[indices[i]]
    return (<double> sum) / (<double> length)

def gini_index_wrapped():
    return FuncWrapper.make_from_ptr(gini_index)

def variance_wrapped():
    return FuncWrapper.make_from_ptr(variance)