# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

cdef double mean(double[::1] lst, int[:] indices):
    '''
    Function that calculates the mean of a dataset
        ----------

        Parameters
        ----------
        lst : memoryview of NDArray
            The values to calculate the mean for

        indices : memoryview of NDArray
            The indices to calculate the mean at

        Returns
        -------
        double
            The mean of lst
    '''
    cdef double sum = 0.0
    cdef int i
    cdef int length = indices.shape[0]

    for i in range(length):
        sum += lst[indices[i]]
    return sum / (<double> length)
