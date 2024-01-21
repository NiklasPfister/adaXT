# cython: boundscheck=False, wraparound=False, cdivision=True

cdef double mean(double[:] lst, int[:] indices):
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
    cdef:
        double sum = 0.0
        int i 
        int length = indices.shape[0]

    for i in range(length):
        sum += lst[indices[i]]
    return sum / (<double> length)


cdef double weighted_mean(double[:] lst, int[:] indices, double[:] sample_weights):
    '''
    Function that calculates the weighted mean of a dataset
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
    cdef:
        double sum = 0.0
        int i, p
        int length = indices.shape[0]
        double weight
        double weighted_obs = 0.0

    for i in range(length):
        p = indices[i]
        weight = sample_weights[p]
        sum += lst[p]*weight
        weighted_obs += weight
    return sum / weighted_obs
