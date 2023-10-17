# cython: boundscheck=False, wraparound=False, cdivision=True

cdef int fill_class_lists(double[:] y, int[:] indices, double* class_labels, int* n_in_class):
    cdef:
        int n_classes = 0    
        int n_obs = indices.shape[0]
        int i,j
        bint seen
    
    # Loop over all the observations to calculate the gini index for
    for i in range(n_obs):
        seen = False
        # loop over all the classes we have seen so far
        for j in range(n_classes):
            # If the current element is one we have already seen, increase it's counter
            if class_labels[j] == y[indices[i]]:
                n_in_class[j] += 1
                seen = True
                break
        # If the current element has not been seen already add it to the elements seen already an start it's count.
        if (not seen):
            class_labels[n_classes] = y[indices[i]]
            n_in_class[n_classes] = 1
            n_classes += 1
    return n_classes

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
            mean of lst
    '''
    cdef double sum = 0.0
    cdef int i 
    cdef int length = indices.shape[0]

    for i in range(length):
        sum += lst[indices[i]]
    return sum / (<double> length)