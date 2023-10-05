# cython: profile=True, boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport fabs as cabs

from ._func_wrapper cimport FuncWrapper

cdef double gini_index(double[:, ::1] x, double[:] y, int[:] indices, double* class_labels, int* n_in_class):
    """
    Function that calculates the gini index of a dataset
        ----------

        Parameters
        ----------
        x : memoryview of NDArray
            The feature values of the dataset
        
        y : memoryview of NDArray
            The response values of the dataset
        
        indices : memoryview of NDArray
            The indices to calculate the gini index for
        
        class_labels : double pointer
            A pointer to a double array for the class_labels 

        n_in_class : int pointer
            A pointer to an int array for the number of elements seen of each class
            
    Returns 
        -----------
        double
            The value of the gini index
    """

    cdef:
        double sum = 0.0
        int n_obs = indices.shape[0]
        int n_classes = 0
        double proportion_cls 
        int seen
    
    # Loop over all the observations to calculate the gini index for
    for i in range(n_obs):
        seen = 0
        # Loop over all the classes we have seen so far
        for j in range(n_classes):
            # If the current element is one we have already seen, increase it's counter
            if class_labels[j] == y[indices[i]]:
                n_in_class[j] = n_in_class[j] + 1
                seen = 1
                break
        # If the current element has not been seen already add it to the elements seen already and start it's count.
        if (seen == 0):
            class_labels[n_classes] = y[indices[i]]
            n_in_class[n_classes] = 1
            n_classes += 1
    # Loop over all the seen classes and calculate the gini index using: gini_index = 1 - sum(p_i^2) where p_i is the
    # probability that an element is in a given class.
    for i in range(n_classes):
        proportion_cls = (<double> n_in_class[i]) / (<double> n_obs)
        sum += proportion_cls*proportion_cls
    return 1 - sum  

cdef double variance(double[:, ::1] x, double[:] y, int[:] indices, double* class_labels, int* n_in_class):
    """
    Function that calculates the variance of a dataset
        ----------

        Parameters
        ----------
        x : memoryview of NDArray
            The feature values of the dataset
        
        y : memoryview of NDArray
            The response values of the dataset
        
        indices : memoryview of NDArray
            The indices to calculate the gini index for
        
        class_labels : double pointer
            A pointer to a double array for the class_labels 

        n_in_class : int pointer
            A pointer to an int array for the number of elements seen of each class

        Returns
        -------
        double
            The variance of the response y
    """
    cdef double cur_sum = 0
    cdef double mu = mean(y, indices) # set mu to be the mean of the dataset
    cdef int i 
    cdef int y_len = y.shape[0]
    cdef int n_indices = indices.shape[0]

    # Calculate the variance using: variance = sum((x_i - mu)^2)
    for i in range(n_indices):
        cur_sum += (y[indices[i]] - mu) * (y[indices[i]] - mu)

    return (<double> cur_sum) / (<double> y_len)   

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
    return (<double> sum) / (<double> length)

# Wrap gini_index using a FuncWrapper object
def gini_index_wrapped():
    return FuncWrapper.make_from_ptr(gini_index)

# Wrap variance using a FuncWrapper object
def variance_wrapped():
    return FuncWrapper.make_from_ptr(variance)
