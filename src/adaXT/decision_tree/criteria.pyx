# cython: profile=True, boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport log2
from libc.stdlib cimport hash, malloc, free

from .func_wrapper cimport FuncWrapper
from .crit_helpers cimport *

cdef double _gini_index(double[:, ::1] x, double[:] y, int[:] indices, double* class_labels, int* n_in_class):
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
        int i

    n_classes = fill_class_lists(y, indices, class_labels, n_in_class)

    for i in range(n_classes):
        proportion_cls = (<double> n_in_class[i]) / (<double> n_obs)
        sum += proportion_cls*proportion_cls
    return 1 - sum  

cdef double _squared_error(double[:, ::1] x, double[:] y, int[:] indices, double* class_labels, int* n_in_class):
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
    cdef double cur_sum = 0.0
    cdef double mu = mean(y, indices) # set mu to be the mean of the dataset
    cdef int i 
    cdef int y_len = y.shape[0]
    cdef int n_indices = indices.shape[0]

    # Calculate the variance using: variance = sum((y_i - mu)^2)/y_len
    for i in range(n_indices):
        cur_sum += (y[indices[i]] - mu) * (y[indices[i]] - mu)

    return (<double> cur_sum) / (<double> y_len)   

cdef double _entropy(double[:, ::1] x, double[:] y, int[:] indices, double* class_labels, int* n_in_class):
    """
    Function to calculate the entropy given the formula:
        Entropy(y) = sum_{i=0}^n -(p_i log_2(p_i))

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
        The entropy calculation
    """
    cdef:
        int i
        double sum  = 0
        int n_obs = indices.shape[0]
        double pi
     
    n_classes = fill_class_lists(y, indices, class_labels, n_in_class)
    for i in range(n_classes):
        pp = (<double> n_in_class[i])/(<double> n_obs)
        sum += - (pp) * log2(pp)
    return sum

cdef double _gini_index_new(double[:, ::1] x, double[:] y, int[:] indices, double* class_labels, int* n_in_class):
    cdef double* labels = <double *> malloc(sizeof(double) * 5)
    cdef int* num_in_class = <int *> malloc(sizeof(int) * 5)

    cdef:
        double sum = 0.0
        int n_obs = indices.shape[0]
        int n_classes = 0
        double proportion_cls 
        int i

    n_classes = fill_class_lists(y, indices, labels, num_in_class)

    for i in range(n_classes):
        proportion_cls = (<double> num_in_class[i]) / (<double> n_obs)
        sum += proportion_cls*proportion_cls

    free(labels)
    free(num_in_class)

    return 1 - sum  

# Wrap gini_index using a FuncWrapper object
def gini_index():
    return FuncWrapper.make_from_ptr(_gini_index)

def gini_index_new():
    return FuncWrapper.make_from_ptr(_gini_index_new)

# Wrap variance using a FuncWrapper object
def squared_error():
    return FuncWrapper.make_from_ptr(_squared_error)

def entropy():
    return FuncWrapper.make_from_ptr(_entropy)