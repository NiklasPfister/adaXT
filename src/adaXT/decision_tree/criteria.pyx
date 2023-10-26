# cython: profile=True, boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport fabs as cabs
from libc.stdlib cimport hash, malloc, free
import numpy as np

from .func_wrapper cimport FuncWrapper
from .crit_helpers cimport *


cdef class Criterion:
    # Method used to calculate the impurity of a split at given indices
    cpdef double impurity(self, double[:, ::1] x, double[:] y, int[:] indices):
        raise NotImplementedError("Sub Classes of Criterion must implement the impurity method")


cdef class gini_index(Criterion):
    cdef int num_responses
    cdef double* labels
    cdef int* num_in_class

    def __cinit__(self, x, y):
        self.num_responses = len(np.unique(y))
        self.labels = <double *> malloc(sizeof(double) * self.num_responses)
        self.num_in_class_left = <int *> malloc(sizeof(int) * self.num_responses)
        self.num_in_class_right = <int *> malloc(sizeof(int) * self.num_responses)

    cdef double impurity(self, double[:, ::1] x, double[:] y, int[:] indices):
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

        n_classes = fill_class_lists(y, indices, self.labels, self.num_in_class)

        for i in range(n_classes):
            proportion_cls = (<double> self.num_in_class[i]) / (<double> n_obs)
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

# Wrap gini_index using a FuncWrapper object
#def gini_index():
#    return FuncWrapper.make_from_ptr(_gini_index)

def gini_index_new():
    return FuncWrapper.make_from_ptr(_gini_index_new)

# Wrap variance using a FuncWrapper object
def squared_error():
    return FuncWrapper.make_from_ptr(_squared_error)


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