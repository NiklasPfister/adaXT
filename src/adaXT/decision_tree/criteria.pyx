# cython: profile=True, boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport log2, fabs
from libc.stdlib cimport hash, malloc, free

from .crit_helpers cimport fill_class_lists, mean

import numpy as np
cimport cython

cdef class Criteria:

    def set_x_and_y(self, double[:, ::1] x, double[:] y):
        self.x = x
        self.y = y

    cpdef double impurity(self, int[:] indices):
        raise Exception("Impurity must be implemented!")
        return 0.0

    @cython.cdivision(False)
    cdef (double, double, double, double) evaluate_split(self, int[:] indices, int split_idx, int feature):
        """
        Function to evaluate how good a split is
        ----------

        Parameters
        ----------
        indices: int[:]
            the indices of a given node

        split_idx: int
            the index of the split, such that left indices are indices[:split_idx] and right indices are indices[split_idx:]
        
        feature: int
            The current feature we are working on
            
        Returns 
        -----------
        (double, double, double, double)
            A quadruple containing the criteria value, the left impurity, the right impurity and the mean threshold between the two
            closest datapoints of the current feature
        """
        cdef:
            double mean_thresh
            double left_imp = 0.0
            double right_imp = 0.0
            double crit = 0.0
            int n_indices = indices.shape[0] # total in node
            double[:, ::1] features = self.x
            int[:] left_indices = indices[:split_idx] 
            int[:] right_indices = indices[split_idx:]
            int n_left =  left_indices.shape[0]
            int n_right = right_indices.shape[0]
        
        left_indices = indices[0:split_idx] 
        right_indices = indices[split_idx:n_indices]
        n_left =  left_indices.shape[0]
        n_right = right_indices.shape[0]

        # calculate criteria value on the left dataset
        print("Left_indices: ", list(left_indices), " Right_indices: ", list(right_indices))
        if n_left != 0:
            left_imp = self.impurity(left_indices)
        crit = left_imp * (n_left/n_indices)

        # calculate criteria value on the right dataset
        if n_right != 0:
            right_imp = self.impurity(right_indices)
        crit += (right_imp) * (n_right/n_indices)

        print("Left_imp: ", left_imp, "Right_imp: ", right_imp)

        mean_thresh = (features[left_indices[split_idx-1], feature] + features[right_indices[0], feature]) / 2
        
        return (crit, left_imp, right_imp, mean_thresh)


cdef class Gini_index(Criteria):
    cdef:
        double* class_labels
        int* n_in_class

    def __del__(self): # Called by garbage collector.
        free(self.class_labels)
        free(self.n_in_class)
    
    cdef void make_c_list(self, int[:] indices):
        n_class = np.unique(self.y.base[indices]).shape[0]
        self.class_labels = <double *> malloc(sizeof(double) * n_class)
        self.n_in_class = <int *> malloc(sizeof(int) * n_class)

    cpdef double impurity(self, int[:] indices):
        if self.class_labels == NULL:
            self.make_c_list(indices)
        return self._gini(indices)

    cdef double _gini(self, int[:] indices):
        cdef:
            double sum = 0.0
            int n_obs = indices.shape[0]
            int n_classes = 0
            double proportion_cls 
            int i
            double[:] y = self.y
        class_labels = self.class_labels
        n_in_class = self.n_in_class

        n_classes = fill_class_lists(y, indices, class_labels, n_in_class)

        for i in range(n_classes):
            proportion_cls = (<double> n_in_class[i]) / (<double> n_obs)
            sum += proportion_cls*proportion_cls
        return 1 - sum  
    
cdef class Entropy(Criteria):
    cdef:
        double* class_labels
        int* n_in_class

    def __del__(self): # Called by garbage collector.
        free(self.class_labels)
        free(self.n_in_class)
    
    cdef void make_c_list(self, int[:] indices):
        n_class = np.unique(self.y.base[indices]).shape[0]
        self.class_labels = <double *> malloc(sizeof(double) * n_class)
        self.n_in_class = <int *> malloc(sizeof(int) * n_class)

    cpdef double impurity(self, int[:] indices):
        if self.class_labels == NULL:
            self.make_c_list(indices)
        return self._entropy(indices)

    cdef double _entropy(self, int[:] indices):
        cdef:
            int i
            double sum = 0
            int n_obs = indices.shape[0]
            double pp
            double[:] y = self.y

        class_labels = self.class_labels
        n_in_class = self.n_in_class

        n_classes = fill_class_lists(y, indices, class_labels, n_in_class)
        for i in range(n_classes):
            pp = (<double> n_in_class[i])/(<double> n_obs)
            sum += - (pp) * log2(pp)
        return sum

cdef class Squared_error(Criteria):

    cpdef double impurity(self, int[:] indices):
        return self._squared_error(indices)

    cdef double _squared_error(self, int[:] indices):
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
        cdef:
            double cur_sum = 0.0
            double[:] y = self.y
            double mu = mean(y, indices) # set mu to be the mean of the dataset
            double square_err
            int i 
            int y_len = y.shape[0]
            int n_indices = indices.shape[0]

        # Calculate the variance using: variance = sum((y_i - mu)^2)/y_len
        for i in range(n_indices):
            cur_sum += (y[indices[i]] - mu) * (y[indices[i]] - mu)

        square_err = (<double> cur_sum) / (<double> y_len) 

        return square_err
