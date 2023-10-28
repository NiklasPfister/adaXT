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
            int n_left = left_indices.shape[0]
            int n_right = right_indices.shape[0]
        
        left_indices = indices[:split_idx] 
        right_indices = indices[split_idx:]
        n_left =  left_indices.shape[0]
        n_right = right_indices.shape[0]

        # calculate criteria value on the left dataset
        if n_left != 0.0:
            left_imp = self.impurity(left_indices)
        crit = left_imp * (<double > n_left)/ (<double> n_indices)

        # calculate criteria value on the right dataset
        if n_right != 0.0:
            right_imp = self.impurity(right_indices)
        crit += (right_imp) * (<double> n_right)/(<double> n_indices)

        mean_thresh = (features[indices[split_idx-1], feature] + features[indices[split_idx], feature]) / 2.0
        
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

    cdef:
        double old_left_square_sum
        double old_left_sum
        double old_right_square_sum
        double old_right_sum
        int old_obs
        int old_split
        int old_feature

    def __init__(self) -> None:
        self.old_obs = -1
    
    cdef double update_left(self, int[:] indices, int new_split):
        # All new values in node from before
        cdef:
            int i, start_idx
            double tmp, square_sum, cur_sum, new_mu, n_obs
        n_obs = <double> new_split
        square_sum = self.old_left_square_sum
        cur_sum = self.old_left_sum
        start_idx = self.old_split
        for i in range(start_idx, new_split):
            tmp = self.y[indices[i]] 
            square_sum += tmp * tmp
            cur_sum += tmp
        self.old_left_square_sum = square_sum
        self.old_left_sum = cur_sum
        new_mu = cur_sum / n_obs
        return (square_sum/n_obs - new_mu*new_mu)

    cdef double update_right(self, int[:] indices, int new_split):
        cdef:
            int i, start_idx
            double tmp, square_sum, cur_sum, new_mu, n_obs, square_err
        n_obs = <double> (indices.shape[0] - new_split)
        square_sum = self.old_right_square_sum
        cur_sum = self.old_right_sum
        start_idx = self.old_split
        for i in range(start_idx, new_split):
            tmp = self.y[indices[i]] 
            square_sum -= tmp * tmp
            cur_sum -= tmp
        self.old_right_square_sum = square_sum
        self.old_right_sum = cur_sum
        new_mu = cur_sum / n_obs
        return (square_sum/n_obs - new_mu*new_mu)
    
    cpdef double impurity(self, int[:] indices):
        return self._square_error(indices)

    # Override the default evaluate_split
    cdef (double, double, double, double) evaluate_split(self, int[:] indices, int split_idx, int feature):
        cdef:
            double mean_thresh
            double left_imp = 0.0
            double right_imp = 0.0
            double crit = 0.0
            double n_left = <double> split_idx
            int n_obs = indices.shape[0] # total in node
            double n_right = (<double> n_obs) - n_left

        if n_obs == self.old_obs and feature == self.old_feature: # If we are checking the same node with same sorting
            left_imp = self.update_left(indices, split_idx)
            right_imp = self.update_right(indices, split_idx)
        else:
            left_imp = self._square_error(indices[:split_idx], 0)
            right_imp = self._square_error(indices[split_idx:], 1)

        self.old_feature = feature
        self.old_obs = n_obs
        self.old_split = split_idx
        crit = left_imp * n_left/ (<double> n_obs)
        crit += right_imp * (n_right)/(<double> n_obs)

        mean_thresh = (self.x[indices[split_idx-1], feature] + self.x[indices[split_idx], feature]) / 2.0
        if n_obs == 3:
            print("left_imp, right_imp, indices, split, crit, mean_thresh", left_imp, right_imp, indices.base, split_idx, crit, mean_thresh)
        return (crit, left_imp, right_imp, mean_thresh)

    cdef double _square_error(self, int[:] indices, int left_or_right = -1):
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
            double square_err, tmp
            int i 
            int n_obs = indices.shape[0]
            int n_indices = indices.shape[0]
        # Calculate the variance using: variance = sum((y_i - mu)^2)/y_len
        for i in range(n_indices):
            tmp = y[indices[i]]
            cur_sum += tmp*tmp
        square_err = cur_sum/(<double> n_obs) - mu*mu
        if left_or_right != -1:
            # Left subnode
            if left_or_right == 0: 
                self.old_left_sum = mu * (<double> n_obs)
                self.old_left_square_sum = cur_sum
            # Right subnode
            elif left_or_right == 1:
                self.old_right_sum = mu*(<double> n_obs)
                self.old_right_square_sum = cur_sum
        return square_err

