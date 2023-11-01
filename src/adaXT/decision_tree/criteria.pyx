# cython: profile=True, boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport log2, fabs
from libc.stdlib cimport hash, malloc, free
import numpy as np

from .crit_helpers cimport mean

import numpy as np
cimport cython

cdef class Criteria:
    def __cinit__(self, double[:, ::1] x, double[::1] y):
        self.x = x
        self.y = y

    cpdef double impurity(self, int[:] indices):
        raise Exception("Impurity must be implemented!")

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

        mean_thresh = (features[indices[split_idx-1]][feature] + features[indices[split_idx]][feature]) / 2.0
        
        return (crit, left_imp, right_imp, mean_thresh)






cdef class Gini_index(Criteria):
    cdef:
        double[::1] class_labels
        int* n_in_class
        int* n_in_class_left
        int* n_in_class_right
        int num_classes
        int old_obs
        int old_split
        int old_feature
    
    def __init__(self, double[:, ::1] x, double[::1] y):
        self.class_labels = np.unique(y.base)
        self.num_classes = self.class_labels.shape[0]
        self.n_in_class = <int *> malloc(sizeof(int) * self.num_classes)
        self.n_in_class_left = <int *> malloc(sizeof(int) * self.num_classes)
        self.n_in_class_right = <int *> malloc(sizeof(int) * self.num_classes)
        self.old_obs = -1

    def __del__(self): # Called by garbage collector.
        free(self.n_in_class)
        free(self.n_in_class_left)
        free(self.n_in_class_right)

    cpdef double impurity(self, int[:] indices):
        return self._gini(indices, self.n_in_class)

    cdef void reset_n_in_class(self,int* n_in_class):
        cdef int i
        for i in range(self.num_classes):
            n_in_class[i] = 0    

    cdef double _gini(self, int[:] indices, int* n_in_class):
        """
        Function that calculates the gini index of a dataset
        ----------

        Parameters
        ----------        
        indices : memoryview of NDArray
            The indices to calculate the gini index for

        n_in_class : int pointer
            A pointer to an int array for the number of elements seen of each class
                
        Returns 
        -----------
        double
            The value of the gini index
        """
        self.reset_n_in_class(n_in_class) # Reset the counter such that no previous values influence the new ones

        cdef:
            double sum = 0.0
            int n_obs = indices.shape[0]
            double proportion_cls 
            int i, j
            double[:] y = self.y
        class_labels = self.class_labels

        for i in range(n_obs): # loop over all indices
            for j in range(self.num_classes): # Find the element we are currently on and increase it's counter
                if y[indices[i]] == class_labels[j]:
                    n_in_class[j] += 1

        # Loop over all classes and calculate gini_index 
        for i in range(self.num_classes):
            proportion_cls = (<double> n_in_class[i]) / (<double> n_obs)
            sum += proportion_cls * proportion_cls

        return 1 - sum  
    
    cdef double update_left(self, int[:] indices, int new_split):
        # All new values in node from before
        cdef:
            int i, j, start_idx
            int n_obs = new_split
            double tmp, proportion_cls
            double sum = 0.0
        start_idx = self.old_split

        for i in range(start_idx, new_split): # loop over indices to be updated
            tmp = self.y[indices[i]] 
            for j in range(self.num_classes):
                if tmp == self.class_labels[j]:
                    self.n_in_class_left[j] += 1
                    break
        
        # Loop over all classes and calculate gini_index 
        for i in range(self.num_classes):
            proportion_cls = (<double> self.n_in_class_left[i]) / (<double> n_obs)
            sum += proportion_cls * proportion_cls

        return 1 - sum  
    
    cdef double update_right(self, int[:] indices, int new_split):
        # All new values in node from before
        cdef:
            int i, j, start_idx
            int n_obs = indices.shape[0] - new_split
            double tmp, proportion_cls
            double sum = 0.0
        start_idx = self.old_split

        for i in range(start_idx, new_split): # loop over indices to be updated
            tmp = self.y[indices[i]] 
            for j in range(self.num_classes):
                if tmp == self.class_labels[j]:
                    self.n_in_class_right[j] -= 1
                    break
        
        # Loop over all classes and calculate gini_index 
        for i in range(self.num_classes):
            proportion_cls = (<double> self.n_in_class_right[i]) / (<double> n_obs)
            sum += proportion_cls * proportion_cls

        return 1 - sum  

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
            double[:, ::1] features = self.x

        if n_obs == self.old_obs and feature == self.old_feature: # If we are checking the same node with same sorting
            left_imp = self.update_left(indices, split_idx)
            right_imp = self.update_right(indices, split_idx)

        else:
            left_imp = self._gini(indices[:split_idx], self.n_in_class_left)
            right_imp = self._gini(indices[split_idx:], self.n_in_class_right)
        self.old_feature = feature
        self.old_obs = n_obs
        self.old_split = split_idx
        crit = left_imp * n_left / (<double> n_obs)
        crit += right_imp * n_right / (<double> n_obs)

        mean_thresh = (features[indices[split_idx-1]][feature] + features[indices[split_idx]][feature]) / 2.0
        return (crit, left_imp, right_imp, mean_thresh)





cdef class Squared_error(Criteria):
    cdef:
        double old_left_square_sum
        double old_left_sum
        double old_right_square_sum
        double old_right_sum
        int old_obs
        int old_split
        int old_feature

    def __init__(self, double[:, ::1] x, double[:] y):
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
            double[:, ::1] features = self.x

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

        mean_thresh = (features[indices[split_idx-1]][feature] + features[indices[split_idx]][feature]) / 2.0
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





cdef class Entropy(Criteria):
    cdef:
        double[::1] class_labels
        int* n_in_class
        int* n_in_class_left
        int* n_in_class_right
        int num_classes
        int old_obs
        int old_split
        int old_feature
    
    def __init__(self, double[:, ::1] x, double[::1] y):
        self.class_labels = np.unique(y.base)
        self.num_classes = self.class_labels.shape[0]
        self.n_in_class = <int *> malloc(sizeof(int) * self.num_classes)
        self.n_in_class_left = <int *> malloc(sizeof(int) * self.num_classes)
        self.n_in_class_right = <int *> malloc(sizeof(int) * self.num_classes)
        self.old_obs = -1

    def __del__(self): # Called by garbage collector.
        free(self.n_in_class)
        free(self.n_in_class_left)
        free(self.n_in_class_right)

    cpdef double impurity(self, int[:] indices):
        return self._entropy(indices, self.n_in_class)

    cdef void reset_n_in_class(self, int* n_in_class):
        cdef int i
        for i in range(self.num_classes):
            n_in_class[i] = 0

    cdef double _entropy(self, int[:] indices, int* n_in_class):
        """
        Function that calculates the gini index of a dataset
        ----------

        Parameters
        ----------        
        indices : memoryview of NDArray
            The indices to calculate the gini index for

        n_in_class : int pointer
            A pointer to an int array for the number of elements seen of each class
                
        Returns 
        -----------
        double
            The value of the gini index
        """
        self.reset_n_in_class(n_in_class) # Reset the counter such that no previous values influence the new ones

        cdef:
            double sum = 0.0
            int n_obs = indices.shape[0]
            double pp 
            int i, j
            double[:] y = self.y
        class_labels = self.class_labels

        for i in range(n_obs): # loop over all indices
            for j in range(self.num_classes): # Find the element we are currently on and increase it's counter
                if y[indices[i]] == class_labels[j]:
                    n_in_class[j] += 1

        # Loop over all classes and calculate entropy 
        for i in range(self.num_classes):
            if n_in_class[i] == 0: # To make sure we dont take log(0)
                continue
            pp = (<double> n_in_class[i])/(<double> n_obs)
            sum += - (pp) * log2(pp)
        return sum

    cdef double update_left(self, int[:] indices, int new_split):
        # All new values in node from before
        cdef:
            int i, j, start_idx
            int n_obs = new_split
            double tmp, pp
            double sum = 0.0
        start_idx = self.old_split

        for i in range(start_idx, new_split): # loop over indices to be updated
            tmp = self.y[indices[i]] 
            for j in range(self.num_classes):
                if tmp == self.class_labels[j]:
                    self.n_in_class_left[j] += 1
                    break
        
        # Loop over all classes and calculate entropy 
        for i in range(self.num_classes):
            if self.n_in_class_left[i] == 0: # To make sure we dont take log(0)
                continue
            pp = (<double> self.n_in_class_left[i])/(<double> n_obs)
            sum += - (pp) * log2(pp)
        return sum 
    
    cdef double update_right(self, int[:] indices, int new_split):
        # All new values in node from before
        cdef:
            int i, j, start_idx
            int n_obs = indices.shape[0] - new_split
            double tmp, pp
            double sum = 0.0
        start_idx = self.old_split

        for i in range(start_idx, new_split): # loop over indices to be updated
            tmp = self.y[indices[i]] 
            for j in range(self.num_classes):
                if tmp == self.class_labels[j]:
                    self.n_in_class_right[j] -= 1
                    break
        
        # Loop over all classes and calculate entropy 
        for i in range(self.num_classes):
            if self.n_in_class_right[i] == 0: # To make sure we dont take log(0)
                continue
            pp = (<double> self.n_in_class_right[i])/(<double> n_obs)
            sum += - (pp) * log2(pp)
        return sum 

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
            double[:, ::1] features = self.x

        if n_obs == self.old_obs and feature == self.old_feature: # If we are checking the same node with same sorting
            left_imp = self.update_left(indices, split_idx)
            right_imp = self.update_right(indices, split_idx)
        else:
            left_imp = self._entropy(indices[:split_idx], self.n_in_class_left)
            right_imp = self._entropy(indices[split_idx:], self.n_in_class_right)

        self.old_feature = feature
        self.old_obs = n_obs
        self.old_split = split_idx
        crit = left_imp * n_left / (<double> n_obs)
        crit += right_imp * n_right / (<double> n_obs)

        mean_thresh = (features[indices[split_idx-1]][feature] + features[indices[split_idx]][feature]) / 2.0
        return (crit, left_imp, right_imp, mean_thresh)