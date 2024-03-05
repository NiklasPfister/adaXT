# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from libc.math cimport log2
from libc.stdlib cimport malloc, free
import numpy as np
from .crit_helpers cimport weighted_mean

cdef class Criteria:
    def __cinit__(self, double[:, ::1] x, double[::1] y, double[::1] sample_weight):
        self.x = x
        self.y = y
        self.sample_weight = sample_weight

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
            A quadruple containing the criteria value,
            the left impurity, the right impurity and
            the mean threshold between the two
            closest datapoints of the current feature
        """
        cdef:
            double mean_thresh
            double left_imp = 0.0
            double right_imp = 0.0
            double crit = 0.0
            int n_indices = indices.shape[0]  # total in node
            double[:, ::1] features = self.x
            int[:] left_indices = indices[:split_idx]
            int[:] right_indices = indices[split_idx:]
            int n_left = left_indices.shape[0]
            int n_right = right_indices.shape[0]

        left_indices = indices[:split_idx]
        right_indices = indices[split_idx:]
        n_left = left_indices.shape[0]
        n_right = right_indices.shape[0]

        # calculate criteria value on the left dataset
        if n_left != 0.0:
            left_imp = self.impurity(left_indices)
        crit = left_imp * (<double > n_left)/(<double> n_indices)

        # calculate criteria value on the right dataset
        if n_right != 0.0:
            right_imp = self.impurity(right_indices)
        crit += (right_imp) * (<double> n_right)/(<double> n_indices)

        mean_thresh = (features[indices[split_idx-1]][feature] + features[indices[split_idx]][feature]) / 2.0

        return (crit, left_imp, right_imp, mean_thresh)

cdef class Gini_index(Criteria):
    cdef:
        double[::1] class_labels
        double* n_in_class_left
        double* n_in_class_right
        double weight_left
        double weight_right
        int num_classes
        int old_obs
        int old_split
        int old_feature

    def __init__(self, double[:, ::1] x, double[::1] y, double[::1] sample_weight):
        self.old_obs = -1

    def __del__(self):  # Called by garbage collector.
        free(self.n_in_class_left)
        free(self.n_in_class_right)

    cpdef double impurity(self, int[:] indices):
        self.class_labels = np.unique(self.y.base[indices])
        self.num_classes = self.class_labels.shape[0]
        self.n_in_class_left = <double *> malloc(sizeof(double) * self.num_classes)
        self.n_in_class_right = <double *> malloc(sizeof(double) * self.num_classes)

        # n_in_class_left can be use as the int pointer as it will be cleared before and after this use
        return self._gini(indices, self.n_in_class_left, 1)

    cdef void reset_n_in_class(self, double* class_occurences):
        cdef int i
        for i in range(self.num_classes):
            class_occurences[i] = 0.0

    cdef double _gini(self, int[:] indices, double* class_occurences, int left_or_right):
        """
        Function that calculates the gini index of a dataset
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            The indices to calculate the gini index for

        class_occurences : double pointer
            A pointer to an double array for the number of elements seen of each class

        Returns
        -----------
        double
            The value of the gini index
        """
        self.reset_n_in_class(class_occurences)  # Reset the counter such that no previous values influence the new ones

        cdef:
            double sum = 0.0
            int n_obs = indices.shape[0]
            double obs_weight = 0.0
            double proportion_cls, weight
            int i, j, p
            double[:] y = self.y
            double[:] class_labels = self.class_labels

        for i in range(n_obs):  # loop over all indices
            for j in range(self.num_classes):  # Find the element we are currently on and increase it's counter
                p = indices[i]
                if y[p] == class_labels[j]:
                    weight = self.sample_weight[p]
                    class_occurences[j] += weight
                    obs_weight += weight

        # Loop over all classes and calculate gini_index
        for i in range(self.num_classes):
            proportion_cls = (class_occurences[i]) / obs_weight
            sum += proportion_cls * proportion_cls

        # update left or right weight
        if left_or_right == 1:
            self.weight_left = obs_weight
        else:
            self.weight_right = obs_weight
        return 1.0 - sum

    cdef double update_left(self, int[:] indices, int new_split):
        # All new values in node from before
        cdef:
            int i, j, p
            int start_idx = self.old_split
            double proportion_cls, weight
            double sum = 0.0

        for i in range(start_idx, new_split):  # loop over indices to be updated
            for j in range(self.num_classes):
                p = indices[i]
                if self.y[p] == self.class_labels[j]:
                    weight = self.sample_weight[p]
                    self.n_in_class_left[j] += weight
                    self.weight_left += weight
                    break

        # Loop over all classes and calculate gini_index
        for i in range(self.num_classes):
            proportion_cls = (self.n_in_class_left[i]) / self.weight_left
            sum += proportion_cls * proportion_cls

        return 1.0 - sum

    cdef double update_right(self, int[:] indices, int new_split):
        # All new values in node from before
        cdef:
            int i, j, p
            int start_idx = self.old_split
            int n_obs = indices.shape[0] - new_split
            double proportion_cls, weight
            double sum = 0.0

        for i in range(start_idx, new_split):  # loop over indices to be updated
            for j in range(self.num_classes):
                p = indices[i]
                if self.y[p] == self.class_labels[j]:
                    weight = self.sample_weight[p]
                    self.n_in_class_right[j] -= weight
                    self.weight_right -= weight
                    break

        # Loop over all classes and calculate gini_index
        for i in range(self.num_classes):
            proportion_cls = (self.n_in_class_right[i]) / (<double> n_obs)
            sum += proportion_cls * proportion_cls

        return 1.0 - sum

    # Override the default evaluate_split
    cdef (double, double, double, double) evaluate_split(self, int[:] indices, int split_idx, int feature):
        cdef:
            int n_obs = indices.shape[0]
            int n_left = split_idx
            int n_right = n_obs - n_left
            double mean_thresh
            double left_imp = 0.0
            double right_imp = 0.0
            double crit = 0.0
            double[:, ::1] features = self.x

        if n_obs == self.old_obs and feature == self.old_feature:  # If we are checking the same node with same sorting
            left_imp = self.update_left(indices, split_idx)
            right_imp = self.update_right(indices, split_idx)

        else:
            left_imp = self._gini(indices[:split_idx], self.n_in_class_left, 1)
            right_imp = self._gini(indices[split_idx:], self.n_in_class_right, 0)

        self.old_feature = feature
        self.old_obs = n_obs
        self.old_split = split_idx
        crit = left_imp * n_left / n_obs
        crit += right_imp * n_right / n_obs

        mean_thresh = (features[indices[split_idx-1]][feature] + features[indices[split_idx]][feature]) / 2.0
        return (crit, left_imp, right_imp, mean_thresh)

cdef class Entropy(Criteria):
    cdef:
        double[::1] class_labels
        double* n_in_class_left
        double* n_in_class_right
        double weight_left
        double weight_right
        int num_classes
        int old_obs
        int old_split
        int old_feature

    def __init__(self, double[:, ::1] x, double[::1] y, double[::1] sample_weight):
        self.old_obs = -1

    def __del__(self):  # Called by garbage collector.
        free(self.n_in_class_left)
        free(self.n_in_class_right)

    cpdef double impurity(self, int[:] indices):
        self.class_labels = np.unique(self.y.base[indices])
        self.num_classes = self.class_labels.shape[0]
        self.n_in_class_left = <double *> malloc(sizeof(double) * self.num_classes)
        self.n_in_class_right = <double *> malloc(sizeof(double) * self.num_classes)
        # n_in_class_left can be use as the int pointer as it will be cleared before and after this use
        return self._entropy(indices, self.n_in_class_left, 1)

    cdef void reset_n_in_class(self, double* class_occurences):
        cdef int i
        for i in range(self.num_classes):
            class_occurences[i] = 0.0

    cdef double _entropy(self, int[:] indices, double* class_occurences, int left_or_right):
        """
        Function that calculates the entropy index of a dataset
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            The indices to calculate

        class_occurences : double pointer
            A pointer to an double array for the number of elements seen of each class

        Returns
        -----------
        double
            The value of the entropy index
        """
        self.reset_n_in_class(class_occurences)  # Reset the counter such that no previous values influence the new ones

        cdef:
            double sum = 0.0
            double obs_weight = 0.0
            int n_obs = indices.shape[0]
            double pp, weight
            int i, j, p
            double[:] y = self.y
            double[:] class_labels = self.class_labels

        for i in range(n_obs):  # loop over all indices
            for j in range(self.num_classes):  # Find the element we are currently on and increase it's counter
                p = indices[i]
                if y[p] == class_labels[j]:
                    weight = self.sample_weight[p]
                    obs_weight += weight
                    class_occurences[j] += weight

        # Loop over all classes and calculate entropy
        for i in range(self.num_classes):
            if class_occurences[i] == 0:  # To make sure we dont take log(0)
                continue
            pp = (class_occurences[i])/(obs_weight)
            sum += - (pp) * log2(pp)

        # Save total weight
        if left_or_right == 1:
            self.weight_left = obs_weight
        else:
            self.weight_right = obs_weight
        return sum

    cdef double update_left(self, int[:] indices, int new_split):
        # All new values in node from before
        cdef:
            int i, j, p
            int start_idx = self.old_split
            double pp, weight
            double sum = 0.0

        for i in range(start_idx, new_split):  # loop over indices to be updated
            for j in range(self.num_classes):
                p = indices[i]
                if self.y[p] == self.class_labels[j]:
                    weight = self.sample_weight[p]
                    self.n_in_class_left[j] += weight
                    self.weight_left += weight
                    break

        # Loop over all classes and calculate entropy
        for i in range(self.num_classes):
            if self.n_in_class_left[i] == 0:  # To make sure we dont take log(0)
                continue
            pp = (self.n_in_class_left[i])/(self.weight_left)
            sum -= (pp) * log2(pp)
        return sum

    cdef double update_right(self, int[:] indices, int new_split):
        # All new values in node from before
        cdef:
            int i, j, p
            int start_idx = self.old_split
            double pp, weight
            double sum = 0.0

        for i in range(start_idx, new_split):  # loop over indices to be updated
            for j in range(self.num_classes):
                p = indices[i]
                if self.y[p] == self.class_labels[j]:
                    weight = self.sample_weight[p]
                    self.n_in_class_right[j] -= weight
                    self.weight_right -= weight
                    break

        # Loop over all classes and calculate entropy
        for i in range(self.num_classes):
            if self.n_in_class_right[i] == 0.0:  # To make sure we dont take log(0)
                continue
            pp = (self.n_in_class_right[i])/(self.weight_right)
            sum += - (pp) * log2(pp)
        return sum

    # Override the default evaluate_split
    cdef (double, double, double, double) evaluate_split(self, int[:] indices, int split_idx, int feature):
        cdef:
            int n_obs = indices.shape[0]
            int n_left = split_idx
            int n_right = n_obs - n_left
            double mean_thresh
            double left_imp = 0.0
            double right_imp = 0.0
            double crit = 0.0
            double[:, ::1] features = self.x

        if n_obs == self.old_obs and feature == self.old_feature:  # If we are checking the same node with same sorting
            left_imp = self.update_left(indices, split_idx)
            right_imp = self.update_right(indices, split_idx)
        else:
            left_imp = self._entropy(indices[:split_idx], self.n_in_class_left, 1)
            right_imp = self._entropy(indices[split_idx:], self.n_in_class_right, 0)

        self.old_feature = feature
        self.old_obs = n_obs
        self.old_split = split_idx
        crit = left_imp * n_left / n_obs
        crit += right_imp * n_right / n_obs

        mean_thresh = (features[indices[split_idx-1]][feature] + features[indices[split_idx]][feature]) / 2.0
        return (crit, left_imp, right_imp, mean_thresh)

cdef class Squared_error(Criteria):
    cdef:
        double left_square_sum
        double left_sum
        double right_square_sum
        double right_sum
        double weight_left
        double weight_right
        int old_obs
        int old_split
        int old_feature

    def __init__(self, double[:, ::1] x, double[:] y, double[::1] sample_weight):
        self.old_obs = -1

    cdef (double, double) update_proxy(self, int[:] indices, int new_split):
        cdef:
            int i, idx
            double y_val, weight

        for i in range(self.old_split, new_split):
            idx = indices[i]
            y_val = self.y[idx]
            weight = self.sample_weight[idx]
            self.left_square_sum += y_val * y_val
            self.weight_left += weight
            self.right_square_sum -= y_val * y_val
            self.weight_right -= weight

        return (self.left_square_sum / self.weight_left,
                self.right_square_sum / self.weight_right)

    cdef (double, double) proxy_improvement(self, int[:] indices, int split_idx):
        cdef:
            int i, idx
            double proxy_improvement_left, proxy_improvement_right
            int n_obs = indices.shape[0]
            double y_val
        self.left_square_sum = 0.0
        self.right_square_sum = 0.0
        self.weight_left = 0.0
        self.weight_right = 0.0

        for i in range(split_idx):
            idx = indices[i]
            y_val = self.y[idx]
            self.left_square_sum += y_val*y_val
            self.weight_left += self.sample_weight[idx]

        for i in range(split_idx, n_obs):
            idx = indices[i]
            y_val = self.y[idx]
            self.right_square_sum += y_val*y_val
            self.weight_right += self.sample_weight[idx]

        return (self.left_square_sum / self.weight_left,
                self.right_square_sum / self.weight_right)

    # Override the default evaluate_split
    cdef (double, double, double, double) evaluate_split(self, int[:] indices, int split_idx, int feature):
        cdef:
            int n_obs = indices.shape[0]
            int n_left = split_idx
            int n_right = n_obs - n_left
            double mean_thresh
            double left_imp = 0.0
            double right_imp = 0.0
            double crit = 0.0
            double[:, ::1] features = self.x

        if n_obs == self.old_obs and feature == self.old_feature:  # If we are checking the same node with same sorting
            left_imp, right_imp = self.update_proxy(indices, split_idx)

        else:
            left_imp, right_imp = self.proxy_improvement(indices, split_idx)

        self.old_feature = feature
        self.old_obs = n_obs
        self.old_split = split_idx
        crit = left_imp + right_imp

        mean_thresh = (features[indices[split_idx-1]][feature] + features[indices[split_idx]][feature]) / 2.0
        return (crit, left_imp, right_imp, mean_thresh)

    cpdef double impurity(self, int[:] indices):
        return self._squared_error(indices)

    cdef double _squared_error(self, int[:] indices):
        """
        Function used to calculate the squared error of y[indices]
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            The indices to calculate

        left_or_right : int
            An int indicating whether we are calculating on the left or right dataset

        Returns
        -------
        double
            The variance of the response y
        """
        cdef:
            double cur_sum = 0.0
            double[::1] y = self.y
            double mu = weighted_mean(y, indices, self.sample_weight)  # set mu to be the mean of the dataset
            double square_err, tmp
            double obs_weight = 0.0
            int i, p
            int n_indices = indices.shape[0]
        # Calculate the variance using: variance = sum((y_i - mu)^2)/y_len
        for i in range(n_indices):
            p = indices[i]
            tmp = y[p] * self.sample_weight[p]
            cur_sum += tmp*tmp
            obs_weight += self.sample_weight[p]
        square_err = cur_sum/obs_weight - mu*mu
        return square_err

cdef class Linear_regression(Criteria):

    # Custom mean function, such that we don't have to loop through twice.
    cdef (double, double) custom_mean(self, int[:] indices):
        cdef:
            double sumX, sumY
            int i
            int length = indices.shape[0]
        sumX = 0.0
        sumY = 0.0
        for i in range(length):
            sumX += self.x[indices[i], 0]
            sumY += self.y[indices[i]]

        return ((sumX / (<double> length)), (sumY/ (<double> length)))

    cdef (double, double) theta(self, int[:] indices):
        """
        Calculate theta0 and theta1 by a linear regression
        of Y on X[:, 0]
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            indices included in calculation

        Returns
        -------
        (double, double)
            where the first element is theta0 and second element is theta1
        """
        cdef:
            double muX, muY, theta0, theta1
            int length, i
            double numerator, denominator
            double X_diff

        length = indices.shape[0]
        denominator = 0.0
        numerator = 0.0
        muX, muY = self.custom_mean(indices)
        for i in range(length):
            X_diff = self.x[indices[i], 0] - muX
            numerator += (X_diff)*(self.y[indices[i]]-muY)
            denominator += (X_diff)*X_diff
        if denominator == 0.0:
            theta1 = 0.0
        else:
            theta1 = numerator / denominator
        theta0 = muY - theta1*muX
        return (theta0, theta1)

    cpdef double impurity(self, int[:] indices):
        """
        Calculates the impurity of a node by
        L = sum_{i in indices} (Y[i] - theta0 - theta1 X[i, 0])^2
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            The indices to calculate

        Returns
        -------
        double
            evaluated impurity
        """
        cdef:
            double step_calc, theta0, theta1, cur_sum
            int i, length

        length = indices.shape[0]
        theta0, theta1 = self.theta(indices)
        cur_sum = 0.0
        for i in range(length):
            step_calc = self.y[indices[i]] - theta0 - theta1 * self.x[indices[i], 0]
            cur_sum += step_calc*step_calc
        return cur_sum
