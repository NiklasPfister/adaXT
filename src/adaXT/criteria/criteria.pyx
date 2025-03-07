from libc.math cimport log2
from libc.stdlib cimport malloc, free
from libc.string cimport memset
import numpy as np
from .crit_helpers cimport weighted_mean


# Abstract Criteria class
cdef class Criteria:
    def __init__(self, double[:, ::1] X, double[:, ::1] Y, double[::1] sample_weight):
        self.X = X
        self.Y = Y
        self.sample_weight = sample_weight
        self.old_obs = -1
        self.old_feature = -1

    cdef double update_proxy(self, int[::1] indices, int split_idx):
        return self.proxy_improvement(indices, split_idx)

    cdef double proxy_improvement(self, int[::1] indices, int split_idx):
        cdef:
            double left_imp = 0.0
            double right_imp = 0.0
            double crit = 0.0
            int[::1] left_indices = indices[:split_idx]
            int[::1] right_indices = indices[split_idx:]
            int n_left = left_indices.shape[0]
            int n_right = right_indices.shape[0]

        left_indices = indices[:split_idx]
        right_indices = indices[split_idx:]
        n_left = left_indices.shape[0]
        n_right = right_indices.shape[0]

        # calculate criteria value on the left dataset
        if n_left != 0.0:
            left_imp = self.impurity(left_indices)
        crit = left_imp * (<double > n_left)

        # calculate criteria value on the right dataset
        if n_right != 0.0:
            right_imp = self.impurity(right_indices)
        crit += (right_imp) * (<double> n_right)

        return crit

    cpdef double impurity(self, int[::1] indices):
        raise Exception("Impurity must be implemented!")

    cdef (double, double) evaluate_split(self, int[::1] indices, int split_idx, int feature):
        cdef:
            double mean_thresh
            int n_obs = indices.shape[0]  # total in node

        if n_obs == self.old_obs and feature == self.old_feature:  # If we are checking the same node with same sorting
            crit = self.update_proxy(indices, split_idx)

        else:
            crit = self.proxy_improvement(indices, split_idx)
            self.old_feature = feature
            self.old_obs = n_obs

        self.old_split = split_idx
        mean_thresh = (self.X[indices[split_idx-1]][feature] + self.X[indices[split_idx]][feature]) / 2.0

        return (crit, mean_thresh)

    @staticmethod
    def loss(double[:,  ::1] Y_pred, double[:, ::1]  Y_true, double[:, ::1] sample_weight) -> float:
        raise ValueError("Loss is not implemented for the given Criteria")


cdef class ClassificationCriteria(Criteria):
    def __init__(self, double[:, ::1] X, double[:, ::1] Y, double[::1]
                 sample_weight) -> None:
        super().__init__(X, Y, sample_weight)
        self.first_call = True

    def __del__(self) -> None:
        free(self.weight_in_class_left)
        free(self.weight_in_class_right)

    cdef void reset_weight_list(self, double* class_occurences):
        # Use memset to set the entire malloc to 0
        memset(class_occurences, 0, self.num_classes*sizeof(double))

    @staticmethod
    def loss(double[:,  ::1] Y_pred, double[:, ::1]  Y_true, double[::1] sample_weight) -> float:
        """ Zero one loss function """
        cdef:
            int i
            int n_samples = Y_pred.shape[0]
            double weighted_samples = 0.0
            double tot_sum = 0.0

        if Y_true.shape[0] != n_samples:
            raise ValueError(
                    "Y_pred and Y_true have different number of samples in loss"
                    )
        for i in range(n_samples):
            if Y_pred[i, 0] != Y_true[i, 0]:
                tot_sum += sample_weight[i]

            weighted_samples += sample_weight[i]

        return tot_sum / n_samples

# Gini index criteria
cdef class Gini_index(ClassificationCriteria):

    cpdef double impurity(self, int[::1] indices):
        if self.first_call:
            self.class_labels = np.unique(self.Y.base[indices, 0])
            self.num_classes = self.class_labels.shape[0]

            self.weight_in_class_right = <double *> malloc(sizeof(double) * self.num_classes)
            self.weight_in_class_left = <double *> malloc(sizeof(double) * self.num_classes)

            self.first_call = False

        return self.__gini(indices, self.weight_in_class_left)

    cdef double __gini(self, int[::1] indices, double* class_occurences):
        self.reset_weight_list(class_occurences)  # Reset the counter such that no previous values influence the new ones

        cdef:
            double sum = 0.0
            int n_obs = indices.shape[0]
            double obs_weight = 0.0
            double proportion_cls, weight
            int i, j, p
            double[:, ::1] Y = self.Y
            double[:] class_labels = self.class_labels

        for i in range(n_obs):  # loop over all indices
            for j in range(self.num_classes):  # Find the element we are currently on and increase it's counter
                p = indices[i]
                if Y[p, 0] == class_labels[j]:
                    weight = self.sample_weight[p]
                    class_occurences[j] += weight
                    obs_weight += weight
                    break

        # Loop over all classes and calculate gini_index
        for i in range(self.num_classes):
            proportion_cls = (class_occurences[i]) / obs_weight
            sum += proportion_cls * proportion_cls

        return 1.0 - sum

    cdef double update_proxy(self, int[::1] indices, int new_split):
        # All new values in node from before
        cdef:
            int i, j, p
            int start_idx = self.old_split
            double proportion_cls_left, proportion_cls_right, weight
            double sum_left = 0.0
            double sum_right = 0.0

        for i in range(start_idx, new_split):  # loop over indices to be updated
            for j in range(self.num_classes):
                p = indices[i]
                if self.Y[p, 0] == self.class_labels[j]:
                    weight = self.sample_weight[p]
                    self.weight_in_class_left[j] += weight
                    self.weight_left += weight
                    self.weight_in_class_right[j] -= weight
                    self.weight_right -= weight
                    break

        # Loop over all classes and calculate gini_index
        for i in range(self.num_classes):
            proportion_cls_left = (self.weight_in_class_left[i]) / self.weight_left
            proportion_cls_right = (self.weight_in_class_right[i]) / self.weight_right
            sum_left += proportion_cls_left * proportion_cls_left
            sum_right += proportion_cls_right * proportion_cls_right

        # No need to divide by the total weight, as the proxy proxy_improvement is always compared to itself
        return (1.0 - sum_left)*self.weight_left + (1.0 - sum_right)*self.weight_right

    cdef double proxy_improvement(self, int[::1] indices, int split_idx):
        cdef:
            int n_obs = indices.shape[0]
            double sum_left = 0.0
            double sum_right = 0.0
            double proportion_cls_left, proportion_cls_right, weight
            int i, j, p
            double[:, ::1] Y = self.Y
            double[:] class_labels = self.class_labels

        # Reset weights as we are in a new node
        self.reset_weight_list(self.weight_in_class_left)
        self.reset_weight_list(self.weight_in_class_right)
        self.weight_left = 0.0
        self.weight_right = 0.0

        for i in range(split_idx):
            for j in range(self.num_classes):
                p = indices[i]
                if Y[p, 0] == class_labels[j]:
                    weight = self.sample_weight[p]
                    self.weight_in_class_left[j] += weight
                    self.weight_left += weight
                    break

        for i in range(split_idx, n_obs):
            for j in range(self.num_classes):
                p = indices[i]
                if Y[p, 0] == class_labels[j]:
                    weight = self.sample_weight[p]
                    self.weight_in_class_right[j] += weight
                    self.weight_right += weight
                    break

        # Loop over all classes and calculate gini_index
        for i in range(self.num_classes):
            proportion_cls_left = (self.weight_in_class_left[i]) / self.weight_left
            proportion_cls_right = (self.weight_in_class_right[i]) / self.weight_right
            sum_left += proportion_cls_left * proportion_cls_left
            sum_right += proportion_cls_right * proportion_cls_right

        # No need to divide by the total weight, as the proxy proxy_improvement is always compared to itself
        return (1.0 - sum_left)*self.weight_left + (1.0 - sum_right)*self.weight_right


# Entropy criteria
cdef class Entropy(ClassificationCriteria):

    cpdef double impurity(self, int[::1] indices):
        if self.first_call:
            self.class_labels = np.unique(self.Y.base[indices, 0])
            self.num_classes = self.class_labels.shape[0]

            self.weight_in_class_right = <double *> malloc(sizeof(double) * self.num_classes)
            self.weight_in_class_left = <double *> malloc(sizeof(double) * self.num_classes)

            self.first_call = False

        # weight_in_class_left can be use as the int pointer as it will be cleared before and after this use
        return self.__entropy(indices, self.weight_in_class_left)

    cdef double __entropy(self, int[:] indices, double* class_occurences):
        self.reset_weight_list(class_occurences)  # Reset the counter such that no previous values influence the new ones

        cdef:
            double sum = 0.0
            double obs_weight = 0.0
            int n_obs = indices.shape[0]
            double pp, weight
            int i, j, p
            double[:, ::1] Y = self.Y
            double[:] class_labels = self.class_labels

        for i in range(n_obs):  # loop over all indices
            for j in range(self.num_classes):  # Find the element we are currently on and increase it's counter
                p = indices[i]
                if Y[p, 0] == class_labels[j]:
                    weight = self.sample_weight[p]
                    obs_weight += weight
                    class_occurences[j] += weight
                    break

        # Loop over all classes and calculate entropy
        for i in range(self.num_classes):
            if class_occurences[i] == 0:  # To make sure we dont take log(0)
                continue
            pp = (class_occurences[i])/(obs_weight)
            sum -= (pp) * log2(pp)

        return sum

    cdef double proxy_improvement(self, int[::1] indices, int split_idx):
        cdef:
            double sum_left = 0.0
            double sum_right = 0.0
            int n_obs = indices.shape[0]
            double weight
            int i, j, p
            double[:, ::1] Y = self.Y
            double[:] class_labels = self.class_labels

        # Reset weights as we are in a new node
        self.reset_weight_list(self.weight_in_class_left)
        self.reset_weight_list(self.weight_in_class_right)
        self.weight_left = 0.0
        self.weight_right = 0.0

        for i in range(split_idx):
            for j in range(self.num_classes):
                p = indices[i]
                if Y[p, 0] == class_labels[j]:
                    weight = self.sample_weight[p]
                    self.weight_in_class_left[j] += weight
                    self.weight_left += weight
                    break

        for i in range(split_idx, n_obs):
            for j in range(self.num_classes):
                p = indices[i]
                if Y[p, 0] == class_labels[j]:
                    weight = self.sample_weight[p]
                    self.weight_in_class_right[j] += weight
                    self.weight_right += weight
                    break

        # Loop over all classes and calculate entropy
        for i in range(self.num_classes):
            if self.weight_in_class_left[i] != 0.0:  # To make sure we dont take log(0)
                pp = (self.weight_in_class_left[i])/(self.weight_left)
                sum_left -= (pp) * log2(pp)

            if self.weight_in_class_right[i] != 0.0:
                pp = (self.weight_in_class_right[i])/(self.weight_right)
                sum_right -= (pp) * log2(pp)

        return sum_left*self.weight_left + sum_right*self.weight_right

    cdef double update_proxy(self, int[::1] indices, int new_split):
        # All new values in node from before
        cdef:
            int i, j, p
            int start_idx = self.old_split
            double weight
            double sum_left = 0.0
            double sum_right = 0.0

        for i in range(start_idx, new_split):  # loop over indices to be updated
            for j in range(self.num_classes):
                p = indices[i]
                if self.Y[p, 0] == self.class_labels[j]:
                    weight = self.sample_weight[p]
                    self.weight_in_class_left[j] += weight
                    self.weight_left += weight
                    self.weight_in_class_right[j] -= weight
                    self.weight_right -= weight
                    break

        # Loop over all classes and calculate entropy
        for i in range(self.num_classes):
            if self.weight_in_class_left[i] != 0.0:  # To make sure we dont take log(0)
                pp = (self.weight_in_class_left[i])/(self.weight_left)
                sum_left -= (pp) * log2(pp)

            if self.weight_in_class_right[i] != 0.0:
                pp = (self.weight_in_class_right[i])/(self.weight_right)
                sum_right -= (pp) * log2(pp)

        return sum_left*self.weight_left + sum_right*self.weight_right


cdef class RegressionCriteria(Criteria):
    @staticmethod
    def loss(double[:,  ::1] Y_pred, double[:, ::1] Y_true, double[::1] sample_weight) -> float:
        """ Mean squared error loss """
        cdef:
            int i
            int n_samples = Y_pred.shape[0]
            double weighted_samples = 0.0
            double temp
            double tot_sum = 0.0

        if Y_true.shape[0] != n_samples:
            raise ValueError(
                    "Y_pred and Y_true have different number of samples in loss"
                    )
        for i in range(n_samples):
            temp = (Y_true[i, 0] - Y_pred[i, 0])*sample_weight[i]
            weighted_samples += sample_weight[i]
            tot_sum += temp*temp

        return tot_sum / weighted_samples


# Squared error criteria
cdef class Squared_error(RegressionCriteria):

    cdef double update_proxy(self, int[::1] indices, int new_split):
        cdef:
            int i, idx
            double y_val, weight
        for i in range(self.old_split, new_split):
            idx = indices[i]
            weight = self.sample_weight[idx]
            y_val = self.Y[idx, 0]*weight
            self.left_sum += y_val
            self.right_sum -= y_val
            self.weight_left += weight
            self.weight_right -= weight

        return -((self.left_sum*self.left_sum) / self.weight_left +
                 (self.right_sum*self.right_sum) / self.weight_right)

    cdef double proxy_improvement(self, int[::1] indices, int split_idx):
        cdef:
            int i, idx
            int n_obs = indices.shape[0]
            double y_val, weight

        self.left_sum = 0.0
        self.right_sum = 0.0
        self.weight_left = 0.0
        self.weight_right = 0.0

        for i in range(split_idx):
            idx = indices[i]
            weight = self.sample_weight[idx]
            y_val = self.Y[idx, 0]*weight
            self.left_sum += y_val
            self.weight_left += weight

        for i in range(split_idx, n_obs):
            idx = indices[i]
            weight = self.sample_weight[idx]
            y_val = self.Y[idx, 0]*weight
            self.right_sum += y_val
            self.weight_right += weight

        return -((self.left_sum*self.left_sum) / self.weight_left +
                 (self.right_sum*self.right_sum) / self.weight_right)

    cpdef double impurity(self, int[::1] indices):
        return self.__squared_error(indices)

    cdef double __squared_error(self, int[::1] indices):
        cdef:
            double cur_sum = 0.0
            double[:, ::1] Y = self.Y
            double mu = weighted_mean(Y[:, 0], indices, self.sample_weight)  # set mu to be the mean of the dataset
            double square_err, tmp
            double obs_weight = 0.0
            int i, p
            int n_indices = indices.shape[0]
        # Calculate the variance using: variance = sum((y_i - mu)^2)/y_len
        for i in range(n_indices):
            p = indices[i]
            tmp = Y[p, 0] * self.sample_weight[p]
            cur_sum += tmp*tmp
            obs_weight += self.sample_weight[p]
        square_err = cur_sum/obs_weight - mu*mu
        return square_err


# Partial linear criteria
cdef class Partial_linear(RegressionCriteria):

    # Custom mean function, such that we don't have to loop through twice.
    cdef (double, double) __custom_mean(self, int[:] indices):
        cdef:
            double sumX, sumY
            int i
            int length = indices.shape[0]
        sumX = 0.0
        sumY = 0.0
        for i in range(length):
            sumX += self.X[indices[i], 0]
            sumY += self.Y[indices[i], 0]

        return ((sumX / (<double> length)), (sumY/ (<double> length)))

    cdef (double, double) __theta(self, int[:] indices):
        cdef:
            double muX, muY, theta0, theta1
            int length, i
            double numerator, denominator
            double X_diff

        length = indices.shape[0]
        denominator = 0.0
        numerator = 0.0
        muX, muY = self.__custom_mean(indices)
        for i in range(length):
            X_diff = self.X[indices[i], 0] - muX
            numerator += (X_diff)*(self.Y[indices[i], 0]-muY)
            denominator += (X_diff)*X_diff
        if denominator == 0.0:
            theta1 = 0.0
        else:
            theta1 = numerator / denominator
        theta0 = muY - theta1*muX
        return (theta0, theta1)

    cpdef double impurity(self, int[::1] indices):
        cdef:
            double step_calc, theta0, theta1, cur_sum
            int i, length

        length = indices.shape[0]
        theta0, theta1 = self.__theta(indices)
        cur_sum = 0.0
        for i in range(length):
            step_calc = self.Y[indices[i], 0] - theta0 - theta1 * self.X[indices[i], 0]
            cur_sum += step_calc * step_calc
        return cur_sum / length

cdef class Partial_quadratic(RegressionCriteria):

    # Custom mean function, such that we don't have to loop through twice.
    cdef (double, double, double) __custom_mean(self, int[:] indices):
        cdef:
            double sumXsq, sumX, sumY
            int i
            int length = indices.shape[0]
        sumX = 0.0
        sumXsq = 0.0
        sumY = 0.0
        for i in range(length):
            sumX += self.X[indices[i], 0]
            sumXsq += self.X[indices[i], 0] * self.X[indices[i], 0]
            sumY += self.Y[indices[i], 0]

        return ((sumX / (<double> length)), (sumXsq / (<double> length)), (sumY/ (<double> length)))

    cdef (double, double, double) __theta(self, int[:] indices):
        """
        Estimates regression parameters for a linear regression of the response
        on the first coordinate, i.e., Y is approximated by theta0 + theta1 *
        X[:, 0] + theta2 * X[:, 0] ** 2.
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            indices included in calculation

        Returns
        -------
        (double, double, double)
            where first element is theta0, second is theta1, third is theta2
            and fourth is the mean of Y
        """
        cdef:
            double muX, muXsq, muY, theta0, theta1, theta2
            int length, i
            double covXXsq, varX, varXsq, covXY, covXsqY
            double X_diff, Xsq_diff

        length = indices.shape[0]
        covXXsq = 0.0
        covXY = 0.0
        covXsqY = 0.0
        varX = 0.0
        varXsq = 0.0
        muX, muXsq, muY = self.__custom_mean(indices)
        for i in range(length):
            X_diff = self.X[indices[i], 0] - muX
            Xsq_diff = self.X[indices[i], 0] * self.X[indices[i], 0] - muXsq
            Y_diff = self.Y[indices[i], 0] - muY
            covXXsq += X_diff * Xsq_diff
            varX += X_diff * X_diff
            varXsq += Xsq_diff * Xsq_diff
            covXY += X_diff * Y_diff
            covXsqY += Xsq_diff * Y_diff
        det = varX * varXsq - covXXsq * covXXsq
        if det == 0.0:
            theta1 = 0.0
            theta2 = 0.0
        else:
            theta1 = (varXsq*covXY - covXXsq * covXsqY) / det
            theta2 = (varX*covXsqY - covXXsq * covXY) / det
        theta0 = muY - theta1*muX - theta2*muXsq
        return (theta0, theta1, theta2)

    cpdef double impurity(self, int[::1] indices):
        """
        Calculates the impurity of a node by
        L = sum_{i in indices} (Y[i] - theta0 - theta1*X[i, 0] - theta2*X[i, 0]**2)**2
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            indices included in calculation

        Returns
        -------
        double
            evaluated impurity
        """
        cdef:
            double step_calc, theta0, theta1, theta2, cur_sum
            int i, length

        length = indices.shape[0]
        theta0, theta1, theta2 = self.__theta(indices)
        cur_sum = 0.0
        for i in range(length):
            step_calc = self.Y[indices[i], 0] - theta0
            step_calc -= theta1 * self.X[indices[i], 0]
            step_calc -= theta2 * self.X[indices[i], 0] * self.X[indices[i], 0]
            cur_sum += step_calc * step_calc
        return cur_sum / length
