# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from .nodes cimport LeafNode, LinearRegressionLeafNode
import numpy as np
cimport numpy as cnp

cdef class LeafBuilder:
    def __cinit__(self, double[:, ::1] x, double[::1] y, int[::1] all_idx, **kwargs):
       self.x = x
       self.y = y

    cpdef LeafNode build_leaf(self,
                             int leaf_id,
                             int[::1] indices,
                             int depth,
                             double impurity,
                             int n_samples,
                             object parent):
        raise NotImplementedError("Build leaf not implemented for this LeafBuilder")


cdef class LeafBuilderClassification(LeafBuilder):
    def __cinit__(self, double[:, ::1] x, double[::1] y, int[::1] all_idx, **kwargs):
        self.classes = np.array(np.unique(y.base[all_idx]), dtype=np.double)
        self.n_classes = self.classes.shape[0]

    cdef double[::1] __get_mean(self, int[::1] indices, int n_samples):
        cdef:
            cnp.ndarray[double, ndim=1] ret
            int i, idx

        ret = np.zeros(self.n_classes)
        for idx in range(n_samples):
            for i in range(self.n_classes):
                if self.y[indices[idx]] == self.classes[i]:
                    ret[i] += 1  # add 1, if the value is the same as class value
                    break

        ret = ret / n_samples
        return ret

    cpdef LeafNode build_leaf(self,
                             int leaf_id,
                             int[::1] indices,
                             int depth,
                             double impurity,
                             int n_samples,
                             object parent):
        cdef double[::1] mean = self.__get_mean(indices, n_samples)
        return LeafNode(leaf_id, indices, depth, impurity, n_samples, mean, parent)


cdef class LeafBuilderRegression(LeafBuilder):

    cdef double __get_mean(self, int[::1] indices):
        cdef:
            int i
            double sum = 0.0
            int count = len(indices)

        for i in indices:
           sum += self.y[i]

        return sum / count

    cpdef LeafNode build_leaf(self,
                             int leaf_id,
                             int[::1] indices,
                             int depth,
                             double impurity,
                             int n_samples,
                             object parent):

        cdef double[::1] mean = np.array(self.__get_mean(indices), dtype=np.double, ndmin=1)
        return LeafNode(leaf_id, indices, depth, impurity, n_samples, mean, parent)

cdef class LinearRegressionLeafBuilder(LeafBuilderRegression):

    # Custom mean function, such that we don't have to loop through twice.
    cdef (double, double) custom_mean(self, int[::1] indices):
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

    cdef (double, double, double) theta(self, int[::1] indices):
        """
        Calculate theta0 and theta1 used for a Linear Regression
        on X[:, 0] and Y
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            The indices to calculate

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
        return (theta0, theta1, muY)


    cpdef LeafNode build_leaf(self,
                             int leaf_id,
                             int[::1] indices,
                             int depth,
                             double impurity,
                             int n_samples,
                             object parent):
        cdef:
            double[::1] mean
            double theta0, theta1

        theta0, theta1, muY = self.theta(indices)
        mean = np.array(muY, dtype=np.double, ndmin=1)

        return LinearRegressionLeafNode(
                leaf_id, indices, depth, impurity, n_samples, mean, parent, theta0, theta1
                )


