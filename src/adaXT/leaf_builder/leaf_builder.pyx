# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from ..decision_tree.nodes import (LeafNode, LocalLinearLeafNode,
                                   LocalQuadraticLeafNode)
import numpy as np
cimport numpy as cnp

cdef class LeafBuilder:
    def __cinit__(self, double[:, ::1] x, double[::1] y, int[::1] all_idx, **kwargs):
        self.x = x
        self.y = y

    cpdef object build_leaf(self,
                            int leaf_id,
                            int[::1] indices,
                            int depth,
                            double impurity,
                            double weighted_samples,
                            object parent):
        raise NotImplementedError("Build leaf not implemented for this LeafBuilder")


cdef class LeafBuilderClassification(LeafBuilder):
    def __cinit__(self, double[:, ::1] x, double[::1] y, int[::1] all_idx, **kwargs):
        self.classes = np.array(np.unique(y.base[all_idx]), dtype=np.double)
        self.n_classes = self.classes.shape[0]

    cdef double[::1] _get_mean(self, int[::1] indices):
        cdef:
            cnp.ndarray[double, ndim=1] ret
            int i, idx, n_samples

        n_samples = indices.shape[0]
        ret = np.zeros(self.n_classes)
        for idx in range(n_samples):
            for i in range(self.n_classes):
                if self.y[indices[idx]] == self.classes[i]:
                    ret[i] += 1  # add 1, if the value is the same as class value
                    break

        ret = ret / n_samples
        return ret

    cpdef object build_leaf(self,
                            int leaf_id,
                            int[::1] indices,
                            int depth,
                            double impurity,
                            double weighted_samples,
                            object parent):
        cdef double[::1] mean = self._get_mean(indices)
        return LeafNode(leaf_id, indices, depth, impurity, weighted_samples,
                        mean, parent)


cdef class LeafBuilderRegression(LeafBuilder):

    cdef double _get_mean(self, int[::1] indices):
        cdef:
            int i
            double sum = 0.0
            int count = len(indices)

        for i in indices:
            sum += self.y[i]

        return sum / count

    cpdef object build_leaf(self,
                            int leaf_id,
                            int[::1] indices,
                            int depth,
                            double impurity,
                            double weighted_samples,
                            object parent):

        cdef double[::1] mean = np.array(self._get_mean(indices), dtype=np.double, ndmin=1)
        return LeafNode(leaf_id, indices, depth, impurity, weighted_samples,
                        mean, parent)

cdef class LeafBuilderLocalLinear(LeafBuilderRegression):

    # Custom mean function, such that we don't have to loop through twice.
    cdef (double, double) _custom_mean(self, int[::1] indices):
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

    cdef (double, double, double) _theta(self, int[::1] indices):
        """
        Estimates regression parameters for a linear regression of the response
        on the first coordinate, i.e., Y is approximated by theta0 + theta1 *
        X[:, 0].
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            The indices to calculate

        Returns
        -------
        (double, double, double)
            where first element is theta0, second is theta1 and third is the
            mean of Y
        """
        cdef:
            double muX, muY, theta0, theta1
            int length, i
            double numerator, denominator
            double X_diff

        length = indices.shape[0]
        denominator = 0.0
        numerator = 0.0
        muX, muY = self._custom_mean(indices)
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

    cpdef object build_leaf(self,
                            int leaf_id,
                            int[::1] indices,
                            int depth,
                            double impurity,
                            double weighted_samples,
                            object parent):
        cdef:
            double[::1] mean
            double theta0, theta1

        theta0, theta1, muY = self._theta(indices)
        mean = np.array(muY, dtype=np.double, ndmin=1)

        return LocalLinearLeafNode(leaf_id, indices, depth, impurity,
                                   weighted_samples, mean, parent, theta0,
                                   theta1)


cdef class LeafBuilderLocalQuadratic(LeafBuilderRegression):

    cdef (double, double, double) _custom_mean(self, int[:] indices):
        cdef:
            double sumXsq, sumX, sumY
            int i
            int length = indices.shape[0]
        sumX = 0.0
        sumXsq = 0.0
        sumY = 0.0
        for i in range(length):
            sumX += self.x[indices[i], 0]
            sumXsq += self.x[indices[i], 0] ** 2
            sumY += self.y[indices[i]]

        return ((sumX / (<double> length)), (sumXsq / (<double> length)), (sumY/ (<double> length)))

    cdef (double, double, double, double) _theta(self, int[::1] indices):
        """
        Estimates regression parameters for a linear regression of the response
        on the first coordinate, i.e., Y is approximated by theta0 + theta1 *
        X[:, 0] + theta2 * X[:, 0] ** 2.

        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            The indices to calculate

        Returns
        -------
        (double, double, double, double)
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
        muX, muXsq, muY = self._custom_mean(indices)
        for i in range(length):
            X_diff = self.x[indices[i], 0] - muX
            Xsq_diff = self.x[indices[i], 0] ** 2 - muXsq
            Y_diff = self.y[indices[i]] - muY
            covXXsq += X_diff * Xsq_diff
            varX += X_diff ** 2
            varXsq += Xsq_diff ** 2
            covXY += X_diff * Y_diff
            covXsqY += Xsq_diff * Y_diff
        det = varX * varXsq - covXXsq ** 2
        if det == 0.0:
            theta1 = 0.0
            theta2 = 0.0
        else:
            theta1 = (varXsq*covXY - covXXsq * covXsqY) / det
            theta2 = (varX*covXsqY - covXXsq * covXY) / det
        theta0 = muY - theta1*muX - theta2*muXsq
        return (theta0, theta1, theta2, muY)

    cpdef object build_leaf(self,
                            int leaf_id,
                            int[::1] indices,
                            int depth,
                            double impurity,
                            double weighted_samples,
                            object parent):
        cdef:
            double[::1] mean
            double theta0, theta1, theta2, muY

        theta0, theta1, theta2, muY = self._theta(indices)
        mean = np.array(muY, dtype=np.double, ndmin=1)

        return LocalQuadraticLeafNode(leaf_id, indices, depth, impurity,
                                      weighted_samples, mean, parent, theta0,
                                      theta1, theta2)
