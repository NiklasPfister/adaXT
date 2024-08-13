# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
from numpy import float64 as DOUBLE
from ..decision_tree.nodes import DecisionNode
from collections.abc import Sequence
cimport numpy as cnp


cdef class Predict():

    def __cinit__(self, double[:, ::1] X, double[::1] Y, object root):
        self.X = X
        self.Y = Y
        self.n_features = X.shape[1]
        self.root = root

    def __reduce__(self):
        return (self.__class__, (self.X.base, self.Y.base, self.root))

    cdef double[:, ::1] __check_dimensions(self, object X):
        X = np.ascontiguousarray(X, dtype=DOUBLE)
        # If there is only a single point
        if X.ndim == 1:
            if (X.shape[0] != self.n_features):
                raise ValueError(f"Number of features should be {self.n_features}, got {X.shape[0]}")

            # expand the dimensions
            X = np.expand_dims(X, axis=0)
        else:
            if X.shape[1] != self.n_features:
                raise ValueError(f"Dimension should be {self.n_features}, got {X.shape[1]}")
        return X

    def predict(self, object X, **kwargs):
        raise NotImplementedError("Function predict is not implemented for this Predict class")

    cpdef list predict_proba(self, object X):
        raise NotImplementedError("Function predict_proba is not implemented for this Predict class")

    cpdef cnp.ndarray predict_leaf_matrix(self, object X, bint scale = False):
        cdef:
            int i
            int row
            dict ht
            int cur_split_idx
            double cur_threshold

        # Make sure that x fits the dimensions.
        X = self.__check_dimensions(X)
        row = X.shape[0]

        ht = {}
        for i in range(row):
            cur_node = self.root
            while isinstance(cur_node, DecisionNode):
                cur_split_idx = cur_node.split_idx
                cur_threshold = cur_node.threshold
                if X[i, cur_split_idx] < cur_threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child

            if cur_node.id not in ht.keys():
                ht[cur_node.id] = [i]
            else:
                ht[cur_node.id] += [i]
        matrix = np.zeros((row, row))
        for key in ht.keys():
            indices = ht[key]
            val = 1
            count = len(indices)
            if scale:
                val = 1/count
            matrix[np.ix_(indices, indices)] = val

        return matrix

    @staticmethod
    def forest_predict(predictions: np.ndarray, **kwargs):
        raise NotImplementedError("The forest predict function is not implemented for this Predict Class")

    @staticmethod
    def forest_predict_proba(predictions: np.ndarray, **kwargs):
        raise NotImplementedError("The forest predict function is not implemented for this Predict Class")


cdef class PredictClassification(Predict):
    def __cinit__(self, double[:, ::1] X, double[::1] Y, object root, **kwargs):
        self.classes = np.unique(Y)

    cdef int __find_max_index(self, double[::1] lst):
        cdef:
            int cur_max, i
        cur_max = 0
        for i in range(1, len(lst)):
            if lst[cur_max] < lst[i]:
                cur_max = i
        return cur_max

    def predict(self, object X, **kwargs):
        cdef:
            int i, cur_split_idx, idx, n_obs
            double cur_threshold
            object cur_node
            double[:] Y

        # Make sure that x fits the dimensions.
        X = Predict.__check_dimensions(self, X)
        n_obs = X.shape[0]
        Y = np.empty(n_obs)

        for i in range(n_obs):
            cur_node = self.root
            while isinstance(cur_node, DecisionNode):
                cur_split_idx = cur_node.split_idx
                cur_threshold = cur_node.threshold
                if X[i, cur_split_idx] < cur_threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child

            idx = self.__find_max_index(cur_node.value)
            if self.classes is not None:
                Y[i] = self.classes[idx]
        return Y

    cpdef list predict_proba(self, object X):
        cdef:
            int i, cur_split_idx, n_obs
            double cur_threshold
            object cur_node
            list ret_val

        # Make sure that x fits the dimensions.
        X = Predict.__check_dimensions(self, X)
        n_obs = X.shape[0]
        ret_val = []
        for i in range(n_obs):
            cur_node = self.root
            while isinstance(cur_node, DecisionNode):
                cur_split_idx = cur_node.split_idx
                cur_threshold = cur_node.threshold
                if X[i, cur_split_idx] < cur_threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            if self.classes is not None:
                ret_val.append(cur_node.value)
        return ret_val

    @staticmethod
    def forest_predict(predictions: np.ndarray, **kwargs):
        def __most_frequent_element(arr):
            values, counts = np.unique(arr, return_counts=True)
            return values[np.argmax(counts)]
        return np.apply_along_axis(
            __most_frequent_element, 1, predictions
        )

    @staticmethod
    def forest_predict_proba(predictions: np.ndarray, **kwargs):
        return np.mean(predictions, axis=-1)


cdef class PredictRegression(Predict):
    def predict(self, object X, **kwargs):
        cdef:
            int i, cur_split_idx, n_obs
            double cur_threshold
            object cur_node
            double[:] Y

        # Make sure that x fits the dimensions.
        X = Predict.__check_dimensions(self, X)
        n_obs = X.shape[0]
        Y = np.empty(n_obs)

        for i in range(n_obs):
            cur_node = self.root
            while isinstance(cur_node, DecisionNode):
                cur_split_idx = cur_node.split_idx
                cur_threshold = cur_node.threshold
                if X[i, cur_split_idx] < cur_threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            Y[i] = cur_node.value[0]
        return Y

    @staticmethod
    def forest_predict(predictions: np.ndarray, **kwargs):
        return np.mean(predictions, axis=-1)


cdef class PredictLocalPolynomial(PredictRegression):

    def predict(self, object X, **kwargs):
        cdef:
            int i, cur_split_idx, n_obs, ind, oo
            double cur_threshold
            object cur_node
            double[:, ::1] deriv_mat

        if "order" not in kwargs.keys():
            order = [0, 1, 2]
        else:
            order = np.array(kwargs['order'], ndmin=1, dtype='int')
            if np.max(order) > 2 or np.min(order) < 0 or len(order) > 3:
                raise ValueError('order needs to be convertable to an array of length at most 3 with values in 0, 1 or 2')


        X = Predict.__check_dimensions(self, X)
        n_obs = X.shape[0]
        deriv_mat = np.empty((n_obs, len(order)))

        for i in range(n_obs):
            cur_node = self.root
            while isinstance(cur_node, DecisionNode):
                cur_split_idx = cur_node.split_idx
                cur_threshold = cur_node.threshold
                if X[i, cur_split_idx] < cur_threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            ind = 0
            for oo in order:
                if oo == 0:
                    deriv_mat[i, ind] = cur_node.theta0 + cur_node.theta1*X[i, 0] + cur_node.theta2*X[i, 0]*X[i, 0]
                elif oo == 1:
                    deriv_mat[i, ind] = cur_node.theta1 + 2.0 * cur_node.theta2*X[i, 0]
                elif oo == 2:
                    deriv_mat[i, ind] = 2.0 * cur_node.theta2
                ind += 1
        return deriv_mat


cdef class PredictQuantile(Predict):

    def predict(self, object X, **kwargs):
        cdef:
            int i, cur_split_idx, n_obs
            double cur_threshold
            object cur_node
            bint save_indices
        quantile = kwargs['quantile']
        if "save_indices" in kwargs.keys():
            save_indices = <bint> kwargs['save_indices']
        else:
            save_indices = False
        # Make sure that x fits the dimensions.
        X = Predict.__check_dimensions(self, X)
        n_obs = X.shape[0]
        # Check if quantile is an array
        if isinstance(quantile, Sequence):
            Y = np.empty((n_obs, len(quantile)))
        else:
            Y = np.empty(n_obs)

        for i in range(n_obs):
            cur_node = self.root
            while isinstance(cur_node, DecisionNode):
                cur_split_idx = cur_node.split_idx
                cur_threshold = cur_node.threshold
                if X[i, cur_split_idx] < cur_threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child

            if save_indices:
                Y[i] = self.Y.base[cur_node.indices]
            else:
                Y[i] = np.quantile(self.Y.base[cur_node.indices], quantile)
        return Y

    # TODO: Check whether this does what it should
    @staticmethod
    def forest_predict(predictions: np.ndarray, **kwargs):
        quantile = kwargs['quantile']
        return np.quantile(predictions, quantile, axis=-1)
