# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
from numpy import float64 as DOUBLE
from ..decision_tree.nodes import DecisionNode
from collections.abc import Sequence
from statistics import mode
cimport numpy as cnp
from adaXT.parallel import shared_numpy_array

# Use with cdef code instead of the imported DOUBLE
ctypedef cnp.float64_t DOUBLE_t


def predict_default(tree, X, **kwargs):
    return np.array(tree.predict(X, **kwargs))


def predict_proba(tree, Y, X, unique_classes):
    cdef:
        int i, cur_split_idx
        double cur_threshold
        object cur_node
        list ret_val

    # Make sure that x fits the dimensions.
    n_obs = X.shape[0]
    ret_val = []
    for i in range(n_obs):
        cur_node = tree.root
        while isinstance(cur_node, DecisionNode):
            cur_split_idx = cur_node.split_idx
            cur_threshold = cur_node.threshold
            if X[i, cur_split_idx] < cur_threshold:
                cur_node = cur_node.left_child
            else:
                cur_node = cur_node.right_child
        cur_array = np.zeros(unique_classes)
        n_samples = len(cur_node.indices)
        for idx in cur_node.indices:
            cur_array[np.where(unique_classes == Y[idx, 0])] += 1

        ret_val.append(cur_array/n_samples)

    return np.array(ret_val)


def predict_quantile(tree, X, n_obs):
    # Check if quantile is an array
    indices = []

    for i in range(n_obs):
        cur_node = tree.root
        while isinstance(cur_node, DecisionNode):
            cur_split_idx = cur_node.split_idx
            cur_threshold = cur_node.threshold
            if X[i, cur_split_idx] < cur_threshold:
                cur_node = cur_node.left_child
            else:
                cur_node = cur_node.right_child

        indices.append(cur_node.indices)
    return indices


cdef class Predict():

    def __cinit__(self, double[:, ::1] X, double[:, ::1] Y, object root):
        self.X = X
        self.Y = Y
        self.n_features = X.shape[1]
        self.root = root

    def __reduce__(self):
        return (self.__class__, (self.X.base, self.Y.base, self.root))

    # TODO: predict_indices

    def predict(self, object X, **kwargs) -> np.ndarray:
        raise NotImplementedError("Function predict is not implemented for this Predict class")

    cpdef dict predict_leaf(self, object X):
        cdef:
            int i
            int row
            dict ht
            int cur_split_idx
            double cur_threshold

        # Make sure that x fits the dimensions.
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
        return ht

    @staticmethod
    def forest_predict(X_old, Y_old, X_new, trees, parallel, **kwargs):
        predictions = parallel.async_map(predict_default,
                                         trees,
                                         X=X_new,
                                         **kwargs)
        return np.mean(predictions, axis=0, dtype=DOUBLE)


cdef class PredictClassification(Predict):
    def __cinit__(self, double[:, ::1] X, double[:, ::1] Y, object root, **kwargs):
        self.classes = np.unique(Y)

    cdef int __find_max_index(self, double[::1] lst):
        cdef:
            int cur_max, i
        cur_max = 0
        for i in range(1, len(lst)):
            if lst[cur_max] < lst[i]:
                cur_max = i
        return cur_max

    cdef cnp.ndarray __predict(self, object X):
        cdef:
            int i, cur_split_idx, idx, n_obs
            double cur_threshold
            object cur_node
            cnp.ndarray prediction

        # Make sure that x fits the dimensions.
        n_obs = X.shape[0]
        prediction = np.empty(n_obs, dtype=DOUBLE)

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
                prediction[i] = self.classes[idx]
        return prediction

    cdef cnp.ndarray __predict_proba(self, object X):
        cdef:
            int i, cur_split_idx
            double cur_threshold
            object cur_node
            list ret_val

        # Make sure that x fits the dimensions.
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
        return np.array(ret_val)

    def predict(self, object X, **kwargs):
        if "predict_proba" in kwargs:
            if kwargs["predict_proba"]:
                return self.__predict_proba(X)

        # if predict_proba = False this return is hit
        return self.__predict(X)

    @staticmethod
    def forest_predict(X_old, Y_old, X_new, trees, parallel, **kwargs):
        # Forest_predict_proba
        if "predict_proba" in kwargs:
            if kwargs["predict_proba"]:
                unique_classes = shared_numpy_array(np.unique(Y_old))
                Y_old = shared_numpy_array(Y_old)
                predictions = parallel.async_map(predict_proba, trees, Y=Y_old,
                                                 unique_classes=unique_classes)
                return np.mean(predictions, axis=0, dtype=int)

        predictions = parallel.async_map(predict_default, trees, X=X_new,
                                         **kwargs)
        return np.array(np.apply_along_axis(mode, 0, predictions), dtype=DOUBLE)


cdef class PredictRegression(Predict):
    def predict(self, object X, **kwargs):
        cdef:
            int i, cur_split_idx, n_obs, n_col
            double cur_threshold
            object cur_node
            cnp.ndarray prediction

        # Make sure that x fits the dimensions.
        n_obs = X.shape[0]
        n_col = self.Y.shape[1]
        if n_col > 1:
            prediction = np.empty((n_obs, n_col), dtype=DOUBLE)
        else:
            prediction = np.empty(n_obs, dtype=DOUBLE)

        for i in range(n_obs):
            cur_node = self.root
            while isinstance(cur_node, DecisionNode):
                cur_split_idx = cur_node.split_idx
                cur_threshold = cur_node.threshold
                if X[i, cur_split_idx] < cur_threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            if cur_node.value.ndim == 1:
                prediction[i] = cur_node.value[0]
            else:
                prediction[i] = cur_node.value
        return prediction


cdef class PredictLocalPolynomial(PredictRegression):

    def predict(self, object X, **kwargs):
        cdef:
            int i, cur_split_idx, n_obs, ind, oo
            double cur_threshold
            object cur_node
            cnp.ndarray[DOUBLE_t, ndim=2] deriv_mat

        if "order" not in kwargs.keys():
            order = [0, 1, 2]
        else:
            order = np.array(kwargs['order'], ndmin=1, dtype=int)
            if np.max(order) > 2 or np.min(order) < 0 or len(order) > 3:
                raise ValueError('order needs to be convertable to an array of length at most 3 with values in 0, 1 or 2')

        n_obs = X.shape[0]
        deriv_mat = np.empty((n_obs, len(order)), dtype=DOUBLE)

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
            cnp.ndarray prediction
        if "quantile" not in kwargs.keys():
            raise ValueError(
                        "quantile called without quantile passed as argument"
                    )
        quantile = kwargs['quantile']
        # Make sure that x fits the dimensions.
        n_obs = X.shape[0]
        # Check if quantile is an array
        if isinstance(quantile, Sequence):
            prediction = np.empty((n_obs, len(quantile)), dtype=DOUBLE)
        else:
            prediction = np.empty(n_obs, dtype=DOUBLE)

        for i in range(n_obs):
            cur_node = self.root
            while isinstance(cur_node, DecisionNode):
                cur_split_idx = cur_node.split_idx
                cur_threshold = cur_node.threshold
                if X[i, cur_split_idx] < cur_threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child

            prediction[i] = np.quantile(self.Y.base[cur_node.indices, 0], quantile)
        return prediction

    @staticmethod
    def forest_predict(X_old, Y_old, X_new, trees, parallel, **kwargs):
        cdef:
            int i, j, n_obs, n_trees
            list prediction_indices, pred_indices_combined, indices_combined
        if "quantile" not in kwargs.keys():
            raise ValueError(
                "quantile called without quantile passed as argument"
            )
        quantile = kwargs['quantile']
        n_obs = X_new.shape[0]
        prediction_indices = parallel.async_map(predict_quantile,
                                                map_input=trees, X=X_new,
                                                n_obs=n_obs)
        # In case the leaf nodes have multiple elements and not just one, we
        # have to combine them together
        n_trees = len(prediction_indices)
        pred_indices_combined = []
        for i in range(n_obs):
            indices_combined = []
            for j in range(n_trees):
                indices_combined.extend(prediction_indices[j][i])
            pred_indices_combined.append(indices_combined)
        ret = [np.quantile(Y_old[pred_indices_combined[i]], quantile) for i in
               range(n_obs)]
        return np.array(ret, dtype=DOUBLE)
