import numpy as np
from numpy import float64 as DOUBLE
from ..decision_tree.nodes import DecisionNode
from collections.abc import Sequence
from statistics import mode

cimport numpy as cnp
from ..decision_tree.nodes cimport Node, DecisionNode
cimport cython

from ..parallel import ParallelModel

import time

# Circular import. Since only used for typing, this fixes the issue.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..decision_tree import DecisionTree


# Use with cdef code instead of the imported DOUBLE
ctypedef cnp.float64_t DOUBLE_t


def predict_default(
        tree,
        double[:, ::1] X_pred,
        **kwargs) -> np.ndarray:

    res = tree.predict(X_pred, **kwargs)
    return res


def predict_proba(
        tree,
        double[:, ::1] X_pred,
        **kwargs) -> np.ndarray:

    res = tree.predict(X_pred, predict_proba=True, **kwargs)
    return res


def predict_quantile(
    tree,
    X_pred: double[:, ::1],
) -> list:
    cdef:
        int i, cur_split_idx
        double cur_threshold
        int n_obs = X_pred.shape[0]
        Node cur_node
        DecisionNode dNode
    # Check if quantile is an array
    indices = []

    for i in range(n_obs):
        cur_node = <Node> tree.root
        while not cur_node.is_leaf:
            dNode = <DecisionNode> cur_node
            cur_split_idx = dNode.split_idx
            cur_threshold = dNode.threshold
            if X_pred[i, cur_split_idx] <= cur_threshold:
                cur_node = <Node> dNode.left_child
            else:
                cur_node = <Node> dNode.right_child

        indices.append(cur_node.indices)

    return indices


cdef class Predictor():
    def __init__(self, const double[:, ::1] X, const double[:, ::1] Y, object root, **kwargs):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.root = root
        self.n_features = X.shape[1]

    def predict(self, double[:, ::1] X, **kwargs) -> np.ndarray:
        raise NotImplementedError("Function predict is not implemented for this Predictor class")

    cpdef dict predict_leaf(self, double[:, ::1] X):
        cdef:
            int i, cur_split_idx, n_obs
            double cur_threshold
            Node cur_node
            DecisionNode dNode
            dict ht

        # Make sure that x fits the dimensions.
        ht = {}
        n_obs = X.shape[0]

        for i in range(n_obs):
            cur_node = <Node> self.root
            while not cur_node.is_leaf:
                dNode = <DecisionNode> cur_node
                cur_split_idx = dNode.split_idx
                cur_threshold = dNode.threshold
                if X[i, cur_split_idx] <= cur_threshold:
                    cur_node = <Node> dNode.left_child
                else:
                    cur_node = <Node> dNode.right_child

            if cur_node.id not in ht.keys():
                ht[cur_node.id] = [i]
            else:
                ht[cur_node.id] += [i]
        return ht

    def __get_state__(self):
        return {
                "root": self.root,
                "X": np.asarray(self.X),
                "Y": np.asarray(self.Y),
                }

    def __set_state__(self, d: dict):
        self.X = d["X"]
        self.Y = d["Y"]
        self.root = d["root"]

    @staticmethod
    def forest_predict(cnp.ndarray[DOUBLE_t, ndim=2] X_train,
                       cnp.ndarray[DOUBLE_t, ndim=2] Y_train,
                       cnp.ndarray[DOUBLE_t, ndim=2] X_pred,
                       trees: list[DecisionTree],
                       parallel: ParallelModel,
                       **kwargs) -> np.ndarray:
        sequential = True
        if "sequential" in kwargs:
            sequential = kwargs["sequential"]
            kwargs.pop("sequential")
        predictions = parallel.async_map(predict_default,
                                         trees,
                                         X_pred=X_pred,
                                         sequential=sequential,
                                         **kwargs)
        return np.mean(predictions, axis=0, dtype=DOUBLE)


@cython.final
cdef class PredictorClassification(Predictor):
    def __init__(self,
                 const double[:, ::1] X,
                 const double[:, ::1] Y,
                 object root, **kwargs) -> None:
        super().__init__(X, Y, root, **kwargs)
        self.classes = np.unique(Y)

    cdef int __find_max_index(self, float[::1] lst) noexcept nogil:
        cdef:
            int cur_max, i
        cur_max = 0
        for i in range(1, lst.shape[0]):
            if lst[cur_max] < lst[i]:
                cur_max = i
        return cur_max

    cdef inline double[::1] __predict(self, double[:, ::1] X) noexcept:
        cdef:
            int i, cur_split_idx, n_obs
            double cur_threshold
            Node cur_node
            DecisionNode dNode
            double[::1] prediction

        # Make sure that x fits the dimensions.
        n_obs = X.shape[0]
        prediction = np.empty(n_obs, dtype=DOUBLE)

        for i in range(n_obs):
            cur_node = <Node> self.root
            while not cur_node.is_leaf:
                dNode = <DecisionNode> cur_node
                cur_split_idx = dNode.split_idx
                cur_threshold = dNode.threshold
                if X[i, cur_split_idx] <= cur_threshold:
                    cur_node = <Node> dNode.left_child
                else:
                    cur_node = <Node> dNode.right_child
            idx = self.__find_max_index(cur_node.value)
            prediction[i] = self.classes[idx]
        return prediction

    cdef inline cnp.ndarray __predict_proba(self, double[:, ::1] X):
        cdef:
            int i, cur_split_idx, n_obs
            double cur_threshold
            Node cur_node
            DecisionNode dNode
            list ret_val

        # Make sure that x fits the dimensions.
        n_obs = X.shape[0]
        ret_val = []

        for i in range(n_obs):
            cur_node = <Node> self.root
            while not cur_node.is_leaf:
                dNode = <DecisionNode> cur_node
                cur_split_idx = dNode.split_idx
                cur_threshold = dNode.threshold
                if X[i, cur_split_idx] <= cur_threshold:
                    cur_node = <Node> dNode.left_child
                else:
                    cur_node = <Node> dNode.right_child

            ret_val.append(cur_node.value)
        return np.array(ret_val)

    def predict(self, double[:, ::1] X, **kwargs) -> np.ndarray:
        if "predict_proba" in kwargs:
            if kwargs["predict_proba"]:
                return self.__predict_proba(X)

        # if predict_proba = False this return is hit
        return np.asarray(self.__predict(X))

    @staticmethod
    def forest_predict(cnp.ndarray[DOUBLE_t, ndim=2] X_train,
                       cnp.ndarray[DOUBLE_t, ndim=2] Y_train,
                       cnp.ndarray[DOUBLE_t, ndim=2] X_pred,
                       trees: list[DecisionTree],
                       parallel: ParallelModel,
                       **kwargs) -> np.ndarray:
        sequential = True
        if "sequential" in kwargs:
            sequential = kwargs["sequential"]
            kwargs.pop("sequential")

        # Forest_predict_proba
        if "predict_proba" in kwargs:
            if kwargs["predict_proba"]:
                predictions = parallel.async_map(predict_proba,
                                                 map_input=trees,
                                                 X_pred=X_pred,
                                                 sequential=sequential,
                                                 **kwargs)
                return np.mean(predictions, axis=0, dtype=DOUBLE)

        st = time.time()
        predictions = parallel.async_map(predict_default,
                                         trees,
                                         X_pred=X_pred,
                                         sequential=sequential,
                                         **kwargs)
        et = time.time()
        print("Parallel time predict: ", et - st)
        return np.array(np.apply_along_axis(mode, 0, predictions), dtype=int)


@cython.final
cdef class PredictorRegression(Predictor):
    def predict(self, double[:, ::1] X, **kwargs) -> np.ndarray:
        cdef:
            int i, cur_split_idx, n_obs, n_col
            double cur_threshold
            Node cur_node
            DecisionNode dNode
            cnp.ndarray prediction

        # Make sure that x fits the dimensions.
        n_obs = X.shape[0]
        n_col = self.Y.shape[1]
        if n_col > 1:
            prediction = np.empty((n_obs, n_col), dtype=DOUBLE)
        else:
            prediction = np.empty(n_obs, dtype=DOUBLE)

        for i in range(n_obs):
            cur_node = <Node> self.root
            while not cur_node.is_leaf:
                dNode = <DecisionNode> cur_node
                cur_split_idx = dNode.split_idx
                cur_threshold = dNode.threshold
                if X[i, cur_split_idx] <= cur_threshold:
                    cur_node = <Node> dNode.left_child
                else:
                    cur_node = <Node> dNode.right_child

            if cur_node.value.ndim == 1:
                prediction[i] = cur_node.value[0]
            else:
                prediction[i] = cur_node.value
        return prediction


cdef class PredictorLocalPolynomial(Predictor):
    def predict(self, double[:, ::1] X, **kwargs) -> np.ndarray:
        cdef:
            int i, cur_split_idx, n_obs, ind, oo
            double cur_threshold
            Node cur_node
            DecisionNode dNode
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
            cur_node = <Node> self.root
            while not cur_node.is_leaf:
                dNode = <DecisionNode> cur_node
                cur_split_idx = dNode.split_idx
                cur_threshold = dNode.threshold
                if X[i, cur_split_idx] <= cur_threshold:
                    cur_node = <Node> dNode.left_child
                else:
                    cur_node = <Node> dNode.right_child

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


cdef class PredictorQuantile(Predictor):
    def predict(self, double[:, ::1] X, **kwargs) -> np.ndarray:
        cdef:
            int i, cur_split_idx, n_obs
            double cur_threshold
            Node cur_node
            DecisionNode dNode
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
            cur_node = <Node> self.root
            while not cur_node.is_leaf:
                dNode = <DecisionNode> cur_node
                cur_split_idx = dNode.split_idx
                cur_threshold = dNode.threshold
                if X[i, cur_split_idx] <= cur_threshold:
                    cur_node = <Node> dNode.left_child
                else:
                    cur_node = <Node> dNode.right_child

            prediction[i] = np.quantile(self.Y[cur_node.indices, 0], quantile)
        return prediction

    @staticmethod
    def forest_predict(cnp.ndarray[DOUBLE_t, ndim=2] X_train,
                       cnp.ndarray[DOUBLE_t, ndim=2] Y_train,
                       cnp.ndarray[DOUBLE_t, ndim=2] X_pred,
                       trees: list[DecisionTree],
                       parallel: ParallelModel,
                       **kwargs) -> np.ndarray:
        cdef:
            int i, j, n_obs, n_trees
            list pred_indices_combined, indices_combined, prediction_indices
        if "quantile" not in kwargs.keys():
            raise ValueError(
                "quantile called without quantile passed as argument"
            )
        sequential = True
        if "sequential" in kwargs:
            sequential = kwargs["sequential"]
            kwargs.pop("sequential")

        quantile = kwargs['quantile']
        n_obs = X_pred.shape[0]
        prediction_indices = parallel.async_map(predict_quantile,
                                                map_input=trees,
                                                sequential=sequential,
                                                X_pred=X_pred)
        # In case the leaf nodes have multiple elements and not just one, we
        # have to combine them together
        n_trees = len(prediction_indices)
        pred_indices_combined = []
        for i in range(n_obs):
            indices_combined = []
            for j in range(n_trees):
                indices_combined.extend(prediction_indices[j][i])
            pred_indices_combined.append(indices_combined)
        ret = np.array([np.quantile(Y_train[indices, 0], quantile) for
                        indices in pred_indices_combined])
        return ret
