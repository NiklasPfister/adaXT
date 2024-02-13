import numpy as np
from numpy import float64 as DOUBLE
from .nodes import DecisionNode, LeafNode

cdef class Predict():

    def __cinit__(self, double[:, ::1] X, double[::1] Y, str tree_type, object root):
        print(type(root))
        if not (isinstance(root, DecisionNode)):
            raise ValueError("Prediction expected a DecisionNode as root")
        self.X = X
        self.Y = Y
        self.n_features = X.shape[1]
        self.tree_type = tree_type
        if tree_type == "Classification":
            self.classes = np.unique(Y)
        self.root = root

    cdef double[:, ::1] __check_dimensions(self, double[:, ::1] X):
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

    cdef int __find_max_index(self, list lst):
        cdef:
            int cur_max, i
        cur_max = 0
        for i in range(1, len(lst)):
            if lst[cur_max] < lst[i]:
                cur_max = i
        return cur_max

    cpdef double[:] predict(self, double[:, ::1] X):
        cdef:
            int i, cur_split_idx, idx, n_obs
            double cur_threshold
            object cur_node
            double[:] Y
        if not self.root:
            raise AttributeError("The tree has not been fitted before trying to call predict")

        # Make sure that x fits the dimensions.
        X = self.__check_dimensions(X)
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
            if self.tree_type == "Regression":
                Y[i] = cur_node.value[0]
            elif self.tree_type == "Classification":
                idx = self.__find_max_index(cur_node.value)
                if self.classes is not None:
                    Y[i] = self.classes[idx]
        return Y

    cpdef list predict_proba(self, double[:, ::1] X):
        cdef:
            int i, cur_split_idx, n_obs
            double cur_threshold
            object cur_node
            list ret_val

        if not self.root:
            raise AttributeError("The tree has not been fitted before trying to call predict_proba")

        if self.tree_type != "Classification":
            raise ValueError("predict_proba can only be called on a Classification tree")

        # Make sure that x fits the dimensions.
        X = self.__check_dimensions(X)
        n_obs = X.shape[0]
        n_classes = self.classes.shape[0]
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


    cpdef double[:, ::1] predict_leaf_matrix(self, double[:, ::1] X, bint scale = False):
        cdef:
            int i
            int row
            dict ht
            int cur_split_idx
            double cur_threshold

        if not self.root:
            raise ValueError("The tree has not been trained before trying to predict")

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
