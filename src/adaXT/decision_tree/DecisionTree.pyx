# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

# General
import numpy as np
from numpy import float64 as DOUBLE
import sys
from typing import List

# Custom
from .splitter import Splitter
from .criteria import Criteria
from .DepthTreeBuilder import DepthTreeBuilder
from .Nodes import Node, DecisionNode, LeafNode

cdef double EPSILON = np.finfo('double').eps

class DecisionTree:
    """
    DecisionTree object
    """

    def __init__(
            self,
            tree_type: str,
            criteria: Criteria,
            max_depth: int = sys.maxsize,
            impurity_tol: float = 1e-20,
            min_samples: int = 1,
            root: Node | None = None,
            n_nodes: int = -1,
            n_features: int = -1,
            n_classes: int = -1,
            n_obs: int = -1,
            leaf_nodes: List[Node] | None = None,
            pre_sort: None | np.ndarray = None,
            double[:] classes = None) -> None:
        """
        Parameters
        ----------
        tree_type : str
            Classification or Regression
        max_depth : int
            maximum depth of the tree, by default int(np.inf)
        impurity_tol : float
            the tolerance of impurity in a leaf node, by default 1e-20
        min_samples : int
            the minimum amount of samples in a leaf node, by deafult 2
        root : Node | None
            root node, by default None, added after fitting
        n_nodes : int | None
            number of nodes in the tree, by default -1, added after fitting
        n_features : int | None
            number of features in the dataset, by default -1, added after fitting
        n_classes : int | None
            number of classes in the dataset, by default -1, added after fitting
        n_obs : int | None
            number of observations in the dataset, by default -1, added after fitting
        leaf_nodes : List[Node] | None
            number of leaf nodes in the tree, by default None, added after fitting
        pre_sort: np.ndarray | None
            a sorted index matrix for the dataset
        classes : np.ndarray | None
            the different classes in response, by default None, added after fitting
        """
        tree_types = ["Classification", "Regression"]
        assert tree_type in tree_types, f"Expected Classification or Regression as tree type, got: {tree_type}"
        self.max_depth = max_depth
        self.criteria = criteria
        self.impurity_tol = impurity_tol
        self.min_samples = min_samples
        self.tree_type = tree_type
        self.leaf_nodes = leaf_nodes
        self.root = root
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_obs = n_obs
        self.pre_sort = pre_sort
        self.classes = classes

    def check_input(self, X: object, Y: object):
        # Make sure input arrays are c contigous
        X = np.ascontiguousarray(X, dtype=DOUBLE)
        Y = np.ascontiguousarray(Y, dtype=DOUBLE)

        return X, Y

    def fit(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            splitter: Splitter | None = None,
            feature_indices: np.ndarray | None = None,
            sample_indices: np.ndarray | None = None) -> None:
        """
        Function used to fit the data on the tree using the DepthTreeBuilder

        Parameters
        ----------
        X : np.ndarray
            feature values
        Y : np.ndarray
            outcome values
        criteria : FuncWrapper
            Callable criteria function used to calculate impurity wrapped in Funcwrapper class.
        splitter : Splitter | None, optional
            Splitter class if None uses premade Splitter class
        feature_indices : np.ndarray | None, optional
            which features to use from the data X, by default uses all
        sample_indices : np.ndarray | None, optional
            which samples to use from the data X and Y, by default uses all
        """
        # TODO: test feature and sample indexing
        X, Y = self.check_input(X, Y)
        row, col = X.shape
        if sample_indices is None:
            sample_indices = np.arange(row)
        if feature_indices is None:
            feature_indices = np.arange(col)
        builder = DepthTreeBuilder(
            X,
            Y,
            feature_indices,
            sample_indices,
            self.criteria(X, Y),
            splitter,
            self.impurity_tol,
            pre_sort=self.pre_sort)
        builder.build_tree(self)

    def predict(self, double[:, :] X):
        """
        Predicts a y-value for given X values

        Parameters
        ----------
        X : np.ndarray
            (N, M) numpy array with features to predict

        Returns
        -------
        np.ndarray
            (N) numpy array with the prediction
        """
        cdef:
            int i, cur_split_idx, idx
            double cur_threshold
            int row = X.shape[0]
            double[:] Y = np.empty(row)
            object cur_node

        if not self.root:
            return Y
        for i in range(row):
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
                idx = self.find_max_index(cur_node.value)
                if self.classes is not None:
                    Y[i] = self.classes[idx]
        return Y

    def find_max_index(self, lst):
        cur_max = 0
        for i in range(1, len(lst)):
            if lst[cur_max] < lst[i]:
                cur_max = i 
        return cur_max

    def get_leaf_matrix(self) -> np.ndarray:
        """
        Creates NxN matrix,
        where N is the number of observations.
        If a given value is 1, then they are in the same leaf,
        otherwise it is 0

        Returns
        -------
        np.ndarray
            NxN matrix
        """
        leaf_nodes = self.leaf_nodes
        n_obs = self.n_obs

        data = np.zeros((n_obs, n_obs))
        if (not leaf_nodes):  # make sure that there are calculated observations
            return data
        for node in leaf_nodes:
            data[np.ix_(node.indices, node.indices)] = 1

        # TODO scale
        return data

    def predict_leaf_matrix(self, double[:, :] X, scale: bool = False):
        cdef:
            int i
            int row = X.shape[0]
            double[:] Y = np.empty(row)
            dict ht = {}
            int cur_split_idx
            double cur_threshold

        if not self.root:
            return Y
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