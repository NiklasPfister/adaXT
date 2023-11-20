import numpy as np
from .splitter import Splitter

class Node:
    def __init__(
            self,
            indices: np.ndarray,
            depth: int,
            impurity: float,
            n_samples: int) -> None:
        """
        Parameters
        ----------
        indices : np.ndarray
            indices in node
        depth : int
            depth of noe
        impurity : float
            impurity of node
        n_samples : int
            number of samples in node
        """
        pass

class DecisionNode(Node):
    def __init__(
            self,
            indices: np.ndarray,
            depth: int,
            impurity: float,
            n_samples: int,
            threshold: float,
            split_idx: int,
            left_child: "DecisionNode|LeafNode|None" = None,
            right_child: "DecisionNode|LeafNode|None" = None,
            parent: "DecisionNode|None" = None) -> None:
        """
        Parameters
        ----------
        indices : np.ndarray
            indices ni node
        depth : int
            depth of node
        impurity : float
            impurity in node
        n_samples : int
            number of samples in node
        threshold : float
            the threshold value of a split
        split_idx : int
            the feature index to split on
        left_child : DecisionNode|LeafNode|None, optional
            the left child, by default None
        right_child : DecisionNode|LeafNode|None, optional
            the right child, by default None
        parent : DecisionNode|None, optional
            the parent node, by default None
        """
        pass

class LeafNode(Node):
    def __init__(
            self,
            id: int,
            indices: np.ndarray,
            depth: int,
            impurity: float,
            n_samples: int,
            value: list[float],
            parent: DecisionNode) -> None:
        """
        Parameters
        ----------
        indices : np.ndarray
            Indices of leaf node
        depth : int
            depth the leaf node is located at
        impurity : float
            Impurity of leaf node
        n_samples : int
            Number of samples in leaf node
        value : list[float]
            The mean values of classes in leaf node
        parent : DecisionNode
            The parent node
        """
        pass

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
            leaf_nodes: list[Node] | None = None,
            pre_sort: None | np.ndarray = None,
            classes: np.ndarray | None = None) -> None:
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
        leaf_nodes : list[Node] | None
            number of leaf nodes in the tree, by default None, added after fitting
        pre_sort: np.ndarray | None
            a sorted index matrix for the dataset
        classes : np.ndarray | None
            the different classes in response, by default None, added after fitting
        """
        pass

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
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
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
        pass

    def weight_matrix(self) -> np.ndarray:
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
        pass

    def predict_matrix(self, X: np.ndarray, scale: bool = False):

        pass