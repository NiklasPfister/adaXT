import numpy as np
from .splitter import Splitter
from .criteria import Criteria
from .Nodes import *
import sys

class DecisionTree:
    """
    DecisionTree object
    """
    max_depth: int
    impurity_tol: float
    min_samples_split: int
    min_samples_leaf: int
    min_improvement: float
    criteria: Criteria
    tree_type: str
    leaf_nodes: list[LeafNode]
    root: Node
    n_nodes: int
    n_features: int
    n_classes: int
    n_obs: int
    classes: np.ndarray

    def __init__(
            self,
            tree_type: str,
            criteria: Criteria,
            max_depth: int = sys.maxsize,
            impurity_tol: float = 0,
            min_samples_split: int = 1,
            min_samples_leaf: int = 1,
            min_improvement: float = 0) -> None:
        """
        Parameters
        ----------
        tree_type : str
            Classification or Regression
        criteria: Criteria
            The Criteria class to use, should be of the type Criteria implemented by AdaXT
        max_depth : int
            maximum depth of the tree, by default maximum system size
        impurity_tol : float
            the tolerance of impurity in a leaf node, by default 0
        min_samples_split : int
            the minimum amount of samples in a split, by default 1
        min_samples_split : int
            the minimum amount of samples in a leaf node, by default 1
        min_improvement: float
            the minimum improvement gained from performing a split, by default 0
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
            response values
        splitter : Splitter | None, optional
            splitter class, if None uses premade Splitter class
        feature_indices : np.ndarray | None, optional
            which features to use from the data X, by default uses all
        sample_indices : np.ndarray | None, optional
            which samples to use from the data X and Y, by default uses all
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Works in two ways depending on if the tree is a Classification or Regression tree.

        Classification:
        ----------
        Returns the class with the highest proportion within the final leaf node

        Regression:
        ----------
        Returns the mean value of the outcomes within the final leaf node.

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

    def predict_proba(self, X: np.ndarray):
        """
        Predicts a probability for each response for given X values. Only useable by the Classification Tree.

        Parameters
        ----------
        X : np.ndarray
            (N, M) numpy array with features to predict

        Returns
        -------
        Tuple(np.ndarray, np.ndarray)
            returns a tuple where the first element are the reponsense, and the othe element are the probability for each class per observation in X.
        """
        pass

    def get_leaf_matrix(self, scale: bool = False) -> np.ndarray:
        """
        Creates NxN matrix,
        where N is the number of observations.
        If A_{i,j} = 1 then i and j are in the same LeafNode, otherwise 0.
        If they are scaled, then A_{i,j} is instead scaled by the number
        of elements in the leaf node.


        Parameters
        ----------
        scale : bool, optional
            whether to scale the entries, by default False

        Returns
        -------
        np.ndarray
            NxN matrix
        """
        pass

    def predict_leaf_matrix(self, X: np.ndarray, scale: bool = False) -> np.ndarray:
        """
        Creates NxN matrix,
        where N is the number of observations in X.
        If A_{i,j} = 1 then i and j are in the same leafnode, otherwise 0.
        If they are scaled, then A_{i,j} is instead scaled by the number
        of elements in the leaf node.

        Parameters
        ----------
        X : np.ndarray
            new values to be fitted
        scale : bool, optional
            whether to scale the entries, by default False

        Returns
        -------
        np.ndarray
            NxN matrix
        """
        pass