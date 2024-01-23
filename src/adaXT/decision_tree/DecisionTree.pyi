import numpy as np
from .splitter import Splitter
from ..criteria import Criteria
from .Nodes import *
import sys


class DecisionTree:
    """
    Attributes
    ----------
    max_depth: int
        maximum depth of the tree
    tree_type: str
        Which type of tree it is
    leaf_nodes: list[LeafNode]
        a list of all the leaf nodes in the tree
    root: Node
        the root Node of the tree
    n_nodes: int
        the number of nodes in the tree
    n_features: int
        the number of features of the training data
    n_classes: int
        the number of classes of the training data, 0 for Regression Tree
    n_obs: int
        the number of observations used for training
    classes: np.ndarray
        the different classes in the tree given a Classification tree
    """

    def __init__(
            self,
            tree_type: str,
            criteria: Criteria,
            max_depth: int = sys.maxsize,
            impurity_tol: float = 0,
            min_samples_split: int = 1,
            min_samples_leaf: int = 1,
            min_improvement: float = 0,
            splitter: Splitter | None = None) -> None:
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
        min_samples_leaf : int
            the minimum amount of samples in a leaf node, by default 1
        min_improvement: float
            the minimum improvement gained from performing a split, by default 0
        splitter : Splitter | None, optional
            Splitter class if None uses premade Splitter class
        """
        pass

    def fit(
            self,
            X,
            Y,
            sample_indices: np.ndarray | None = None,
            feature_indices: np.ndarray | None = None,
            sample_weight: np.ndarray | None = None,) -> None:
        """
        Function used to fit the data on the tree using the DepthTreeBuilder

        Parameters
        ----------
        X : array-like object
            feature values, will internally be converted to np.ndarray with dtype=np.float64
        Y : array-like object
            response values, will internally be converted to np.ndarray with dtype=np.float64
        sample_indices : array-like object
            specific indices of the dataset you wish to use
        feature_indices : np.ndarray | None, optional
            which features to use from the data X, by default uses all
        sample_weight : np.ndarray | None, optional
            np.ndarray of shape (n_samples,) currently only supports weights in {0, 1}
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
        np.ndarray
            Returns an np.ndarray with the the probabilities for each class per observation in X, the order of the classes corresponds to that in the attribute classes.
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
