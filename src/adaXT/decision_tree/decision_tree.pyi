from typing import Type, Literal
import numpy as np
from .splitter import Splitter
from ..criteria import Criteria
from .nodes import LeafNode, Node
from .predict import Predict
from .leafbuilder import LeafBuilder
import sys

class DecisionTree:
    """
    Attributes
    ----------
    max_depth: int
        The maximum depth of the tree.
    tree_type: str
        The type of tree, either "Regression" or "Classification".
    leaf_nodes: list[LeafNode]
        A list of all leaf nodes in the tree.
    root: Node
        The root node of the tree.
    n_nodes: int
        The number of nodes in the tree.
    n_features: int
        The number of features in the training data.
    n_classes: int
        The number of classes in the training data. None for "Regression" tree.
    n_rows: int
        The number of rows in the training data.
    classes: np.ndarray
        A list of all class labels. None for "Regression" tree.
    """

    max_depth: int
    tree_type: str
    leaf_nodes: list[LeafNode]
    root: Node
    n_nodes: int
    n_features: int
    n_classes: int
    n_rows: int
    classes: np.ndarray

    def __init__(
        self,
        tree_type: str | None = None,
        max_depth: int = sys.maxsize,
        impurity_tol: float = 0,
        max_features: int | float | Literal["sqrt", "log2"] | None = None,
        skip_check_input: bool = False,
        min_samples_split: int = 1,
        min_samples_leaf: int = 1,
        min_improvement: float = 0,
        criteria: Type[Criteria] | None = None,
        leaf_builder: Type[LeafBuilder] | None = None,
        predict: Type[Predict] | None = None,
        splitter: Type[Splitter] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        tree_type : str
            Classification or Regression
        max_depth : int
            The maximum depth of the tree.
        impurity_tol : float
            The tolerance of impurity in a leaf node.
        max_features: int | float | Literal["sqrt", "log2"] | None
            The number of features to consider when looking for a split,
        skip_check_input : bool
            Whether to skip checking the input for consistency
        min_samples_split : int
            The minimum amount of samples in a split.
        min_samples_leaf : int
            the minimum amount of samples in a leaf node, by default 1
        min_improvement : float
            the minimum improvement gained from performing a split,
            by default 0
        criteria : Criteria
            The Criteria class to use,
            if none defaults to tree_type default
        leaf_builder : LeafBuilder
            LeafBuilder class to use when building a given leaf,
            if none defaults to tree_type default
        predict : Predict
            Predict class to use when predicting,
            if none defaults to tree_type default
        splitter : Splitter | None, optional
            The Splitter class if None uses default Splitter class.
        skip_check_input : bool
            Skips any error checking on the features and response in the fitting function of a tree, should only be used if you know what you are doing, by default false.
        """
        pass

    def fit(
        self,
        X,
        Y,
        sample_indices: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """
        Build the decision tree from the training data (X, y).

        Parameters
        ----------
        X : array-like object
            The feature values used for training. Internally it will be converted to np.ndarray with dtype=np.float64.
        Y : array-like object
            The response values used for training. Internally it will be converted to np.ndarray with dtype=np.float64.
        sample_indices : array-like object | None, optional
            A vector specifying samples of the training data that should be used during training. If None all samples are used.
        sample_weight : np.ndarray | None, optional
            Sample weights. Currently not implemented.
        """
        pass

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
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

    def predict_leaf_matrix(self, X: np.ndarray|None, scale: bool = False) -> np.ndarray:
        """
        Creates NxN matrix,
        where N is the number of observations in X.
        If A_{i,j} = 1 then i and j are in the same leafnode, otherwise 0.
        If they are scaled, then A_{i,j} is instead scaled by the number
        of elements in the leaf node.

        Parameters
        ----------
        X : np.ndarray
            New values to be fitted, if None returns leaf matrix
        scale : bool, optional
            Whether to scale the entries, by default False

        Returns
        -------
        np.ndarray
            NxN matrix
        """
        pass

    def refit_leaf_nodes(self, X:np.ndarray, Y:np.ndarray,
                         sample_weight:np.ndarray, prediction_indices:
                         np.ndarray) -> None:
        """
        Removes all leafnodes created on the initial fit and replaces them by
        predicting all prediction_indices and placing them into new leaf nodes.

        This method can be used to update the leafs node in decision tree based
        on a new data while keeping the original splitting rules. If X does not
        contain the original training data the tree structure might change as
        leaf nodes without samples are collapsed. The method is also used to
        create honest splitting in RandomForests.

        Parameters
        ----------
        X : array-like object
            The feature values used for training. Internally it will be converted to np.ndarray with dtype=np.float64.
        Y : array-like object
            The response values used for training. Internally it will be converted to np.ndarray with dtype=np.float64.
        sample_weight : np.ndarray | None, optional
            Sample weights. Currently not implemented.
        prediction_indices: np.ndarray
            Values to create new leaf nodes with
        """
        pass
