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
        The type of tree, either  a string specifying a supported type
        (currently "Regression", "Classification" and "Quantile") or None.
    leaf_nodes: list[LeafNode]
        A list of all leaf nodes in the tree.
    root: Node
        The root node of the tree.
    n_nodes: int
        The number of nodes in the tree.
    n_features: int
        The number of features in the training data.
    n_rows: int
        The number of rows (i.e., samples) in the training data.
    n_classes: int
        The number of classes in the training data if
        tree_type=="Classification", otherwise None.
    classes: np.ndarray
        A list of all class labels if tree_type=="Classification", otherwise
        None.
    """

    max_depth: int
    tree_type: str
    leaf_nodes: list[LeafNode]
    root: Node
    n_nodes: int
    n_features: int
    n_rows: int
    n_classes: int
    classes: np.ndarray

    def __init__(
        self,
        tree_type: str | None = None,
        max_depth: int = sys.maxsize,
        impurity_tol: float = 0,
        max_features: int | float | Literal["sqrt", "log2"] | None = None,
        min_samples_split: int = 1,
        min_samples_leaf: int = 1,
        min_improvement: float = 0,
        criteria: Type[Criteria] | None = None,
        leaf_builder: Type[LeafBuilder] | None = None,
        predict: Type[Predict] | None = None,
        splitter: Type[Splitter] | None = None,
        skip_check_input: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        tree_type : str | None
            The type of tree, either  a string specifying a supported type
            (currently "Regression", "Classification" and "Quantile") or None.
        max_depth : int
            The maximum depth of the tree.
        impurity_tol : float
            The tolerance of impurity in a leaf node.
        max_features: int | float | Literal["sqrt", "log2"] | None
            The number of features to consider when looking for a split.
        min_samples_split : int
            The minimum number of samples in a split.
        min_samples_leaf : int
            The minimum number of samples in a leaf node.
        min_improvement : float
            The minimum improvement gained from performing a split.
        criteria : Criteria | None
            The Criteria class to use, if None it defaults to the tree_type
            default.
        leaf_builder : LeafBuilder | None
            The LeafBuilder class to use, if None it defaults to the tree_type
            default.
        predict : Predict | None
            The Predict class to use, if None it defaults to the tree_type
            default.
        splitter : Splitter | None
            The Splitter class to use, if None it defaults to the default
            Splitter class.
        skip_check_input : bool
            Whether to skip checking the input for consistency.

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
        Fit the decision tree with training data (X, Y).

        Parameters
        ----------
        X : array-like object
            The feature values used for training. Internally it will be
            converted to np.ndarray with dtype=np.float64. Rows correspond to
            samples.
        Y : array-like object
            The response values used for training. Internally it will be
            converted to np.ndarray with dtype=np.float64.
        sample_indices : array-like object | None
            A vector specifying samples of the training data that should be
            used during training. If None all samples are used.
        sample_weight : np.ndarray | None
            Sample weights. Currently not implemented.
        """
        pass

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict response values at X using fitted decision tree. The behavior
        of this function is determined by the Prediction class used in the
        decision tree. For currently existing tree types the corresponding
        behavior is as follows:

        Classification:
        ----------
        Returns the class with the highest proportion within the final leaf node.

        Given predict_proba=True, it instead calculates the probability
        distribution.

        Regression:
        ----------
        Returns the mean value of the response within the final leaf node.

        Quantile:
        ----------
        Returns the conditional quantile of the response, where the quantile is
        specified by passing a list of quantiles via the `quantile` parameter.


        Parameters
        ----------
        X : np.ndarray
            (N, M) numpy array with features to predict

        Returns
        -------
        np.ndarray
            (N, K) numpy array with the prediction, where K depends on the
            Prediction class and is generally 1
        """
        pass

    def predict_leaf_matrix(
        self, X: np.ndarray | None, scale: bool = False
    ) -> np.ndarray:
        """
        Creates NxN matrix, where N is the number of observations in X.
        If A_{i,j} = Z then i and j are in the same leafnode, Z number of times.
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

    def refit_leaf_nodes(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray | None,
        sample_indices: np.ndarray | None,
    ) -> None:
        """
        Refits the leaf nodes in a previously fitted decision tree.

        More precisely, the method removes all leafnodes created on the initial
        fit and replaces them by predicting all samples in X that appear in
        sample_indices and placing them into new leaf nodes.

        This method can be used to update the leaf nodes in a decision tree based
        on a new data while keeping the original splitting rules. If X does not
        contain the original training data the tree structure might change as
        leaf nodes without samples are collapsed. The method is also used to
        create honest splitting in RandomForests.

        Parameters
        ----------
        X : array-like object
            The feature values used for training. Internally it will be
            converted to np.ndarray with dtype=np.float64.
        Y : array-like object
            The response values used for training. Internally it will be
            converted to np.ndarray with dtype=np.float64.
        sample_weight : np.ndarray | None
            Sample weights. Currently not implemented.
        sample_indices: np.ndarray
            Values to create new leaf nodes with
        """
        pass
