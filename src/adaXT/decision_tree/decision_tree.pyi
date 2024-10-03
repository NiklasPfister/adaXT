from typing import Type, Literal
from numpy.typing import ArrayLike
import numpy as np
from .splitter import Splitter
from ..criteria import Criteria
from .nodes import LeafNode, Node
from ..predict import Predict
from ..leaf_builder import LeafBuilder
from ..base_model import BaseModel
import sys

class DecisionTree(BaseModel):
    """
    Attributes
    ----------
    max_depth: int
        The maximum depth of the tree.
    tree_type: str
        The type of tree, either a string specifying a supported type
        (currently "Regression", "Classification", "Quantile" or "Gradient")
        or None.
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
    """

    max_depth: int
    tree_type: str
    leaf_nodes: list[LeafNode]
    root: Node
    n_nodes: int
    n_features: int
    n_rows: int

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
            The type of tree, either a string specifying a supported type
            (currently "Regression", "Classification", "Quantile" or "Gradient")
            or None.
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
        criteria : Type[Criteria] | None
            The Criteria class to use, if None it defaults to the tree_type
            default.
        leaf_builder : Type[LeafBuilder] | None
            The LeafBuilder class to use, if None it defaults to the tree_type
            default.
        predict : Type[Predict] | None
            The Predict class to use, if None it defaults to the tree_type
            default.
        splitter : Type[Splitter] | None
            The Splitter class to use, if None it defaults to the default
            Splitter class.
        skip_check_input : bool
            Skips any error checking on the features and response in the fitting
            function of a tree, should only be used if you know what you are
            doing, by default false.
        """
        pass

    def fit(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        sample_indices: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> None:
        """
        Fit the decision tree with training data (X, Y).

        Parameters
        ----------
        X : array-like object of dimension 2
            The feature values used for training. Internally it will be
            converted to np.ndarray with dtype=np.float64.
        Y : array-like object of 1 or 2 dimensions
            The response values used for training. Internally it will be
            converted to np.ndarray with dtype=np.float64.
        sample_indices : array-like object of dimension 1 | None
            A vector specifying samples of the training data that should be used
            during training. If None all samples are used.
        sample_weight : array-like object of dimension 1 | None
            Sample weights. May not be implemented for every criteria.
        """
        pass

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
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

        Gradient:
        ----------
        Returns a matrix with columns corresponding to different orders of
        derivatives that can be provided via the 'orders' parameter. Default
        behavior is to compute orders 0, 1 and 2.


        Parameters
        ----------
        X : array-like object of dimension 2
            New samples at which to predict the response. Internally it will be
            converted to np.ndarray with dtype=np.float64.

        Returns
        -------
        np.ndarray
            (N, K) numpy array with the prediction, where K depends on the
            Prediction class and is generally 1
        """
        pass

    def predict_weights(
        self, X: ArrayLike | None = None, scale: bool = True
    ) -> np.ndarray:
        """
        Predicts a weight matrix W, where W[i,j] indicates if X[i, :] and
        Xtrain[j, :] are in the same leaf node, where Xtrain denotes the training data.
        If scale is True, then the value is divided by the number of other
        training samples in the same leaf node.

        Parameters
        ----------
        X: array-like object of dimension 2 (shape Mxd)
            New samples to predict a weight (corresponding to columns in the output).
            If None then the training data is used as X.

        scale: bool
            Whether to do row-wise scaling.

        Returns
        -------
        np.ndarray
            A numpy array of shape MxN, where N denotes the number of rows of
            the original training data and M the number of rows of X.
        """
        pass

    def predict_leaf(self, X: ArrayLike | None) -> dict:
        """
        Computes a hash table indexing in which LeafNodes the rows of the provided
        X fall into.

        Parameters
        ----------
        X : array-like object of dimension 2
            2-dimensional array for which the rows are the samples at which to
            predict.

        Returns
        -------
        dict
            A hash table with keys corresponding to LeafNode ids and values corresponding
            to lists of indices of the rows that land in a given LeafNode.
        """
        pass

    def similarity(self, X0: ArrayLike, X1: ArrayLike) -> np.ndarray:
        """
        Computes a similarity matrix W of size NxM, where each element W[i, j]
        is 1 if and only if X0[i, :] and X1[j, :] end up in the same leaf node.

        Parameters
        ----------
        X0: array-like object of dimension 2 (shape Nxd)
            Array corresponding to rows of W in the output.
        X1: array-like object of dimension 2 (shape Mxd)
            Array corresponding to columns of W in the output.

        Returns
        -------
        np.ndarray
            A NxM shaped np.ndarray.
        """
        pass

    def _tree_based_weights(
        self, hash0: dict, hash1: dict, size_X0: int, size_X1: int, scaling: str
    ) -> np.ndarray:
        pass

    def refit_leaf_nodes(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        sample_indices: ArrayLike | None = None,
        **kwargs
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
        X : array-like object of dimension 2
            The feature values used for training. Internally it will be
            converted to np.ndarray with dtype=np.float64.
        Y : array-like object of dimension 1 or 2
            The response values used for training. Internally it will be
            converted to np.ndarray with dtype=np.float64.
        sample_weight : array-like object of dimension 1 | None
            Sample weights. May not be implemented for all criteria.
        sample_indices: array-like object of dimension 1 | None
            Indices of X which to create new leaf nodes with.
        """
        pass
