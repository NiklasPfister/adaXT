from typing import Type, Literal
from numpy.typing import ArrayLike
import numpy as np
from .splitter import Splitter
from ..criteria import Criteria
from .nodes import LeafNode, Node
from ..predictor import Predictor
from ..leaf_builder import LeafBuilder
from ..base_model import BaseModel
from ._decision_tree import _DecisionTree, DepthTreeBuilder
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
    tree_type: str | None
    leaf_nodes: list[LeafNode]
    root: Node | None
    n_nodes: int
    n_features: int
    n_rows: int
    _tree: _DecisionTree

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
        predictor: Type[Predictor] | None = None,
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
        predictor : Type[Predictor] | None
            The Predictor class to use, if None it defaults to the tree_type
            default.
        splitter : Type[Splitter] | None
            The Splitter class to use, if None it defaults to the default
            Splitter class.
        skip_check_input : bool
            Skips any error checking on the features and response in the fitting
            function of a tree, should only be used if you know what you are
            doing, by default false.
        """

        self.skip_check_input = skip_check_input

        # Input only checked on fitting.
        self.criteria = criteria
        self.predictor = predictor
        self.leaf_builder = leaf_builder
        self.splitter = splitter
        self.max_features = max_features
        self.tree_type = tree_type

        self.skip_check_input = skip_check_input
        self.max_depth = max_depth
        self.impurity_tol = impurity_tol
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_improvement = min_improvement
        self.tree_type = tree_type

    # In python this function is called if the attribute does not exist on the
    # actual instance. Thus we check the wrapped tree instance.
    def __getattr__(self, name):
        if name == "_tree":
            # This is called, if _tree is not already defined.
            return None
        else:
            return getattr(self._tree, name)

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
        # Check inputs
        if not self.skip_check_input:
            X, Y = self._check_input(X, Y)
            self._check_tree_type(
                self.tree_type,
                self.criteria,
                self.splitter,
                self.leaf_builder,
                self.predictor,
            )
            self.max_features = self._check_max_features(
                self.max_features, X.shape[0])

        self._tree = _DecisionTree(
            max_depth=self.max_depth,
            impurity_tol=self.impurity_tol,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_improvement=self.min_improvement,
            max_features=self.max_features,
            criteria=self.criteria,
            leaf_builder=self.leaf_builder,
            predictor=self.predictor,
            splitter=self.splitter,
        )

        self._tree.n_rows_fit = X.shape[0]
        self._tree.n_rows_predict = X.shape[0]
        self._tree.X_n_rows = X.shape[0]
        self._tree.n_features = X.shape[1]

        if not self.skip_check_input:
            sample_weight = self._check_sample_weight(
                sample_weight=sample_weight)
            sample_indices = self._check_sample_indices(
                sample_indices=sample_indices)

        builder = DepthTreeBuilder(
            X=X,
            Y=Y,
            sample_indices=sample_indices,
            max_features=self.max_features,
            sample_weight=sample_weight,
            criteria=self.criteria,
            leaf_builder=self.leaf_builder,
            predictor=self.predictor,
            splitter=self.splitter,
        )
        builder.build_tree(self._tree)

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
        if self.predictor_instance is None:
            raise AttributeError(
                "The tree has not been fitted before trying to call predict"
            )
        if not self.skip_check_input:
            X, _ = self._check_input(X)
            self._check_dimensions(X)
        return self._tree.predict(X=X, **kwargs)

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
        if (X is not None) and not self.skip_check_input:
            X, _ = self._check_input(X)
            self._check_dimensions(X)
        return self._tree.predict_weights(X=X, scale=scale)

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
        if (X is not None) and not self.skip_check_input:
            X, _ = self._check_input(X)
            self._check_dimensions(X)
        return self._tree.predict_leaf(X=X)

    def _tree_based_weights(
            self,
            hash0: dict,
            hash1: dict,
            size_X0: int,
            size_X1: int,
            scaling: str) -> np.ndarray:
        return self._tree._tree_based_weights(
            hash0=hash0,
            hash1=hash1,
            size_X0=size_X0,
            size_X1=size_X1,
            scaling=scaling)

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
        if not self.skip_check_input:
            X0, _ = self._check_input(X0)
            self._check_dimensions(X0)
            X1, _ = self._check_input(X1)
            self._check_dimensions(X1)

        return self._tree.similarity(X0=X0, X1=X1)

    def refit_leaf_nodes(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        sample_indices: ArrayLike | None = None,
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
        if not self.skip_check_input:
            X, Y = self._check_input(X, Y)
            self._check_dimensions(X)
            sample_weight = self._check_sample_weight(sample_weight)
            sample_indices = self._check_sample_indices(sample_indices)
        return self._tree.refit_leaf_nodes(
            X=X,
            Y=Y,
            sample_weight=sample_weight,
            sample_indices=sample_indices,
        )
