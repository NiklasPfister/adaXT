import ctypes
from multiprocessing.managers import BaseManager
import sys
from functools import partial
from multiprocessing import RawArray
from typing import Iterable, Literal

import numpy as np
from numpy import float64 as DOUBLE
from numpy.random import Generator, SeedSequence, default_rng

from adaXT.parallel_class import ParallelModel

from numpy.typing import ArrayLike

from ..criteria import Criteria
from ..decision_tree import DecisionTree
from ..decision_tree.splitter import Splitter
from ..base_model import BaseModel
from ..predict import Predict
from ..leaf_builder import LeafBuilder


def get_single_leaf(tree: DecisionTree, X: np.ndarray | None = None) -> dict:
    return tree.predict_leaf(X=X)


def tree_based_weights(
    tree: DecisionTree,
    X0: np.ndarray,
    X1: np.ndarray,
    size_X0: int,
    size_X1: int,
    scale: bool,
) -> np.ndarray:
    hash1 = tree.predict_leaf(X=X0)
    hash2 = tree.predict_leaf(X=X1)
    return tree.tree_based_weights(
        hash1=hash1,
        hash2=hash2,
        size_X0=size_X0,
        size_X1=size_X1,
        scale=scale,
    )


def get_sample_indices(
    gen: Generator,
    n_rows: int,
    sampling_parameter: int | tuple[int, int],
    sampling: str | None,
) -> Iterable:
    """
    Assumes there has been a previous call to self.__get_sample_indices on the
    RandomForest.
    """
    if sampling == "bootstrap":
        return (gen.integers(low=0, high=n_rows, size=sampling_parameter), None)
    elif sampling == "honest_tree":
        indices = np.arange(0, n_rows)
        gen.shuffle(indices)
        return (indices[:sampling_parameter], indices[sampling_parameter:])
    elif sampling == "honest_forest":
        fitting_indices = gen.integers(
            low=0, high=sampling_parameter[0], size=sampling_parameter[1]
        )
        prediction_indices = gen.integers(
            low=sampling_parameter[0], high=n_rows, size=sampling_parameter[1]
        )
        return (fitting_indices, prediction_indices)
    else:
        return (None, None)


def build_single_tree(
    fitting_indices: np.ndarray | None,
    prediction_indices: np.ndarray | None,
    X: np.ndarray,
    Y: np.ndarray,
    honest_tree: bool,
    criteria: type[Criteria],
    predict: type[Predict],
    leaf_builder: type[LeafBuilder],
    splitter: type[Splitter],
    tree_type: str | None = None,
    max_depth: int = sys.maxsize,
    impurity_tol: float = 0,
    min_samples_split: int = 1,
    min_samples_leaf: int = 1,
    min_improvement: float = 0,
    max_features: int | float | Literal["sqrt", "log2"] | None = None,
    skip_check_input: bool = True,
    sample_weight: np.ndarray | None = None,
):
    # subset the feature indices
    tree = DecisionTree(
        tree_type=tree_type,
        max_depth=max_depth,
        impurity_tol=impurity_tol,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_improvement=min_improvement,
        max_features=max_features,
        skip_check_input=skip_check_input,
        criteria=criteria,
        leaf_builder=leaf_builder,
        predict=predict,
        splitter=splitter,
    )
    tree.fit(X=X, Y=Y, sample_indices=fitting_indices, sample_weight=sample_weight)
    if honest_tree:
        tree.refit_leaf_nodes(
            X=X, Y=Y, sample_weight=sample_weight, sample_indices=prediction_indices
        )

    return tree


# Function used to add a column with zeros for all the classes that are in
# the forest but not in a given tree


def fill_with_zeros_for_missing_classes_in_tree(
    tree_classes, predict_proba, num_rows_predict, classes
):
    n_classes = len(classes)
    ret_val = np.zeros((num_rows_predict, n_classes))

    # Find the indices of tree_classes in forest_classes
    tree_class_indices = np.searchsorted(classes, tree_classes)

    # Only update columns corresponding to tree_classes
    ret_val[:, tree_class_indices] = predict_proba

    return ret_val


def predict_proba_single_tree(
    tree: DecisionTree, predict_values: np.ndarray, classes: np.ndarray, **kwargs
):
    tree_predict_proba = tree.predict(predict_values, predict_proba=True)
    ret_val = fill_with_zeros_for_missing_classes_in_tree(
        tree.classes,
        tree_predict_proba,
        predict_values.shape[0],
        classes=classes,
        **kwargs,
    )
    return ret_val


def predict_single_tree(tree: DecisionTree, predict_values: np.ndarray, **kwargs):
    return tree.predict(predict_values, **kwargs)


def shared_numpy_array(array):
    if array.ndim == 2:
        row, col = array.shape
        shared_array = RawArray(ctypes.c_double, (row * col))
        shared_array_np = np.ndarray(
            shape=(row, col), dtype=np.double, buffer=shared_array
        )
    else:
        row = array.shape[0]
        shared_array = RawArray(ctypes.c_double, row)
        shared_array_np = np.ndarray(shape=row, dtype=np.double, buffer=shared_array)
    np.copyto(shared_array_np, array)
    return shared_array_np


class RandomForest(BaseModel):
    """
    Attributes
    ----------
    max_features: int | float | Literal["sqrt", "log2"] | None = None
        The number of features to consider when looking for a split.
    max_depth : int
        The maximum depth of the tree.
    forest_type : str
        The type of random forest, either  a string specifying a supported type
        (currently "Regression", "Classification" and "Quantile") or None.
    n_estimators : int
        The number of trees in the random forest.
    n_jobs : int
        The number of processes used to fit, and predict for the forest, -1
        uses all available proccesors.
    sampling: str | None
        Either bootstrap, honest_tree, honest_forest or None. See
        sampling_parameter for exact behaviour.
    sampling_parameter: int | float | tuple[int, int|float] | None
        A parameter used to control the behavior of the sampling. For
        bootstrap it can be an int representing the number of randomly
        drawn indices (with replacement) to fit on or a float for a
        percentage.
        For honest_forest it is a tuple of two ints: The first
        value specifies a splitting index such that the indices on the left
        are used in the fitting of all trees and the indices on the right
        are used for prediction (i.e., populating the leafs). The second
        value specifies the number of randomly drawn (with replacement)
        indices used for both fitting and prediction.
        For honest_tree it is the number of elements to use for both fitting
        and prediction, where there might be overlap between trees in
        fitting and prediction data, but not for an individual tree.
        If None, all samples are used for each tree.
    impurity_tol : float
        The tolerance of impurity in a leaf node.
    min_samples_split : int
        The minimum number of samples in a split.
    min_samples_leaf : int
        The minimum number of samples in a leaf node.
    min_improvement: float
        The minimum improvement gained from performing a split.
    """

    # TODO: Save prediction_indicies and fitting_indicies.
    # TODO: predict_tree_weights, which create an NxN matrix similair to
    # the tree.predict_tree_weights
    # TODO: Similairity.
    def __init__(
        self,
        forest_type: str | None,
        n_estimators: int = 100,
        n_jobs: int = -1,
        sampling: str | None = "bootstrap",
        sampling_parameter: int | float | tuple[int, int] | None = None,
        max_features: int | float | Literal["sqrt", "log2"] | None = None,
        max_depth: int = sys.maxsize,
        impurity_tol: float = 0,
        min_samples_split: int = 1,
        min_samples_leaf: int = 1,
        min_improvement: float = 0,
        seed: int | None = None,
        criteria: type[Criteria] | None = None,
        leaf_builder: type[LeafBuilder] | None = None,
        predict: type[Predict] | None = None,
        splitter: type[Splitter] | None = None,
    ):
        """
        Parameters
        ----------
        forest_type : str
            The type of random forest, either  a string specifying a supported type
            (currently "Regression", "Classification" and "Quantile") or None.
        n_estimators : int
            The number of trees in the random forest.
        n_jobs : int
            The number of processes used to fit, and predict for the forest, -1
            uses all available proccesors.
        sampling: str | None
            Either bootstrap, honest_tree, honest_forest or None. See
            sampling_parameter for exact behaviour.
        sampling_parameter: int | float | tuple[int, int|float] | None
            A parameter used to control the behavior of the sampling. For
            bootstrap it can be an int representing the number of randomly
            drawn indices (with replacement) to fit on or a float for a
            percentage.
            For honest_forest it is a tuple of two ints: The first
            value specifies a splitting index such that the indices on the left
            are used in the fitting of all trees and the indices on the right
            are used for prediction (i.e., populating the leafs). The second
            value specifies the number of randomly drawn (with replacement)
            indices used for both fitting and prediction.
            For honest_tree it is the number of elements to use for both fitting
            and prediction, where there might be overlap between trees in
            fitting and prediction data, but not for an individual tree.
            If None, all samples are used for each tree.
        max_features: int | float | Literal["sqrt", "log2"] | None = None
            The number of features to consider when looking for a split.
        max_depth : int
            The maximum depth of the tree.
        impurity_tol : float
            The tolerance of impurity in a leaf node.
        min_samples_split : int
            The minimum number of samples in a split.
        min_samples_leaf : int
            The minimum number of samples in a leaf node.
        min_improvement: float
            The minimum improvement gained from performing a split.
        seed: int | None
            Seed used to reproduce a RandomForest
        criteria : Criteria
            The Criteria class to use, if None it defaults to the forest_type
            default.
        leaf_builder : LeafBuilder
            The LeafBuilder class to use, if None it defaults to the forest_type
            default.
        predict: Predict
            The Prediction class to use, if None it defaults to the forest_type
            default.
        splitter : Splitter | None, optional
            The Splitter class to use, if None it defaults to the default
            Splitter class.
        """
        # Must initialize Manager before ParallelModel
        self.parent_rng = self.__get_random_generator(seed)

        # Make context the same from when getting indices and using
        # parallelModel
        self.parallel = ParallelModel(n_jobs=n_jobs)

        self.check_tree_type(forest_type, criteria, splitter, leaf_builder, predict)

        self.X, self.Y = None, None
        self.max_features = max_features
        self.forest_type = forest_type
        self.n_estimators = n_estimators
        self.sampling = sampling
        self.sampling_parameter = sampling_parameter
        self.max_depth = max_depth
        self.impurity_tol = impurity_tol
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_improvement = min_improvement
        self.forest_fitted = False

    # Check whether dimension of X matches self.n_features
    def __check_dimensions(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Number of features should be {self.n_features}, got {X.shape[1]}"
            )

    # Check whether X and Y match and convert array-like to ndarray
    def __check_input(
        self, X: ArrayLike, Y: ArrayLike | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        Y_check = Y is not None
        # Make sure input arrays are c contigous
        X = np.ascontiguousarray(X, dtype=DOUBLE)
        Y = np.ascontiguousarray(Y, dtype=DOUBLE)

        # Check that X is two dimensional
        if X.ndim != 2:
            raise ValueError("X should be two-dimensional")

        # If Y is not None perform checks for Y
        if Y_check:
            # Check if X and Y has same number of rows
            if X.shape[0] != Y.shape[0]:
                raise ValueError("X and Y should have the same number of rows")

            # Check if Y has dimensions (n, 1) or (n,)
            if 2 < Y.ndim:
                raise ValueError("Y should have dimensions (n,1) or (n,)")
            elif 2 == Y.ndim:
                if 1 < Y.shape[1]:
                    raise ValueError("Y should have dimensions (n,1) or (n,)")
                else:
                    Y = Y.reshape(-1)
        return X, Y

    def __get_random_generator(self, seed):
        if isinstance(seed, int) or (seed is None):
            return default_rng(seed)
        else:
            raise ValueError("Random state either has to be Integral or None")

    def __get_sampling_parameter(self, sampling_parameter):
        if self.sampling == "bootstrap":
            if isinstance(sampling_parameter, int):
                return sampling_parameter
            elif isinstance(sampling_parameter, float):
                return max(round(self.n_rows * sampling_parameter), 1)
            elif sampling_parameter is None:
                return self.n_rows
            raise ValueError(
                "Provided sampling_parameter is not an integer, a float or None as required."
            )
        elif self.sampling == "honest_forest":
            if sampling_parameter is None:
                sampling_parameter = (self.n_rows // 2, self.n_rows // 2)
            elif not isinstance(sampling_parameter, tuple):
                raise ValueError(
                    "The provided sampling parameter is not a tuple for honest_forest."
                )
            split_idx, number_chosen = sampling_parameter
            if not isinstance(split_idx, int):
                raise ValueError(
                    "The provided splitting index (given as the first entry in sampling_parameter) is not an integer"
                )
            if (split_idx > self.n_rows) or (split_idx < 0):
                raise ValueError("The split index does not fit for the given dataset")

            if isinstance(number_chosen, float):
                return (split_idx, max(round(self.n_rows * sampling_parameter), 1))
            elif isinstance(number_chosen, int):
                return (split_idx, number_chosen)

            elif number_chosen is None:
                return (split_idx, int(self.n_rows / 2))

            raise ValueError(
                "The provided number of resamples (given as the second entry in sampling_parameter) is not an integer, float or None"
            )

        elif self.sampling == "honest_tree":
            if sampling_parameter is None:
                sampling_parameter = int(self.n_rows / 2)
            if isinstance(sampling_parameter, int):
                if sampling_parameter > self.n_rows:
                    raise ValueError(
                        "Sample parameter can not be larger than number of rows of X"
                    )
                return sampling_parameter
            elif isinstance(sampling_parameter, float):
                if (sampling_parameter < 0) or (sampling_parameter > 1):
                    raise ValueError(
                        "Sampling parameter must be between 0 and 1 for a float with honest_tree"
                    )
                return max(round(self.n_rows * sampling_parameter), 1)
            else:
                raise ValueError(
                    "Provided sampling parameter is not an integer a float of None"
                )
        elif self.sampling is None:
            return None
        else:
            raise ValueError(f"Provided sampling ({self.sampling}) does not exist")

    def __is_honest(self) -> bool:
        return self.sampling in ["honest_tree", "honest_forest"]

    # Function to build all the trees of the forest, differentiates between
    # running in parallel and sequential

    def __build_trees(self):
        # parent_rng.spawn() spawns random generators that children can use
        fitting_prediction_indices = self.parallel.async_map(
            get_sample_indices,
            map_input=self.parent_rng.spawn(self.n_estimators),
            n_rows=self.n_rows,
            sampling_parameter=self.sampling_parameter,
            sampling=self.sampling,
        )
        self.trees = self.parallel.async_starmap(
            build_single_tree,
            map_input=fitting_prediction_indices,
            X=self.X,
            Y=self.Y,
            honest_tree=self.__is_honest(),
            criteria=self.criteria_class,
            predict=self.predict_class,
            leaf_builder=self.leaf_builder_class,
            splitter=self.splitter,
            tree_type=self.forest_type,
            max_depth=self.max_depth,
            impurity_tol=self.impurity_tol,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_improvement=self.min_improvement,
            max_features=self.max_features,
            skip_check_input=True,
            sample_weight=self.sample_weight,
        )

    def fit(
        self, X: np.ndarray, Y: np.ndarray, sample_weight: np.ndarray | None = None
    ):
        """
        Fit the random forest with training data (X, Y).

        Parameters
        ----------
        X : array-like object of dimension 2
            The feature values used for training. Internally it will be
            converted to np.ndarray with dtype=np.float64.
        Y : array-like object
            The response values used for training. Internally it will be
            converted to np.ndarray with dtype=np.float64.
        sample_weight : np.ndarray | None
            Sample weights. Currently not implemented.
        """
        self.X, self.Y = self.__check_input(X, Y)
        self.X = shared_numpy_array(X)
        self.Y = shared_numpy_array(Y)
        self.n_rows, self.n_features = self.X.shape

        if sample_weight is None:
            self.sample_weight = np.ones(self.X.shape[0])
        else:
            self.sample_weight = sample_weight
        self.sampling_parameter = self.__get_sampling_parameter(self.sampling_parameter)
        # Fit trees
        self.__build_trees()

        # Register that the forest was succesfully fitted
        self.forest_fitted = True

        return self

    def __predict_trees(self, X: np.ndarray, **kwargs):
        predict_value = shared_numpy_array(X)
        return self.predict_class.forest_predict(
            X_old=self.X,
            Y_old=self.Y,
            X_new=predict_value,
            trees=self.trees,
            parallel=self.parallel,
            **kwargs,
        )

    # Function to call predict_proba on all the trees of the forest,
    # differentiates between running in parallel and sequential

    def __predict_proba_trees(self, X: np.ndarray, **kwargs):
        predictions = []
        if self.n_jobs == 1:
            for tree in self.trees:
                predictions.append(tree.predict(X, predict_proba=True))
        else:
            predict_value = shared_numpy_array(X)
            partial_func = partial(
                predict_proba_single_tree,
                predict_values=predict_value,
                classes=self.classes,
                **kwargs,
            )
            with self.ctx.Pool(self.n_jobs) as p:
                promise = p.map_async(partial_func, self.trees)
                predictions = promise.get()

        return np.stack(predictions, axis=-1)

    def predict(self, X: ArrayLike, **kwargs):
        """
        Predicts response values at X using fitted random forest.  The behavior
        of this function is determined by the Prediction class used in the
        decision tree. For currently existing tree types the corresponding
        behavior is as follows:

        Classification:
        ----------
        Returns the class based on majority vote among the trees. In the case
        of tie, the lowest class with the maximum number of votes is returned.

        Regression:
        ----------
        Returns the average response among all trees.

        Quantile:
        ----------
        Returns the conditional quantile of the response, where the quantile is
        specified by passing a list of quantiles via the `quantile` parameter.


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
        if not self.forest_fitted:
            raise AttributeError(
                "The forest has not been fitted before trying to call predict"
            )

        self.__check_dimensions(X)
        if "predict_proba" in kwargs.keys():
            if self.forest_type != "Classification":
                raise ValueError(
                    "predict_proba can only be called on a Classification tree"
                )
            prediction = self.__predict_proba_trees(X, **kwargs)
        else:
            prediction = self.__predict_trees(X, **kwargs)
        return prediction

    def predict_weights(
        self, X: np.ndarray | None = None, scale: bool = True
    ) -> np.ndarray:
        if X is None:
            size_0 = self.n_rows
        else:
            X = shared_numpy_array(X)
            size_0 = X.shape[0]

        if scale:
            scaling = 0
        else:
            scaling = -1

        weight_list = self.parallel.async_map(
            tree_based_weights,
            map_input=self.trees,
            X0=X,
            X1=None,
            size_X0=size_0,
            size_X1=self.n_rows,
            scale=scaling,
        )

        # TODO: How do we want to scale it. Should we do so here?
        ret = np.mean(weight_list, axis=0)
        return ret

    def similarity(self, X0: np.ndarray, X1: np.ndarray, scale: bool = True):
        if scale:
            scaling = 1
        else:
            scaling = -1
        size_0 = X0.shape[0]
        size_1 = X1.shape[0]
        weight_list = self.parallel.async_map(
            tree_based_weights,
            map_input=self.trees,
            X1=X0,
            X2=X1,
            size_X0=size_0,
            size_X1=size_1,
            scale=scaling,
        )
        return np.mean(weight_list, axis=0)
