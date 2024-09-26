import sys
from typing import Iterable, Literal

import numpy as np
from numpy import float64 as DOUBLE
from numpy.random import Generator, default_rng

from adaXT.parallel import ParallelModel, shared_numpy_array

from numpy.typing import ArrayLike

from ..criteria import Criteria
from ..decision_tree import DecisionTree
from ..decision_tree.splitter import Splitter
from ..base_model import BaseModel
from ..predict import Predict
from ..leaf_builder import LeafBuilder


def tree_based_weights(
    tree: DecisionTree,
    X0: np.ndarray | None,
    X1: np.ndarray | None,
    size_X0: int,
    size_X1: int,
    scaling: str,
) -> np.ndarray:
    hash0 = tree.predict_leaf(X=X0)
    hash1 = tree.predict_leaf(X=X1)
    return tree._tree_based_weights(
        hash0=hash0,
        hash1=hash1,
        size_X0=size_X0,
        size_X1=size_X1,
        scaling=scaling,
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
        return (
            gen.integers(
                low=0,
                high=n_rows,
                size=sampling_parameter),
            None)
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
    tree.fit(
        X=X,
        Y=Y,
        sample_indices=fitting_indices,
        sample_weight=sample_weight)
    if honest_tree:
        tree.refit_leaf_nodes(
            X=X,
            Y=Y,
            sample_weight=sample_weight,
            sample_indices=prediction_indices)

    return tree


def predict_single_tree(
        tree: DecisionTree,
        predict_values: np.ndarray,
        **kwargs):
    return tree.predict(predict_values, **kwargs)


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

        self._check_tree_type(
            forest_type,
            criteria,
            splitter,
            leaf_builder,
            predict)

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
                raise ValueError(
                    "The split index does not fit for the given dataset")

            if isinstance(number_chosen, float):
                return (
                    split_idx, max(
                        round(
                            self.n_rows * sampling_parameter), 1))
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
            raise ValueError(
                f"Provided sampling ({self.sampling}) does not exist")

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
        self.fitting_indices, self.prediction_indices = fitting_prediction_indices
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

    def fit(self, X: ArrayLike, Y: ArrayLike,
            sample_weight: np.ndarray | None = None):
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
        X, Y = self._check_input(X, Y)
        self.X = shared_numpy_array(X)
        self.Y = shared_numpy_array(Y)
        self.n_rows, self.n_features = self.X.shape

        if sample_weight is None:
            self.sample_weight = np.ones(self.X.shape[0])
        else:
            self.sample_weight = sample_weight
        self.sampling_parameter = self.__get_sampling_parameter(
            self.sampling_parameter)
        # Fit trees
        self.__build_trees()

        # Register that the forest was succesfully fitted
        self.forest_fitted = True

        return self

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

        X, _ = self._check_input(X)
        self._check_dimensions(X)

        predict_value = shared_numpy_array(X)
        prediction = self.predict_class.forest_predict(
            X_old=self.X,
            Y_old=self.Y,
            X_new=predict_value,
            trees=self.trees,
            parallel=self.parallel,
            **kwargs,
        )
        return prediction

    def predict_weights(
        self, X: np.ndarray | None = None, scale: bool = True
    ) -> np.ndarray:
        if X is None:
            size_0 = self.n_rows
            X = self.X
        else:
            X, _ = self._check_input(X)
            self._check_dimensions(X)
            X = shared_numpy_array(X)
            size_0 = X.shape[0]

        if scale:
            scaling = "row"
        else:
            scaling = "none"

        weight_list = self.parallel.async_map(
            tree_based_weights,
            map_input=self.trees,
            X0=X,
            X1=None,
            size_X0=size_0,
            size_X1=self.n_rows,
            scaling=scaling,
        )

        if scale:
            ret = np.mean(weight_list, axis=0)
        else:
            ret = np.sum(weight_list, axis=0)
        return ret

    def similarity(self, X0: np.ndarray, X1: np.ndarray):
        X0, _ = self._check_input(X0)
        self._check_dimensions(X0)
        X1, _ = self._check_input(X1)
        self._check_dimensions(X1)

        size_0 = X0.shape[0]
        size_1 = X1.shape[0]
        weight_list = self.parallel.async_map(
            tree_based_weights,
            map_input=self.trees,
            X0=X0,
            X1=X1,
            size_X0=size_0,
            size_X1=size_1,
            scaling="similarity",
        )
        return np.mean(weight_list, axis=0)
