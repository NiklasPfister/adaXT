import ctypes
import multiprocessing
import sys
from functools import partial
from multiprocessing import RawArray, cpu_count
from multiprocessing.managers import BaseManager
from numbers import Integral
from typing import Literal

import numpy as np
from numpy import float64 as DOUBLE

from ..criteria import Criteria
from ..decision_tree import DecisionTree
from ..decision_tree.splitter import Splitter
from ..base_model import BaseModel
from ..predict import Predict
from ..leaf_builder import LeafBuilder


def get_single_leaf(tree: DecisionTree, scale: bool):
    return tree.get_leaf_matrix(scale=scale)


def get_sample_indices(
    n_obs: int,
    max_samples: int,
    random_state: np.random.RandomState,
    sampling: str | None,
):
    if max_samples:
        return random_state.randint(low=0, high=n_obs, size=max_samples)
    else:
        return None


def build_single_tree(
    sample_indices: np.ndarray | None,
    X: np.ndarray,
    Y: np.ndarray,
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
    tree.fit(X, Y, sample_indices=sample_indices)

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
    tree_predict_proba = tree.predict_proba(predict_values)
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
    The Random Forest
    """

    def __init__(
        self,
        forest_type: str | None,
        n_estimators: int = 100,
        n_jobs: int = -1,
        sampling: str | None = "bootstrap",
        max_samples: int | float | None = None,
        max_features: int | float | Literal["sqrt", "log2"] | None = None,
        random_state: int | None = None,
        max_depth: int = sys.maxsize,
        impurity_tol: float = 0,
        min_samples_split: int = 1,
        min_samples_leaf: int = 1,
        min_improvement: float = 0,
        criteria: type[Criteria] | None = None,
        leaf_builder: type[LeafBuilder] | None = None,
        predict: type[Predict] | None = None,
        splitter: type[Splitter] | None = None,
    ):
        #TODO: Update bootstrap to instead be sampling
        """
        Parameters
        ----------
        forest_type : str
            Classification or Regression
        criteria : Criteria
            The criteria function used to evaluate a split in a DecisionTree
        n_estimators : int, default=100
            The number of trees in the forest.
        bootstrap : bool, default=True
            Whether bootstrap is used when building trees
        n_jobs : int, default=1
            The number of processes used to fit, and predict for the forest, -1 uses all available proccesors
        max_samples : int | float | None, default=None
            The number of samples drawn when doing bootstrap
        max_features: int | float | Literal["sqrt", "log2"] | None = None
            The number of features to consider when looking for a split,
        random_state: int
            Used for deterministic seeding of the tree
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
        self.check_tree_type(forest_type, criteria, splitter, leaf_builder, predict)
        self.ctx = multiprocessing.get_context("spawn")
        self.X, self.Y = None, None
        self.max_features = max_features
        self.forest_type = forest_type
        self.n_estimators = n_estimators
        self.sampling = sampling
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.impurity_tol = impurity_tol
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_improvement = min_improvement
        self.forest_fitted = False
        BaseManager.register("RandomState", np.random.RandomState)
        self.manager = BaseManager()
        self.manager.start()
        self.random_state = self.__get_random_state(random_state)

    def __get_random_state(self, random_state):
        if isinstance(random_state, Integral) or (random_state is None):
            return self.manager.RandomState(random_state)
        else:
            raise ValueError("Random state either has to be Integral or None")

    def __get_max_samples(self, max_samples):
        if isinstance(max_samples, int):
            if max_samples > self.n_obs:
                raise ValueError("max_samples can not be larger than total samples")
            return max_samples
        elif isinstance(max_samples, float):
            return max(round(self.n_obs * max_samples), 1)

    # Function to build all the trees of the forest, differentiates between
    # running in parallel and sequential
    def __build_trees(self):
        if self.n_jobs == 1:
            for tree in self.trees:
                tree.fit(self.X, self.Y)
        else:
            max_samples = self.__get_max_samples(self.max_samples)
            partial_func = partial(
                build_single_tree,
                X=self.X,
                Y=self.Y,
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
            )
            partial_sample = partial(
                get_sample_indices,
                random_state=self.random_state,
                n_obs=self.n_rows,
                max_samples=max_samples,
            )
            with self.ctx.Pool(self.n_jobs) as p:
                sample_indices = [
                    p.apply(partial_sample) for _ in range(self.n_estimators)
                ]
                promise = p.map_async(partial_func, sample_indices)
                trees = promise.get()
                self.trees = trees

    # Function to call predict on all the trees of the forest, differentiates
    # between running in parallel and sequential
    def __predict_trees(self, X: np.ndarray, **kwargs):
        predictions = []
        if self.n_jobs == 1:
            for tree in self.trees:
                predictions.append(tree.predict(X, **kwargs))
        else:
            predict_value = shared_numpy_array(X)
            partial_func = partial(
                predict_single_tree, predict_values=predict_value, **kwargs
            )
            with self.ctx.Pool(self.n_jobs) as p:
                promise = p.map_async(partial_func, self.trees)
                predictions = promise.get()
        return np.column_stack(predictions)

    # Function to call predict_proba on all the trees of the forest,
    # differentiates between running in parallel and sequential
    def __predict_proba_trees(self, X: np.ndarray, **kwargs):
        predictions = []
        if self.n_jobs == 1:
            for tree in self.trees:
                predictions.append(tree.predict(X))
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

        return predictions

    def __check_dimensions(self, X: np.ndarray):
        # If there is only a single point
        if X.ndim == 1:
            if X.shape[0] != self.n_features:
                raise ValueError(
                    f"Number of features should be {self.n_features}, got {X.shape[0]}"
                )
        else:
            if X.shape[1] != self.n_features:
                raise ValueError(
                    f"Dimension should be {self.n_features}, got {X.shape[1]}"
                )

    def __check_input(self, X: np.ndarray, Y: np.ndarray):
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

        # Make sure input arrays are c contigous
        X = np.ascontiguousarray(X, dtype=DOUBLE)
        Y = np.ascontiguousarray(Y, dtype=DOUBLE)

        return X, Y

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Function used to fit a forest using many DecisionTrees for the given data

        Parameters
        ----------
        X : np.ndarray
            feature values
        Y : np.ndarray
            response values
        """
        self.X, self.Y = self.__check_input(X, Y)
        self.X = shared_numpy_array(X)
        self.Y = shared_numpy_array(Y)
        self.n_rows, self.n_features = self.X.shape
        if self.forest_type == "Classification":
            self.classes = np.unique(self.Y)
        if self.max_samples is None:
            self.max_samples = self.n_rows

        # Fit trees
        self.__build_trees()

        # Register that the forest was succesfully fitted
        self.forest_fitted = True

        return self

    def predict(self, X: np.ndarray, **kwargs):
        """
        Predicts a y-value for given X values using the trees of the forest. In the case of a classification tree with equal majority vote,
        the lowest class that has maximum votes is returned.

        Parameters
        ----------
        X : np.ndarray
            (N, M) numpy array with features to predict

        Returns
        -------
        np.ndarray
            (N) numpy array with the prediction
        """
        if not self.forest_fitted:
            raise AttributeError(
                "The forest has not been fitted before trying to call predict"
            )

        # Predict using all the trees, each column is the predictions from one
        # tree
        tree_predictions = self.__predict_trees(X, **kwargs)

        return self.predict_class.forest_predict(tree_predictions, **kwargs)

    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predicts a probability for each response for given X values using the trees of the forest

        Parameters
        ----------
        X : np.ndarray
            (N, M) numpy array with features to predict

        Returns
        -------
        np.ndarray
            Returns an ndarray with the probabilities for each class per observation in X. The order of the classes corresponds to that in the attribute classes
        """
        if not self.forest_fitted:
            raise AttributeError(
                "The forest has not been fitted before trying to call predict_proba"
            )

        # Make sure that predict_proba is only called on Classification
        # forests
        if self.forest_type != "Classification":
            raise ValueError(
                "predict_proba can only be called on a Classification tree"
            )

        # Check dimensions
        self.__check_dimensions(X)
        # Predict_proba using all the trees, each element of list is the
        # predict_proba from one tree
        tree_predictions = self.__predict_proba_trees(X, **kwargs)
        return self.predict_class.forest_predict_proba(tree_predictions, **kwargs)

    def get_forest_weight(self) -> np.ndarray:
        if not self.forest_fitted:
            raise AttributeError(
                "The forest has not been fitted before trying to call get_forest_weight"
            )
        partial_func = partial(get_single_leaf, scale=False)
        with self.ctx.Pool(self.n_jobs) as p:
            promise = p.map_async(partial_func, self.trees)
            tree_weights = promise.get()
        return np.sum(tree_weights, axis=0) / self.n_estimators
