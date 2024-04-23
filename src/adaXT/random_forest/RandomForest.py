from multiprocessing import Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import BaseManager

from ..decision_tree import DecisionTree
from ..criteria import Criteria, Squared_error
from ..decision_tree.splitter import Splitter


import numpy as np
from numpy import float64 as DOUBLE
import sys
import dill


def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)


def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))


class SharedNumpyArray:
    """
    Wraps a numpy array so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing.
    """

    def __init__(self, array):
        """
        Creates the shared memory and copies the array therein
        """
        # create the shared memory location of the same size of the array
        self._shared = SharedMemory(create=True, size=array.nbytes)

        # save data type and shape, necessary to read the data correctly
        self._dtype, self._shape = array.dtype, array.shape

        # create a new numpy array that uses the shared memory we created
        # at first, it is filled with zeros
        res = np.ndarray(self._shape, dtype=self._dtype, buffer=self._shared.buf)

        # copy data from the array to the shared memory. numpy will
        # take care of copying everything in the correct format
        res[:] = array[:]

    def read(self):
        """
        Reads the array from the shared memory without unnecessary copying.
        """
        # simply create an array of the correct shape and type,
        # using the shared memory location we created earlier
        return np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)

    def unlink(self):
        """
        Releases the allocated memory. Call when finished using the data,
        or when the data was copied somewhere else.
        """
        self._shared.close()
        self._shared.unlink()


class RandomForest:
    """
    The Random Forest
    """

    def __init__(
        self,
        forest_type: str,
        criterion: type[Criteria],
        n_estimators: int = 100,
        bootstrap: bool = True,
        n_jobs: int = -1,
        max_samples: int | None = None,
        max_features: int | None = None,
        random_state: int | None = None,
        max_depth: int = sys.maxsize,
        impurity_tol: float = 0,
        min_samples_split: int = 1,
        min_samples_leaf: int = 1,
        min_improvement: float = 0,
        splitter: Splitter | None = None,
    ):
        """
        Parameters
        ----------
        forest_type : str
            Classification or Regression
        criterion : Criteria
            The criteria function used to evaluate a split in a DecisionTree
        n_estimators : int, default=100
            The number of trees in the forest.
        bootstrap : bool, default=True
            Whether bootstrap is used when building trees
        n_jobs : int, default=1
            The number of processes used to fit, and predict for the forest, -1 uses all available proccesors
        max_samples : int, default=None
            The number of samples drawn when doing bootstrap
        max_features: int, float, {"sqrt", "log2"}, default=None
            The number of features used when doing feature-sampling
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
        if random_state:
            np.random.seed(random_state)
        self.forest_type = forest_type
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_depth = max_depth
        self.impurity_tol = impurity_tol
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_improvement = min_improvement
        self.splitter = splitter
        self.forest_fitted = False
        BaseManager.register("DecisionTree", DecisionTree)
        self.manager = BaseManager()
        self.manager.start()
        self.trees = [
            self.manager.DecisionTree(
                tree_type=self.forest_type,
                criteria=self.criterion,
                max_depth=self.max_depth,
                impurity_tol=self.impurity_tol,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_improvement=self.min_improvement,
                max_features=self.max_features,
                splitter=self.splitter,
                skip_check_input=True,
            )
            for _ in range(self.n_estimators)
        ]

    def __del__(self):
        self.X.unlink()
        self.Y.unlink()

    # Function used to call the fit function of a tree
    @staticmethod
    def __build_single_tree(
        tree: DecisionTree, X: SharedNumpyArray, Y: SharedNumpyArray, max_features: int
    ):
        # subset the feature indices
        X_val = X.read()
        tree.fit(
            X_val,
            Y.read(),
            sample_indices=RandomForest.__get_sample_indices(
                X_val.shape[0], max_features
            ),
        )

    # Function to build all the trees of the forest, differentiates between
    # running in parallel and sequential
    def __build_trees(self):
        if self.n_jobs == 1:
            for tree in self.trees:
                tree.fit(self.X.read(), self.Y.read())
        else:
            with Pool(self.n_jobs) as p:
                jobs = []
                for tree in self.trees:
                    jobs.append(
                        apply_async(
                            p,
                            RandomForest.__build_single_tree,
                            args=(tree, self.X, self.Y, self.max_features),
                        )
                    )
                for job in jobs:
                    job.get()

    # Function used to call the predict function of a tree
    @staticmethod
    def __predict_single_tree(tree: DecisionTree, predict_values: SharedNumpyArray):
        return np.array(tree.predict(predict_values.read()))

    # Function to call predict on all the trees of the forest, differentiates
    # between running in parallel and sequential
    def __predict_trees(self, X: np.ndarray):
        predictions = []
        if self.n_jobs == 1:
            for tree in self.trees:
                predictions.append(tree.predict(X))
        else:
            predict_value = SharedNumpyArray(X)
            jobs = []
            with Pool(self.n_jobs) as p:
                for tree in self.trees:
                    jobs.append(
                        apply_async(
                            p,
                            RandomForest.__predict_single_tree,
                            args=(tree, predict_value),
                        )
                    )
                for job in jobs:
                    predictions.append(job.get())
            predict_value.unlink()

        return np.column_stack(predictions)

    # Function used to add a column with zeros for all the classes that are in
    # the forest but not in a given tree
    def __fill_with_zeros_for_missing_classes_in_tree(
        self, tree_classes, predict_proba, num_rows_predict
    ):
        ret_val = np.zeros((num_rows_predict, len(self.classes)))

        # Find the indices of tree_classes in forest_classes
        tree_class_indices = np.searchsorted(self.classes, tree_classes)

        # Only update columns corresponding to tree_classes
        ret_val[:, tree_class_indices] = predict_proba

        return ret_val

    # Function to call predict_proba for a tree
    def __predict_proba_single_tree(self, tree: DecisionTree):
        tree_predict_proba = tree.predict_proba(self.predict_values.read())
        ret_val = self.__fill_with_zeros_for_missing_classes_in_tree(
            tree.classes, tree_predict_proba, self.predict_values._shape[0]
        )
        return ret_val

    # Function to call predict_proba on all the trees of the forest,
    # differentiates between running in parallel and sequential
    def __predict_proba_trees(self):
        predictions = []

        if self.n_jobs == 1:
            for tree in self.trees:
                predictions.append(self.__predict_proba_single_tree(tree))
        else:
            with Pool(self.n_jobs) as p:
                predictions = p.map(self.__predict_proba_single_tree, self.trees)

        return predictions

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
        if not self.bootstrap and self.max_samples:
            raise AttributeError("Bootstrap can not be False while max_samples is set")

        X, Y = self.__check_input(X, Y)
        self.X = SharedNumpyArray(X)
        self.Y = SharedNumpyArray(Y)
        self.n_obs, self.n_features = self.X._shape
        if self.forest_type == "Classification":
            self.classes = np.unique(self.Y.read())
        if self.max_samples is None:
            self.max_samples = self.n_obs

        # Fit trees
        self.__build_trees()

        # Register that the forest was succesfully fitted
        self.forest_fitted = True

        return self

    # Function that returns an ndarray of random ints used as the
    # sample_indices
    @staticmethod
    def __get_sample_indices(n_obs: int, max_samples: int | None):
        if max_samples:
            return np.random.randint(low=0, high=n_obs, size=max_samples)
        else:
            return None

    def predict(self, X: np.ndarray):
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
        tree_predictions = self.__predict_trees(X)

        if self.forest_type == "Regression":
            # Return the mean answer from all trees for each row
            return np.mean(tree_predictions, axis=1)

        elif self.forest_type == "Classification":
            # Return the element most voted for by trees for each row
            # QUESTION: how to handle equal majority vote?
            return np.apply_along_axis(
                self.__most_frequent_element, 1, tree_predictions
            )

    # Function used to find the most frequent element of an array
    def __most_frequent_element(self, arr):
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]

    def predict_proba(self, X: np.ndarray):
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
        self.check_dimensions(X)

        self.predict_values = SharedNumpyArray(X)
        # Predict_proba using all the trees, each element of list is the
        # predict_proba from one tree
        tree_predictions = self.__predict_proba_trees()
        self.predict_values.unlink()

        # Stack the predict_probas
        stacked_tree_predictions = np.stack(tree_predictions, axis=0)

        # Return the mean along the newly created axis
        return np.mean(stacked_tree_predictions, axis=0)

    def check_dimensions(self, X: np.ndarray):
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
