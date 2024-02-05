from multiprocessing import Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory

from ..decision_tree import DecisionTree
from ..criteria import Criteria, Squared_error
from ..decision_tree.splitter import Splitter


import numpy as np
from numpy import float64 as DOUBLE
import sys

class SharedNumpyArray:
    '''
    Wraps a numpy array so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing.
    '''
    def __init__(self, array):
        '''
        Creates the shared memory and copies the array therein
        '''
        # create the shared memory location of the same size of the array
        self._shared = SharedMemory(create=True, size=array.nbytes)

        # save data type and shape, necessary to read the data correctly
        self._dtype, self._shape = array.dtype, array.shape

        # create a new numpy array that uses the shared memory we created.
        # at first, it is filled with zeros
        res = np.ndarray(
            self._shape, dtype=self._dtype, buffer=self._shared.buf
        )

        # copy data from the array to the shared memory. numpy will
        # take care of copying everything in the correct format
        res[:] = array[:]

    def read(self):
        '''
        Reads the array from the shared memory without unnecessary copying.
        '''
        # simply create an array of the correct shape and type,
        # using the shared memory location we created earlier
        return np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)

    def unlink(self):
        '''
        Releases the allocated memory. Call when finished using the data,
        or when the data was copied somewhere else.
        '''
        self._shared.close()
        self._shared.unlink()


class RandomForest:
    '''
    The Random Forrest
    '''
    def __init__(
            self,
            forrest_type: str,
            n_estimators: int = 100,
            criterion: Criteria = Squared_error,
            bootstrap: bool = True,
            n_jobs: int = 1,
            max_samples: int = None,
            max_depth: int = sys.maxsize,
            impurity_tol: float = 0,
            min_samples_split: int = 1,
            min_samples_leaf: int = 1,
            min_improvement: float = 0,
            splitter: Splitter | None = None):
        """
        Parameters
        ----------
        forrest_type : str
            Classification or Regression
        n_estimators : int, default=100
            The number of trees in the forest.
        criterion : Criteria, default=Squared_errror
            The criteria function used to evaluate a split
        bootstrap : bool, default=True
            Whether bootstrap is used when building trees
        n_jobs : int, default=1
            The number of processes used to fit, and predict for the forrest, -1 means using all proccesors
        max_samples : int, default=None
            The number of samples drawn from the feature values
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
        self.forrest_type = forrest_type
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.impurity_tol = impurity_tol
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_improvement = min_improvement
        self.splitter = splitter
        self.forest_fitted = False

    # Function used to call the fit function of a tree
    def _build_single_tree(self, tree:DecisionTree):
        # subset the feature indices
        tree.fit(self.features.read(), self.outcomes.read(), sample_indices=self.__get_sample_indices())
        return tree

    # Function to build all the trees of the forrest, differentiates between running in parallel and sequential
    def __build_trees(self):
        if(self.n_jobs == 1):
            for tree in self.trees:
                self._build_single_tree(tree)
        else:
            with Pool(self.n_jobs) as p:
                self.trees = p.map(self._build_single_tree, self.trees)
    
    # Function used to call the predict function of a tree
    def _predict_single_tree(self, tree:DecisionTree):
        return tree.predict(self.predict_values.read()).base

    # Function to call predict on all the trees of the forrest, differentiates between running in parallel and sequential
    def __predict_trees(self):
        predictions = []

        if(self.n_jobs == 1):
            for tree in self.trees:
                predictions.append(self._predict_single_tree(tree))
        else:
            with Pool(self.n_jobs) as p:
                predictions = p.map(self._predict_single_tree, self.trees)
                
        return np.column_stack(predictions)

    # Function to call predict_proba for a tree
    def _predict_proba_single_tree(self, tree:DecisionTree):
        return tree.predict_proba(self.predict_values.read())[1]

    # Function to call predict_proba on all the trees of the forrest, differentiates between running in parallel and sequential
    def __predict_proba_trees(self):
        predictions = []

        if(self.n_jobs == 1):
            for tree in self.trees:
                predictions.append(self._predict_proba_single_tree(tree))
        else:
            with Pool(self.n_jobs) as p:
                predictions = p.map(self._predict_proba_single_tree, self.trees)

        return predictions


    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Function used to fit a forrest using many DecisionTrees for the given data

        Parameters
        ----------
        X : np.ndarray
            feature values
        Y : np.ndarray
            response values
        """
        # Should raise error if bootstrap is false, but max_samples is set
        if not self.bootstrap and self.max_samples:
            raise AttributeError("Bootstrap can not be False while max_samples is set")
        
        X, Y = self.check_input(X, Y)
        self.features = SharedNumpyArray(X)
        self.outcomes = SharedNumpyArray(Y)
        self.n_obs, self.n_features = self.features._shape
        if self.forrest_type == "Classification":
            self.classes = np.unique(self.outcomes.read())
        if self.max_samples is None:
            self.max_samples = self.n_obs
        self.trees = [DecisionTree(tree_type=self.forrest_type, 
                                    criteria=self.criterion,
                                    max_depth = self.max_depth,
                                    impurity_tol = self.impurity_tol,
                                    min_samples_split = self.min_samples_split,
                                    min_samples_leaf = self.min_samples_leaf,
                                    min_improvement = self.min_improvement,
                                    splitter=self.splitter) for _ in range(self.n_estimators)]

        # Fit trees
        self.__build_trees()

        # Register that the forrest was succesfully fitted
        self.forest_fitted = True

        return self

    # Function that returns an ndarray of random ints used as the sample_indices
    def __get_sample_indices(self):
        if self.bootstrap:
            return np.random.randint(low=0, high=self.n_obs, size=self.max_samples)
        else:
            return None
    
    def predict(self, X: np.ndarray):
        """
        Predicts a y-value for given X values using the trees of the forrest.

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
            raise AttributeError("The forrest has not been fitted before trying to call predict")

        self.predict_values = SharedNumpyArray(X)

        # Predict using all the trees, each column is the predictions from one tree
        tree_predictions = self.__predict_trees()
        self.predict_values.unlink()

        if self.forrest_type == "Regression":
            # Return the mean answer from all trees for each row
            return np.mean(tree_predictions, axis=1)

        elif self.forrest_type == "Classification":
            # Return the element most voted for by trees for each row
            return np.apply_along_axis(self.most_frequent_element, 1, tree_predictions) # QUESTION: how to handle equal majority vote?

    # Function used to find the most frequent element of an array
    def most_frequent_element(self, arr):
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]

    def predict_proba(self, X: np.ndarray):
        """
        Predicts a probability for each response for given X values using the trees of the forrest

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
            raise AttributeError("The forrest has not been fitted before trying to call predict_proba")

        # Make sure that predict_proba is only called on Classification forrests
        if self.forrest_type != "Classification":
            raise ValueError("predict_proba can only be called on a Classification tree")

        # Check dimensions
        self.check_dimensions(X)

        self.predict_values = SharedNumpyArray(X)
        # Predict_proba using all the trees, each element of list is the predict_proba from one tree
        tree_predictions = self.__predict_proba_trees()
        self.predict_values.unlink()

        # Stack the predict_probas
        stacked_tree_predictions = np.stack(tree_predictions, axis=0)

        # Return the mean along the newly created axis
        return np.mean(stacked_tree_predictions, axis=0)

    def check_dimensions(self, X: np.ndarray):
        # If there is only a single point
        if X.ndim == 1:
            if (X.shape[0] != self.n_features):
                raise ValueError(f"Number of features should be {self.n_features}, got {X.shape[0]}")
        else:
            if X.shape[1] != self.n_features:
                raise ValueError(f"Dimension should be {self.n_features}, got {X.shape[1]}")

    def check_input(self, X: np.ndarray, Y: np.ndarray):
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