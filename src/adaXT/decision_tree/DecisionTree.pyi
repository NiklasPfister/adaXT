import numpy as np
from .splitter import Splitter
from .criteria import Criteria
from .Nodes import Node
import sys

class DecisionTree:
    """
    DecisionTree object
    """

    def __init__(
            self,
            tree_type: str,
            criteria: Criteria,
            max_depth: int = sys.maxsize,
            impurity_tol: float = 1e-20,
            min_samples: int = 1,
            min_improvement: float = 0, 
            pre_sort: None | np.ndarray = None) -> None:
        """
        Parameters
        ----------
        tree_type : str
            Classification or Regression
        max_depth : int
            maximum depth of the tree, by default int(np.inf)
        impurity_tol : float
            the tolerance of impurity in a leaf node, by default 1e-20
        min_samples : int
            the minimum amount of samples in a leaf node, by deafult 2
        min_improvement: float
            the minimum improvement gained from performing a split, by default 0
        pre_sort: np.ndarray | None
            a sorted index matrix for the dataset
        """
        pass

    def fit(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            splitter: Splitter | None = None,
            feature_indices: np.ndarray | None = None,
            sample_indices: np.ndarray | None = None) -> None:
        """
        Function used to fit the data on the tree using the DepthTreeBuilder

        Parameters
        ----------
        X : np.ndarray
            feature values
        Y : np.ndarray
            outcome values
        criteria : FuncWrapper
            Callable criteria function used to calculate impurity wrapped in Funcwrapper class.
        splitter : Splitter | None, optional
            Splitter class if None uses premade Splitter class
        feature_indices : np.ndarray | None, optional
            which features to use from the data X, by default uses all
        sample_indices : np.ndarray | None, optional
            which samples to use from the data X and Y, by default uses all
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts a y-value for given X values

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
        Predicts a probability for each response for given X values

        Parameters
        ----------
        X : np.ndarray
            (N, M) numpy array with features to predict

        Returns
        -------
        Tuple(np.ndarray, np.ndarray)
            Returns a tuple where the first element are the reponsense, and the othe element are the probability for each class per observation in X.
        """
        pass

    def get_leaf_matrix(self) -> np.ndarray:
        """
        Creates NxN matrix,
        where N is the number of observations.
        If a given value is 1, then they are in the same leaf,
        otherwise it is 0

        Returns
        -------
        np.ndarray
            NxN matrix
        """
        pass

    def predict_leaf_matrix(self, X: np.ndarray, scale: bool = False):

        pass