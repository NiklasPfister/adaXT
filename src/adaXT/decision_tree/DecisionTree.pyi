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
            root: Node | None = None,
            n_nodes: int = -1,
            n_features: int = -1,
            n_classes: int = -1,
            n_obs: int = -1,
            leaf_nodes: list[Node] | None = None,
            pre_sort: None | np.ndarray = None,
            classes: np.ndarray | None = None) -> None:
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
        root : Node | None
            root node, by default None, added after fitting
        n_nodes : int | None
            number of nodes in the tree, by default -1, added after fitting
        n_features : int | None
            number of features in the dataset, by default -1, added after fitting
        n_classes : int | None
            number of classes in the dataset, by default -1, added after fitting
        n_obs : int | None
            number of observations in the dataset, by default -1, added after fitting
        leaf_nodes : list[Node] | None
            number of leaf nodes in the tree, by default None, added after fitting
        pre_sort: np.ndarray | None
            a sorted index matrix for the dataset
        classes : np.ndarray | None
            the different classes in response, by default None, added after fitting
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

    def predict_get_probability(self, X: np.ndarray):
        """
        Predicts a probability for each response for given X values

        Parameters
        ----------
        X : np.ndarray
            (N, M) numpy array with features to predict

        Returns
        -------
        List[Dict]
            Returns a list of dict with the lenght N. The keys are the response classes, and the values are the probability for this class.  
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