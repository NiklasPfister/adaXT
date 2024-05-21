import numpy as np
from ..criteria import Criteria


class Splitter:
    """
    Splitter class used to create splits of the data
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, criteria: type[Criteria]) -> None:
        """
        Parameters
        ----------
            x: memoryview of NDArray
                The feature values of the dataset
            y: memoryview of NDArray
                The response values of the dataset
            criteria: Criteria
                The criteria class used to find the impurity of a split
        """
        pass

    def get_split(self, indices: np.ndarray, feature_indices: np.ndarray):
        """
        Function that finds the best split of the dataset
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            Indices constituting the dataset

        Returns
        -----------
        (list, double, int, double, double)
            Returns the best split of the dataset, with the values being:
            (1) a list containing the left and right indices, (2) the best
            threshold for doing the splits, (3) what feature to split on,
            (4) the best criteria score, and (5) the best impurity
        """
        pass
