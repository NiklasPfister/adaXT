import numpy as np
from ..decision_tree.nodes import DecisionNode

class Predict:
    """
    The Predict class which the DecisionTree depends on.
    Other implementations must inherit from this class.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, root: DecisionNode) -> None:
        pass

    def predict(self, X: np.ndarray, **kwargs):
        """
        Prediction function called by the DecisionTree when it is told to
        predict. Can be customised for a different output of the DecisionTree.

        Parameters
        ----------
        X: np.ndarray
            Array that should be carried out predictions on.
        """
        pass

    def predict_proba(self, X: np.ndarray):
        """
        Predict proba function call.

        Parameters
        ----------
        X : np.ndarray
            Array that should be carried out predictions on.
        """
        pass

    def predict_leaf_matrix(self, X: np.ndarray, scale: bool):
        """
        Function called when tree.predict_leaf_matrix is called.

        Parameters
        ----------
        X : np.ndarray
            Data to predict on.
        scale
            Whether to scale the data.
        """
        pass

    @staticmethod
    def forest_predict(predictions: np.ndarray, **kwargs):
        """Function called by the RandomForest class after it has gotten the
        result of predict from each tree. Thise function should then do any
        changes seen fit.

        Parameters
        ----------
        predictions: np.ndarray
            An array of predictions.
        """
        pass

    @staticmethod
    def forest_predict_proba(predictions: np.ndarray, **kwargs):
        """
        Function called by the RandomForest after it has gotten the result of
        predict_proba from each tree. This function should then do any changes
        seen fit.

        Parameters
        ----------
        predictions : np.ndarray
            An array of predictions.
        """
        pass

class PredictClassification(Predict):
    """
    The default prediction class for the Classification DecisionTree.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, root: DecisionNode) -> None:
        pass

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predicts the LeafNode every $x \in X$ would fall into and returns the
        class which occurs the most amount of times within the LeafNode

        Parameters
        ----------
        X: np.ndarray
            Array that should be carried out predictions on.
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the LeafNode every $x \in X$ would fall into and returns the
        fraction of occurences each unique class has within the LeafNode.

        Parameters
        ----------
        X : np.ndarray
            Array that should be carried out predictions on.
        """
        pass

    @staticmethod
    def forest_predict(predictions: np.ndarray, **kwargs) -> np.ndarray:
        """
        With the predictions it returns the most frequent element picked by all
        the individual trees. Thus, this is the majority vote.

        Parameters
        ----------
        predictions
            The predictions given by every tree of the forest.
        """

        pass

    @staticmethod
    def forest_predict_proba(predictions: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculates the mean predictions for all the predictions.

        Parameters
        ----------
        predictions : np.ndarray
            An array of predictions.
        """
        pass

class PredictRegression(Predict):
    """
    Default used for Regression.
    """

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predicts which LeafNode each $x \in X$ would land in, and calculates the
        mean value of all elements in the LeafNode.

        Parameters
        ----------
        X: np.ndarray
            Array that should be carried out predictions on.

        Returns
        -------
        np.ndarray
            An array with predictions for each row of X.
        """
        pass

    pass

    @staticmethod
    def forest_predict(predictions: np.ndarray, **kwargs) -> np.ndarray:
        """
        Returns the mean value of all the predictions along axis 1.

        Parameters
        ----------
        predictions
            Predictions by the trees.

        Returns
        -------
        np.ndarray
            An array of predictions for each element X passed to
            forest.predict.
        """
        pass

class PredictLinearRegression(Predict):
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predicts which LeafNode each $x \in X$ would land in, and calculates the
        $$
        Y_i = L_{j,theta0} + L_{j,theta1}*X_{i, 0}
        $$
        For each $i \in X$ where $L_j$ is the LeafNode x would fall into, and
        theta0 and theta1 are calculated for each LeafNode during training.

        Parameters
        ----------
        X: np.ndarray
            Array that should be carried out predictions on.

        Returns
        -------
        np.ndarray
            An array with predictions for each row of X.
        """
        pass

class PredictLocalPolynomial(Predict):
    """
    Default used for Gradient
    """

    pass

class PredictQuantile(Predict):
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predicts the quantile for each $x \in X$.

        Parameters
        ----------
        X : np.ndarray
            Array that should be carried out predictions on.
        **kwargs
            save_indices : bool
                Whether to save the indices or the quantile.
            quantile : str
                The quantile given to numpys quantile function.

        Returns
        -------
        np.ndarray
            An array with predicstion for each row of X.
        """
        pass

    @staticmethod
    def forest_predict(predictions: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calls quantile on the predictions

        Parameters
        ----------
        predictions
            The indices returned from predict with save_indices=True.

        Returns
        -------
        np.ndarray
            The quantile predictions for each $x \in X$.
        """

        pass
