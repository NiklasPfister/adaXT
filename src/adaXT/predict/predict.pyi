import numpy as np


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

    @staticmethod
    def forest_predict(predictions: np.ndarray, **kwargs):
        """Function called by the RandomForest class. Should not need
        customisation."""
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


class PredictRegression(Predict):
    """
    Default used for Regression
    """

    pass


class PredictLinearRegression(Predict):
    pass


class PredictQuantile(Predict):
    pass
