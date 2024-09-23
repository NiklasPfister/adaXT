import numpy as np
from ..decision_tree.nodes import DecisionNode
from ..decision_tree.decision_tree import DecisionTree
from ..base_model import ParallelModel

class Predict:
    """
    The Predict class which the DecisionTree depends on.
    Other implementations must inherit from this class.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, root: DecisionNode) -> None:
        pass

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Prediction function called by the DecisionTree when it is told to
        predict. Can be customised for a different output of the DecisionTree.

        Parameters
        ----------
        X: np.ndarray
            A 2-dimensional array for which the rows are the samples at which to predict.

        Returns
        -------
        np.ndarray
            An array with predictions for each row of X.
        """
        pass

    def predict_leaf(self, X: np.ndarray) -> dict:
        """
        Computes hash table indexing in which LeafNodes the rows of X fall into.

        Parameters
        ----------
        X : np.ndarray
            2-dimensional array for which the rows are the samples at which to predict.

        Returns
        -------
        dict
            A hash table with keys corresponding to LeafNode ids and values lists of
            indices specifying which rows land in a given LeafNode.
        """
        pass

    @staticmethod
    def forest_predict(
            X_old: np.ndarray,
            Y_old: np.ndarray,
            X_new: np.ndarray,
            trees: list[DecisionTree],
            parallel: ParallelModel,
            **kwargs) -> np.ndarray:
        """
        Internal function used by the RandomForest class 'predict' method to evaluate predictions
        for each tree and aggregate them.

        Needs to be adjusted whenever RandomForest predictions do not simply aggregate the tree
        predictions by averaging.
        
        Parameters
        ----------
        X_old: np.ndarray
            Array of feature values used during training.
        Y_old: np.ndarray
            Array of response values used during training.
        X_new: np.ndarray
            Array of new feature values at which to predict.
        tree: list[DecisionTree]
            List of fitted DecisionTrees fitted within the random forest.
        parallel: ParallelModel
            ParallelModel used for multiprocessing.

        Returns
        -------
        np.ndarray
            An array with predictions for each row of X_new.
        """
        pass

class PredictClassification(Predict):
    """
    The default prediction class for the 'Classification' tree type.
    """

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        For each row in X, the method first predicts the LeafNode into which
        the row falls and then computes the class with the highest number of
        occurances in that LeafNode.

        If the keyword argument 'predict_proba' is set to True. This method outputs
        class probabilities (i.e., the frequence of each label in the same LeafNode).

        Parameters
        ----------
        X: np.ndarray
            A 2-dimensional array for which the rows are the samples at which to predict.
        **kwargs
            predict_proba : bool
                Specifies whether to compute class probabilities or not.
                Defaults to False if not provided.


        Returns
        -------
        np.ndarray
            An array with the predicted class label for each row of X. Or, if 'predict_proba' is set
            to True, a 2-dimensional array with where each row provides an array of the occurance
            frequency of each class label appearing in the same leaf as the corresponding row in X.
        """
        pass

class PredictRegression(Predict):
    """
    The default prediction class for the 'Regression' tree type.
    """

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        For each row in X, the method first predicts the LeafNode into which
        the row falls and then computes average response value in that LeafNode.

        Parameters
        ----------
        X: np.ndarray
            A 2-dimensional array for which the rows are the samples at which to predict.

        Returns
        -------
        np.ndarray
            An array with the predicted values for each row of X.
        """
        pass

class PredictLocalPolynomial(Predict):
    """
    The default prediction class for the 'Gradient' tree type.
    """

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        For each row in X, the method first predicts the LeafNode into which
        the row falls and then computes uses the parameters theta0, theta1 and theta2
        saved on the LeafNode to estimate the predicted value
        $$
        \texttt{theta0} + \texttt{theta1} \texttt{X}[i, 0] + \texttt{theta2} \texttt{X}[i, 0]^2.
        $$

        Note: This predict class requires that the decision tree/random forest
        uses a LocalPolynomialLeafNode and either LeafBuilderPartialLinear or LeafBuilderPartialQuadratic.

        Parameters
        ----------
        X: np.ndarray
            A 2-dimensional array for which the rows are the samples at which to predict.

        Returns
        -------
        np.ndarray
            An array with the predicted values for each row of X.
        """
        pass

class PredictQuantile(Predict):
    """
    The default prediction class for the 'Quantile' tree type.
    """

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        For each row in X, the method first predicts the LeafNode into which
        the row falls and then computes the quantiles of the Y values in that LeafNode.

        For random forests the quantiles across all trees are computed jointly.

        Parameters
        ----------
        X: np.ndarray
            A 2-dimensional array for which the rows are the samples at which to predict.
        **kwargs
            quantile : list[float]
                A list of quantiles that are internally passed to the numpy 'quantile' function.

        Returns
        -------
        np.ndarray
            An array where each row correpsonds to the predicted quantiles for the corresponding
            row of X.
        """
        pass

