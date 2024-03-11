class Predict:
    """
    The Predict class which the DecisionTree depends on.
    Other implementations must inherit from this class.
    """

    pass

class PredictClassification(Predict):
    """
    Default used for Classification
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
