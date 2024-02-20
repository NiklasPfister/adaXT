from adaXT.decision_tree import DecisionTree, LeafNode, DecisionNode
from adaXT.criteria import Gini_index, Squared_error, Entropy, Linear_regression
from adaXT.random_forest import RandomForest
import numpy as np

# We define the last feature of X to be equal to Y such that there is a perfect correlation. Thus when we train a Random Forrest
# on this data, we should have predictions that are always equal to the
# last column of the input data.


def test_dominant_feature():
    # Test data dimensions
    n = 1000
    m = 24

    # Create test data
    X = np.random.uniform(0, 10, (n, m))
    Y = np.random.randint(0, 10, n)
    X = np.column_stack((X, Y))

    # Create forest and fit data
    forest = RandomForest(
        "Classification",
        n_estimators=100,
        criterion=Gini_index,
        bootstrap=False)
    forest.fit(X, Y)

    # Create data for predict
    prediction_data = np.random.randint(0, 10, (n, m + 1))

    # Do prediction
    pred = forest.predict(prediction_data)

    # Assert
    for i, el in enumerate(pred):
        assert (
            el == prediction_data[i, -1]), f"The data for prediction should be equal to the data in the last row of prediction_data, as it is a dominant feature but was {el} and {prediction_data[i, -1]}"


if __name__ == "__main__":
    test_dominant_feature()
