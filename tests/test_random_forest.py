from adaXT.decision_tree import DecisionTree, LeafNode, DecisionNode
from adaXT.criteria import (
    Gini_index,
    Squared_error,
    Entropy,
    Linear_regression,
    criteria,
)
from adaXT.random_forest import RandomForest
import numpy as np

# We define the last feature of X to be equal to Y such that there is a perfect correlation. Thus when we train a Random Forest
# on this data, we should have predictions that are always equal to the
# last column of the input data.


def get_random_data_regression(n, m, lowx=0, highx=1000, lowy=0, highy=5):
    return (np.random.uniform(lowx, highx, (n, m)), np.random.uniform(lowy, highy, n))


def get_random_data_classification(n, m, lowx=0, highx=1000, lowy=0, highy=5):
    return (np.random.uniform(lowx, highx, (n, m)), np.random.randint(lowy, highy, n))


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
        "Classification", n_estimators=100, criterion=Gini_index, bootstrap=False
    )
    forest.fit(X, Y)

    # Create data for predict
    prediction_data = np.random.randint(0, 10, (n, m + 1))

    # Do prediction
    pred = forest.predict(prediction_data)

    # Assert
    for i, el in enumerate(pred):
        assert (
            el == prediction_data[i, -1]
        ), f"The data for prediction should be equal to the data in the last row of prediction_data, as it is a dominant feature but was {el} and {prediction_data[i, -1]}"


def test_deterministic_seeding_regression():
    n = 1000
    m = 10
    random_state = 100
    X, Y = get_random_data_regression(n, m)
    prediction_data = np.random.uniform(0, 10, (n, m))  # Get new data to predict
    forest1 = RandomForest(
        "Regression",
        n_estimators=100,
        criterion=Squared_error,
        random_state=random_state,
    )
    forest1.fit(X, Y)

    forest2 = RandomForest(
        "Regression",
        n_estimators=100,
        criterion=Squared_error,
        random_state=random_state,
    )
    forest2.fit(X, Y)

    pred1 = forest1.predict(prediction_data)
    pred2 = forest2.predict(prediction_data)

    assert np.array_equal(
        pred1, pred2
    ), "The two random forest predictions were different"


if __name__ == "__main__":
    # test_dominant_feature()
    test_deterministic_seeding_regression()
    print("Done")
