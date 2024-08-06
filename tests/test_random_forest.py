from adaXT.decision_tree import DecisionTree
from adaXT.criteria import (
    Gini_index,
    Squared_error,
    Entropy,
    Linear_regression,
)
from adaXT.predict import PredictLinearRegression
from adaXT.leaf_builder import LeafBuilderLinearRegression
from adaXT.random_forest import RandomForest
import numpy as np
import json
from multiprocessing import cpu_count
import sys

# We define the last feature of X to be equal to Y such that there is a perfect correlation. Thus when we train a Random Forest
# on this data, we should have predictions that are always equal to the
# last column of the input data.


def get_regression_data(
        n,
        m,
        random_state: np.random.RandomState,
        lowx=0,
        highx=100,
        lowy=0,
        highy=5):
    X = random_state.uniform(lowx, highx, (n, m))
    Y = random_state.uniform(lowy, highy, n)
    return (X, Y)


def get_classification_data(
        n,
        m,
        random_state: np.random.RandomState,
        lowx=0,
        highx=100,
        lowy=0,
        highy=5):
    X = random_state.uniform(lowx, highx, (n, m))
    Y = random_state.randint(lowy, highy, n)
    return (X, Y)


def run_gini_index(X, Y, n_jobs, n_estimators, seed):
    forest = RandomForest(
        forest_type="Classification",
        criteria=Gini_index,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        sampling="bootstrap",
        sampling_parameter=5,
        random_state=seed,
    )
    forest.fit(X, Y)
    return forest


def run_entropy(X, Y, n_jobs, n_estimators, seed):
    forest = RandomForest(
        forest_type="Classification",
        criteria=Entropy,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        sampling="bootstrap",
        sampling_parameter=5,
        random_state=seed,
    )
    forest.fit(X, Y)
    return forest


def run_squared_error(
        X,
        Y,
        n_jobs,
        n_estimators,
        seed,
        max_samples: int | float = 5,
        max_depth=sys.maxsize,
        sampling: str | None = "bootstrap"):
    forest = RandomForest(
        forest_type="Regression",
        criteria=Squared_error,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        sampling=sampling,
        sampling_parameter=max_samples,
        random_state=seed,
        max_depth=max_depth,
    )
    forest.fit(X, Y)
    return forest


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
        criteria=Gini_index,
        sampling="bootstrap"
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
    random_state = np.random.RandomState(100)
    tree_state = 100
    X, Y = get_regression_data(n, m, random_state=random_state)
    prediction_data = np.random.uniform(
        0, 10, (n, m))  # Get new data to predict
    forest1 = RandomForest(
        "Regression",
        n_estimators=100,
        criteria=Squared_error,
        random_state=tree_state,
        sampling="bootstrap",
    )
    forest1.fit(X, Y)

    forest2 = RandomForest(
        "Regression",
        n_estimators=100,
        criteria=Squared_error,
        random_state=tree_state,
        sampling="bootstrap",
    )
    forest2.fit(X, Y)

    pred1 = forest1.predict(prediction_data)
    pred2 = forest2.predict(prediction_data)

    assert np.array_equal(
        pred1, pred2
    ), "The two random forest predictions were different"


def test_deterministic_seeding_classification():
    n = 1000
    m = 10
    random_state = np.random.RandomState(100)
    tree_state = 100
    X, Y = get_classification_data(n, m, random_state=random_state)
    prediction_data = np.random.uniform(
        0, 10, (n, m))  # Get new data to predict
    forest1 = RandomForest(
        "Classification",
        n_estimators=100,
        criteria=Gini_index,
        random_state=tree_state,
        sampling="bootstrap"
    )
    forest1.fit(X, Y)

    forest2 = RandomForest(
        "Classification",
        n_estimators=100,
        criteria=Gini_index,
        random_state=tree_state,
        sampling="bootstrap",
    )
    forest2.fit(X, Y)

    pred1 = forest1.predict(prediction_data)
    pred2 = forest2.predict(prediction_data)

    assert np.array_equal(
        pred1, pred2
    ), "The two random forest predictions were different"


def test_random_forest():
    random_state = np.random.RandomState(2024)
    seed = 2024
    n = 100
    m = 10
    n_estimators = 100
    X_cla, Y_cla = get_classification_data(n, m, random_state=random_state)
    X_reg, Y_reg = get_regression_data(n, m, random_state=random_state)
    gini_forest = run_gini_index(
        X_cla, Y_cla, n_jobs=cpu_count(), n_estimators=n_estimators, seed=seed
    )
    entropy_forest = run_entropy(
        X_cla, Y_cla, n_jobs=cpu_count(), n_estimators=n_estimators, seed=seed
    )
    squared_forest = run_squared_error(
        X_reg, Y_reg, n_jobs=cpu_count(), n_estimators=n_estimators, seed=seed
    )
    pred = dict()
    pred["gini_pred"] = gini_forest.predict(X_cla)
    pred["entropy_pred"] = entropy_forest.predict(X_cla)
    pred["squared_pred"] = squared_forest.predict(X_reg)
    with open("./tests/data/forestData.json", "r") as f:
        data = json.loads(f.read())
    assert np.array_equal(
        np.array(data["gini_pred"]), pred["gini_pred"]
    ), "Gini Index prediction incorrect"

    assert np.array_equal(
        np.array(data["entropy_pred"]), pred["entropy_pred"]
    ), "Entropy prediction incorrect"

    assert np.array_equal(
        np.array(data["squared_pred"]), pred["squared_pred"]
    ), "Squared Error prediction incorrect"


def test_linear_regression_forest():
    random_state = np.random.RandomState(2024)
    n = 1000
    m = 10
    X_reg, Y_reg = get_regression_data(n, m, random_state=random_state)
    tree = DecisionTree(
        "Linear Regression",
        leaf_builder=LeafBuilderLinearRegression,
        predict=PredictLinearRegression,
        criteria=Linear_regression,
    )
    forest = RandomForest(
        "Linear Regression",
        leaf_builder=LeafBuilderLinearRegression,
        predict=PredictLinearRegression,
        criteria=Linear_regression,
        sampling=None,
    )
    tree.fit(X_reg, Y_reg)
    forest.fit(X_reg, Y_reg)
    tree_predict = tree.predict(X_reg)
    forest_predict = forest.predict(X_reg)
    assert np.allclose(
        tree_predict, forest_predict
    ), "Forest predicts different than tree when it should be equal."


def test_quantile_regression_forest():
    random_state = np.random.RandomState(2024)
    n = 10
    m = 10
    X_reg, Y_reg = get_regression_data(n, m, random_state=random_state)
    tree = DecisionTree(
        "Quantile",
    )
    forest = RandomForest("Quantile", sampling=None)
    tree.fit(X_reg, Y_reg)
    forest.fit(X_reg, Y_reg)
    tree_predict = tree.predict(X_reg, quantile=0.95)
    forest_predict = forest.predict(X_reg, quantile=0.95)
    assert np.allclose(
        tree_predict, forest_predict
    ), "Forest predicts different than tree when it should be equal."


def test_random_forest_weights():
    random_state = np.random.RandomState(2024)
    seed = 2024
    n = 100
    m = 10
    n_estimators = 5
    X_reg, Y_reg = get_regression_data(n, m, random_state=random_state)
    squared_forest = run_squared_error(
        X_reg,
        Y_reg,
        n_jobs=cpu_count(),
        n_estimators=n_estimators,
        seed=seed,
        max_depth=2,
        sampling=None,
    )
    res = squared_forest.predict_forest_weight(X=None, scale=False)
    trees = [DecisionTree("Regression", max_depth=2) for _ in range(100)]
    for item in trees:
        item.fit(X_reg, Y_reg)
    tree_sum = np.mean([tree.predict_leaf_matrix(
        X=None, scale=False) for tree in trees], axis=0)
    assert np.array_equal(tree_sum, res)


def __check_leaf_count(forest: RandomForest, expected_weight: float):
    for tree in forest.trees:
        tree_sum = np.sum([node.weighted_samples for node in tree.leaf_nodes])
        assert tree_sum == expected_weight, "The expected leaf node failed for\
        the given forest"


def test_honest_sampling_leaf_samples():
    random_state = np.random.RandomState(2024)
    n = 10
    m = 10
    n_fit = 5
    n_estimators = 5
    X_reg, Y_reg = get_regression_data(n, m, random_state=random_state)
    honest_tree = RandomForest(
        "Regression",
        n_estimators=n_estimators,
        sampling="honest_tree",
        sampling_parameter=n_fit,
        max_depth=4)
    honest_forest = RandomForest(
        "Regression",
        n_estimators=n_estimators,
        sampling="honest_forest",
        sampling_parameter=(
            n // 2,
            n_fit),
        max_depth=4)
    honest_tree.fit(X_reg, Y_reg)
    honest_forest.fit(X_reg, Y_reg)
    __check_leaf_count(honest_tree, n_fit)
    __check_leaf_count(honest_forest, n_fit)


def test_n_jobs():
    random_state = np.random.RandomState(2024)
    n = 1000
    m = 10
    X_reg, Y_reg = get_regression_data(n, m, random_state=random_state)
    forest_1 = run_squared_error(
        X_reg, Y_reg, n_jobs=1, n_estimators=100, seed=2024)
    forest_5 = run_squared_error(
        X_reg, Y_reg, n_jobs=5, n_estimators=100, seed=2024)
    pred_1 = forest_1.predict(X_reg)
    pred_2 = forest_5.predict(X_reg)
    assert np.allclose(pred_1, pred_2)


def test_n_jobs_predict_forest():
    random_state = np.random.RandomState(2024)
    seed = 2024
    n = 100
    m = 10
    n_estimators = 5
    X_reg, Y_reg = get_regression_data(n, m, random_state=random_state)
    squared_forest = run_squared_error(
        X_reg,
        Y_reg,
        n_jobs=1,
        n_estimators=n_estimators,
        seed=seed,
        max_depth=2,
        sampling=None,
    )
    res = squared_forest.predict_forest_weight(X=X_reg, scale=False)
    trees = [DecisionTree("Regression", max_depth=2) for _ in range(100)]
    for item in trees:
        item.fit(X_reg, Y_reg)
    tree_sum = np.mean([tree.predict_leaf_matrix(
        X=X_reg, scale=False) for tree in trees], axis=0)
    assert np.array_equal(tree_sum, res)


if __name__ == "__main__":
    # test_dominant_feature()
    # test_deterministic_seeding_classification()
    # test_linear_regression_forest()
    # test_quantile_regression_forest()
    # test_random_forest_weights()
    # test_honest_sampling_leaf_samples()
    test_n_jobs_predict_forest()

    print("Done")
