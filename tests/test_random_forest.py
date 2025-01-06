from adaXT.decision_tree import DecisionTree
from adaXT.criteria import (
    Gini_index,
    Squared_error,
    Entropy,
    Partial_quadratic,
)
from adaXT.predictor import PredictorLocalPolynomial
from adaXT.leaf_builder import LeafBuilderPartialQuadratic
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


def run_gini_index(
    X,
    Y,
    n_jobs,
    n_estimators,
    seed,
    max_samples: int | float = 5,
    max_depth=sys.maxsize,
    sampling: str | None = "resampling",
    oob: bool = False,
):
    forest = RandomForest(
        forest_type="Classification",
        criteria=Gini_index,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        sampling=sampling,
        sampling_args={"size": max_samples, "OOB": oob},
        seed=seed,
        max_depth=max_depth,
    )
    forest.fit(X, Y)
    return forest


def run_entropy(
    X,
    Y,
    n_jobs,
    n_estimators,
    seed,
    max_samples: int | float = 5,
    max_depth=sys.maxsize,
    sampling: str | None = "resampling",
    oob: bool = False,
):
    forest = RandomForest(
        forest_type="Classification",
        criteria=Entropy,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        sampling=sampling,
        sampling_args={"size": max_samples, "OOB": oob},
        seed=seed,
        max_depth=max_depth,
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
    sampling: str | None = "resampling",
    oob: bool = False,
):
    forest = RandomForest(
        forest_type="Regression",
        criteria=Squared_error,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        sampling=sampling,
        sampling_args={"size": max_samples, "OOB": oob},
        seed=seed,
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
        sampling="resampling",
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
        seed=tree_state,
        sampling="resampling",
    )
    forest1.fit(X, Y)

    forest2 = RandomForest(
        "Regression",
        n_estimators=100,
        criteria=Squared_error,
        seed=tree_state,
        sampling="resampling",
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
        seed=tree_state,
        sampling="resampling",
    )
    forest1.fit(X, Y)

    forest2 = RandomForest(
        "Classification",
        n_estimators=100,
        criteria=Gini_index,
        seed=tree_state,
        sampling="resampling",
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
        np.array(data["gini_pred"]).astype(int), pred["gini_pred"]
    ), "Gini Index prediction incorrect"

    assert np.array_equal(
        np.array(data["entropy_pred"]).astype(int), pred["entropy_pred"]
    ), "Entropy prediction incorrect"

    assert np.allclose(
        np.array(data["squared_pred"]), pred["squared_pred"]
    ), "Squared Error prediction incorrect"


def create_and_save_predict_data(seed=2024):
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
    pred["gini_pred"] = gini_forest.predict(X_cla).tolist()
    pred["entropy_pred"] = entropy_forest.predict(X_cla).tolist()
    pred["squared_pred"] = squared_forest.predict(X_reg).tolist()

    with open("./tests/data/forestData.json", "w") as f:
        json.dump(pred, f)


def test_gradient_forest():
    random_state = np.random.RandomState(2024)
    n = 1000
    m = 10
    X_reg, Y_reg = get_regression_data(n, m, random_state=random_state)
    tree = DecisionTree(
        "Gradient",
        leaf_builder=LeafBuilderPartialQuadratic,
        predictor=PredictorLocalPolynomial,
        criteria=Partial_quadratic,
    )
    forest = RandomForest(
        "Gradient",
        leaf_builder=LeafBuilderPartialQuadratic,
        predictor=PredictorLocalPolynomial,
        criteria=Partial_quadratic,
        sampling=None,
    )
    tree.fit(X_reg, Y_reg)
    forest.fit(X_reg, Y_reg)
    tree_predict = tree.predict(X_reg)
    print("")
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
    n_estimators = 100
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
    res = squared_forest.predict_weights(X=None, scale=False)
    trees = [DecisionTree("Regression", max_depth=2)
             for _ in range(n_estimators)]
    for item in trees:
        item.fit(X_reg, Y_reg)
    tree_sum = np.sum(
        [tree.predict_weights(X=None, scale=False) for tree in trees], axis=0
    )
    print(tree_sum)
    print(res)
    assert np.array_equal(tree_sum, res)


def test_tree_based_weights():
    random_state = np.random.RandomState(2024)
    seed = 2024
    n = 105
    m = 3
    n_estimators = 100
    Xfull, Yfull = get_regression_data(n, m, random_state=random_state)
    Xtrain = Xfull[:100, :]
    Ytrain = Yfull[:100]
    Xtest = Xfull[100:, :]
    rf_boot = RandomForest(
        "Regression",
        n_estimators=n_estimators,
        seed=seed,
        sampling="resampling",
    )
    rf_boot.fit(Xtrain, Ytrain)
    rf_honest_tree = RandomForest(
        "Regression",
        n_estimators=n_estimators,
        seed=seed,
        sampling="honest_tree",
    )
    rf_honest_tree.fit(Xtrain, Ytrain)
    rf_honest_forest = RandomForest(
        "Regression",
        n_estimators=n_estimators,
        seed=seed,
        sampling="honest_forest",
    )
    rf_honest_forest.fit(Xtrain, Ytrain)
    weights_boot = rf_boot.predict_weights(Xtest)
    weights_honest_tree = rf_honest_tree.predict_weights(Xtest)
    weights_honest_forest = rf_honest_forest.predict_weights(Xtest)
    # Check shapes
    assert np.array_equal(
        weights_boot.shape, [
            Xtest.shape[0], Xtrain.shape[0]])
    assert np.array_equal(
        weights_honest_tree.shape, [
            Xtest.shape[0], Xtrain.shape[0]])
    assert np.array_equal(
        weights_honest_forest.shape, [Xtest.shape[0], Xtrain.shape[0]]
    )
    # Check scaling
    assert np.sum(weights_boot.sum(axis=1)) == Xtest.shape[0]
    assert np.sum(weights_honest_tree.sum(axis=1)) == Xtest.shape[0]
    assert np.sum(weights_honest_forest.sum(axis=1)) == Xtest.shape[0]
    # Check predictions based on weights match regular predictions
    assert np.allclose(rf_boot.predict(Xtest), weights_boot.dot(Ytrain))
    assert np.allclose(
        rf_honest_tree.predict(Xtest),
        weights_honest_tree.dot(Ytrain))
    assert np.allclose(
        rf_honest_forest.predict(Xtest), weights_honest_forest.dot(Ytrain)
    )


def _check_leaf_count(forest: RandomForest, expected_weight: float):
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
        sampling_args={"split": n_fit, "size": n, "replace": False},
        max_depth=4,
    )
    honest_forest = RandomForest(
        "Regression",
        n_estimators=n_estimators,
        sampling="honest_forest",
        sampling_args={"split": n_fit, "size": n // 2, "replace": True},
        max_depth=4,
    )
    honest_tree.fit(X_reg, Y_reg)
    honest_forest.fit(X_reg, Y_reg)
    _check_leaf_count(honest_tree, n_fit)
    _check_leaf_count(honest_forest, n_fit)


def test_n_jobs():
    random_state = np.random.RandomState(2024)
    n = 1000
    m = 10
    X_reg, Y_reg = get_regression_data(n, m, random_state=random_state)
    forest_1 = run_squared_error(
        X_reg,
        Y_reg,
        n_jobs=1,
        n_estimators=100,
        seed=2024)
    forest_5 = run_squared_error(
        X_reg,
        Y_reg,
        n_jobs=5,
        n_estimators=100,
        seed=2024)
    pred_1 = forest_1.predict(X_reg)
    pred_2 = forest_5.predict(X_reg)
    assert np.allclose(pred_1, pred_2)


def test_n_jobs_predict_forest():
    random_state = np.random.RandomState(2024)
    seed = 2024
    n = 5
    m = 5
    n_estimators = 100
    X_reg, Y_reg = get_regression_data(n, m, random_state=random_state)
    squared_forest = run_squared_error(
        X_reg,
        Y_reg,
        n_jobs=2,
        n_estimators=n_estimators,
        seed=seed,
        max_depth=2,
        sampling=None,
    )
    res = squared_forest.predict_weights(X=X_reg, scale=False)
    trees = [DecisionTree("Regression", max_depth=2)
             for _ in range(n_estimators)]
    for item in trees:
        item.fit(X_reg, Y_reg)
    tree_sum = np.sum(
        [tree.predict_weights(X=X_reg, scale=False) for tree in trees], axis=0
    )
    assert np.array_equal(tree_sum, res)


def test_similarity():
    X = np.array([-1, -1, -1, 1, 1, 1])
    X0 = np.array([-1, -0.5, -0.1, 0.1, 0.5, 1])
    X1 = np.array([-0.75, 0.75])
    Y = np.array([1.5, 1.5, 1, 0, 0, 0.5])
    dt = DecisionTree("Regression")
    rf = RandomForest("Regression")
    dt.fit(X, Y)
    rf.fit(X, Y)
    sim_dt = dt.similarity(X0, X1)
    sim_rf = rf.similarity(X0, X1)
    expected_sim = np.array(
        [
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
        ]
    )
    # Check that tree similarity matches exactly
    assert np.array_equal(sim_dt, expected_sim)
    # Check that forest similarity has correct shape and satisfies bounds
    assert np.array_equal(sim_rf.shape, expected_sim.shape)
    assert np.sum(sim_rf <= 1) == 12 and np.sum(sim_rf >= 0) == 12


def check_OOB(X, Y, forest):
    assert hasattr(forest, "oob")

    for i in range(forest.n_estimators):
        if forest.prediction_indices[i] is None:
            picked_indices = forest.fitting_indices[i]
        else:
            picked_indices = np.concatenate(
                (forest.fitting_indices[i], forest.prediction_indices[i])
            )
        out_of_bag = np.setdiff1d(
            np.arange(
                0,
                forest.X.shape[0]),
            picked_indices)
        assert np.array_equal(out_of_bag, forest.out_of_bag_indices[i])


def test_OOB_squared_error():
    seed = 2024
    n = 10000
    m = 5
    n_estimators = 100
    variance = 1
    half = n // 2
    X = np.zeros((n, m))
    # create a perfect split for half the dataset
    X[half:, 1] = 1
    # Create random distribution for the two different LeafNodes
    # Both with standard deviation 1.
    Y = np.random.normal(0, np.sqrt(variance), half)
    Y = np.concatenate((Y, np.random.normal(5, np.sqrt(variance), size=half)))
    squared_forest = run_squared_error(
        X,
        Y,
        n_jobs=cpu_count(),
        n_estimators=n_estimators,
        seed=seed,
        max_depth=2,
        max_samples=n,
        oob=True,
    )
    check_OOB(X, Y, squared_forest)

    # Check that out of bag error is close to variance
    assert np.isclose(
        variance, squared_forest.oob, atol=0.1
    ), f"Squared error OOB is {squared_forest.oob}, should be closer to {variance}"


def test_OOB_entropy():
    seed = 2024
    n = 10000
    m = 5
    n_estimators = 100
    half = n // 2
    X = np.zeros((n, m))
    # create a perfect split for half the dataset
    X[half:, 1] = 1
    # Create random distribution for the two different LeafNodes
    # Both with standard deviation 1.
    Y = np.ones(half)
    # Change 0.05 procent of the values in this half of Y to some other value
    inds = np.random.choice(Y.shape[0], size=int(half * 0.05))
    Y[inds] = -1

    temp = np.full(half, 5)
    # Change 0.05 procent of the values in temp to some other value
    inds = np.random.choice(temp.shape[0], size=int(half * 0.05))
    temp[inds] = 10

    Y = np.concatenate((Y, temp))
    forest = run_entropy(
        X,
        Y,
        n_jobs=cpu_count(),
        n_estimators=n_estimators,
        seed=seed,
        max_depth=2,
        max_samples=n,
        oob=True,
    )
    check_OOB(X, Y, forest)

    # Check that out of bag error is close to variance
    assert np.isclose(
        0.05, forest.oob, atol=0.01
    ), f"Entropy OOB is {forest.oob} should be closer to 0.05"


if __name__ == "__main__":
    test_dominant_feature()
    # test_deterministic_seeding_classification()
    # test_quantile_regression_forest()
    # test_random_forest_weights()
    # test_honest_sampling_leaf_samples()
    # test_n_jobs_predict_forest()
    # test_random_forest()
    # test_gradient_forest()
    # test_OOB_squared_error()
    # test_OOB_entropy()
    test_tree_based_weights()
    test_honest_sampling_leaf_samples()
    print("Done")
