from adaXT.decision_tree import DecisionTree
from adaXT.criteria import Gini_index, Squared_error, Entropy, Linear_regression
from adaXT.decision_tree.nodes import LeafNode, DecisionNode
from adaXT.predict import PredictLinearRegression, PredictQuantile
from adaXT.leaf_builder import (
    LeafBuilderLinearRegression,
    LeafBuilderRegression,
)

import numpy as np


def uniform_x_y(n, m):
    np.random.seed(2024)
    return (
        np.random.uniform(
            1, 1000, (n, m)), np.random.uniform(
            1, 1000, (n)))


def test_predict_leaf_matrix_classification():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])

    tree = DecisionTree("Classification", criteria=Gini_index)
    tree.fit(X, Y_cla)
    res1 = tree.predict_leaf_matrix(X=None)
    res2 = tree.predict_leaf_matrix(X)
    assert res1.shape == res2.shape
    row, col = res1.shape
    for i in range(row):
        for j in range(col):
            assert res1[i, j] == res2[i, j]


def test_predict_leaf_matrix_regression():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_reg = np.array([2.2, -0.5, 0.5, -0.5, 2, -3, 2.2, -3])
    tree = DecisionTree("Regression", criteria=Squared_error)
    tree.fit(X, Y_reg)

    res1 = tree.predict_leaf_matrix(X=None)
    res2 = tree.predict_leaf_matrix(X)
    assert res1.shape == res2.shape
    row, col = res1.shape
    for i in range(row):
        for j in range(col):
            assert res1[i, j] == res2[i, j]


def test_predict_leaf_matrix_regression_with_scaling():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_reg = np.array([2.2, -0.5, 0.5, -0.5, 2, -3, 2.2, -3])
    tree = DecisionTree("Regression", criteria=Squared_error)
    tree.fit(X, Y_reg)

    res1 = tree.predict_leaf_matrix(X=None)
    for i in range(res1.shape[0]):
        res1[i] = res1[i] / np.sum(res1[i])
    res2 = tree.predict_leaf_matrix(X, scale=True)
    assert res1.shape == res2.shape
    row, col = res1.shape
    for i in range(row):
        for j in range(col):
            assert res1[i, j] == res2[i, j]


def test_prediction():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    tree = DecisionTree("Classification", criteria=Gini_index)
    tree.fit(X, Y_cla)
    prediction = tree.predict(X)
    for i in range(len(Y_cla)):
        assert (
            Y_cla[i] == prediction[i]
        ), f"incorrect prediction at {i}, expected {Y_cla[i]} got {prediction[i]}"


def test_predict_proba_probability():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    tree = DecisionTree("Classification", criteria=Gini_index)
    tree.fit(X, Y_cla)
    classes = np.unique(Y_cla)
    prediction = tree.predict_proba(X)
    assert prediction.shape[0] == X.shape[0]
    for i in range(len(Y_cla)):
        assert (
            Y_cla[i] == classes[np.argmax(prediction[i, :])]
        ), f"incorrect prediction at {i}, expected {Y_cla[i]} got {classes[np.argmax(prediction[i, :])]}"


def test_predict_proba_against_predict():
    X = np.random.uniform(0, 100, (10000, 5))
    Y = np.random.randint(0, 5, 10000)

    tree = DecisionTree("Classification", criteria=Gini_index)
    tree.fit(X, Y)

    predict = tree.predict(X)
    classes = np.unique(Y)
    predict_proba = tree.predict_proba(X)

    for i in range(predict.shape[0]):
        assert (
            predict[i] == classes[np.argmax(predict_proba[i, :])]
        ), f"incorrect prediction at {i}, expected {predict[i]} got {classes[np.argmax(predict_proba[i, :])]}"


def test_NxN_matrix():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    tree = DecisionTree("Classification", criteria=Gini_index)
    tree.fit(X, Y_cla)
    leaf_matrix = tree.predict_leaf_matrix(X=None)
    true_weight = np.array(
        [
            [1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 1],
        ]
    )
    for i in range(len(true_weight)):
        for j in range(len(true_weight[0])):
            assert (
                leaf_matrix[i, j] == true_weight[i, j]
            ), f"Failed on ({i}, {j}), should be {true_weight[i, j]} was {leaf_matrix[i, j]}"


def test_max_depth_setting():
    np.random.seed(2023)  # Set seed such that each run is the same
    X = np.random.uniform(0, 100, (10000, 5))
    Y = np.random.randint(0, 5, 10000)
    max_depth_desired = 20

    tree = DecisionTree(
        "Classification", criteria=Gini_index, max_depth=max_depth_desired
    )
    tree.fit(X, Y)

    for node in tree.leaf_nodes:
        assert (
            node.depth <= max_depth_desired
        ), f"Failed as node depth was,{node.depth} but should be at the most {max_depth_desired}"


def test_impurity_tol_setting():
    np.random.seed(2023)  # Set seed such that each run is the same
    X = np.random.uniform(0, 100, (10000, 5))
    Y = np.random.randint(0, 5, 10000)
    impurity_tol_desired = 0.75

    tree = DecisionTree(
        "Classification",
        criteria=Gini_index,
        impurity_tol=impurity_tol_desired)
    tree.fit(X, Y)

    for node in tree.leaf_nodes:
        assert (
            node.impurity <= impurity_tol_desired
        ), f"Failed as node impurity was, {node.impurity} but should be at the most {impurity_tol_desired}"


def test_min_samples_split_setting():
    np.random.seed(2023)  # Set seed such that each run is the same
    X = np.random.uniform(0, 100, (10000, 5))
    Y = np.random.randint(0, 5, 10000)
    min_samples_split_desired = 1000

    tree = DecisionTree(
        "Classification",
        criteria=Gini_index,
        min_samples_split=min_samples_split_desired,
    )
    tree.fit(X, Y)
    for node in tree.leaf_nodes:
        assert (
            min_samples_split_desired <=
            (len(node.parent.indices))
        ), f"Failed as node had a parent with {min_samples_split_desired}, but which should have been a leaf node"


def test_min_samples_leaf_setting():
    np.random.seed(2023)  # Set seed such that each run is the same
    X = np.random.uniform(0, 100, (10000, 5))
    Y = np.random.randint(0, 5, 10000)
    min_samples_leaf_desired = 20

    tree = DecisionTree(
        "Classification",
        criteria=Gini_index,
        min_samples_leaf=min_samples_leaf_desired)
    tree.fit(X, Y)

    for node in tree.leaf_nodes:
        assert (
            min_samples_leaf_desired <= node.weighted_samples
        ), f"Failed as node had a parent with {min_samples_leaf_desired}, but which should have been a leaf node"


def test_min_improvement_setting():
    np.random.seed(2023)  # Set seed such that each run is the same
    X = np.random.randint(0, 100, (10000, 5))
    Y = np.random.randint(0, 10, 10000)
    min_improvement_desired = 0.000008

    tree = DecisionTree(
        "Classification",
        criteria=Gini_index,
        min_improvement=min_improvement_desired)
    tree.fit(X, Y)

    for node in tree.leaf_nodes:
        assert (
            abs(node.parent.impurity - node.impurity) > min_improvement_desired
        ), f"Failed as node had an impurity improvement greater than {abs(node.parent.impurity - node.impurity)}"


def get_x_y_classification(n, m):
    np.random.seed(2024)
    X = np.random.uniform(0, 100, (n, m))
    Y = np.random.randint(0, 10, n)
    return (X, Y)


def get_x_y_regression(n, m):
    np.random.seed(2024)
    X = np.random.uniform(0, 100, (n, m))
    Y = np.random.uniform(0, 10, n)
    return (X, Y)


def assert_tree_equality(t1: DecisionTree, t2: DecisionTree):
    root1 = t1.root
    root2 = t2.root

    q1, q2 = [root1], [root2]
    while len(q1) != 0:
        node1, node2 = q1.pop(), q2.pop()

        assert node1.depth == node2.depth
        assert (
            node1.impurity == node2.impurity
        ), f"{t1.tree_type}: {node1.impurity} != {node2.impurity}"

        if isinstance(node1, DecisionNode):
            assert isinstance(node2, DecisionNode)
            assert (
                node1.threshold == node2.threshold
            ), f"{t1.tree_type}: {node1.threshold} != {node2.threshold}"
            assert node1.depth == node2.depth
            assert node1.split_idx == node2.split_idx
            if node1.left_child:
                assert (
                    node2.left_child is not None
                ), "Node 1 had a left child but not Node 2"
                q1.append(node1.left_child)
                q2.append(node2.left_child)
            if node1.right_child:
                assert (
                    node2.right_child is not None
                ), "Node 1 had a right child but not Node 2"
                q1.append(node1.right_child)
                q2.append(node2.right_child)

        elif isinstance(node1, LeafNode):
            assert node1.weighted_samples == node2.weighted_samples
            assert np.array_equal(
                node1.value, node2.value
            ), f"{t1.tree_type}: {node1.value} != {node2.value}"
    assert len(
        q2) == 0, f"{t2.tree_type}: Queue 2 not empty with length {len(q2)}"


def test_sample_indices_classification():
    N, M = 10000, 5
    X1, Y1 = get_x_y_classification(N, M)
    X2, Y2 = get_x_y_classification(N, M)
    bloat_feature = np.linspace(0, 10, num=M)
    sample_indices = []
    for i in range(1, N * 2, 2):
        X2 = np.insert(X2, i, bloat_feature, axis=0)
        Y2 = np.insert(Y2, i, i)  # i is used as a bloat outcome value
        sample_indices.append(i - 1)

    t1 = DecisionTree("Classification", criteria=Gini_index)
    t2 = DecisionTree("Classification", criteria=Gini_index)

    t1.fit(X1, Y1)
    t2.fit(X2, Y2, sample_indices=sample_indices)

    assert_tree_equality(t1, t2)

    t1 = DecisionTree("Classification", criteria=Entropy)
    t2 = DecisionTree("Classification", criteria=Entropy)

    t1.fit(X1, Y1)
    t2.fit(X2, Y2, sample_indices=sample_indices)

    assert_tree_equality(t1, t2)


def test_sample_indices_regression():
    N, M = 5, 2
    X1, Y1 = get_x_y_regression(N, M)
    X2, Y2 = get_x_y_regression(N, M)
    bloat_feature = np.linspace(0, 10, num=M)
    sample_indices = []
    for i in range(1, N * 2, 2):
        X2 = np.insert(X2, i, bloat_feature, axis=0)
        Y2 = np.insert(Y2, i, i)  # i is used as a bloat outcome value
        sample_indices.append(i - 1)

    t1 = DecisionTree("Regression", criteria=Squared_error)
    t2 = DecisionTree("Regression", criteria=Squared_error)

    t1.fit(X1, Y1)
    t2.fit(X2, Y2, sample_indices=sample_indices)
    assert_tree_equality(t1, t2)

    t1 = DecisionTree("Regression", criteria=Linear_regression)
    t2 = DecisionTree("Regression", criteria=Linear_regression)

    t1.fit(X1, Y1)
    t2.fit(X2, Y2, sample_indices=sample_indices)
    assert_tree_equality(t1, t2)


def test_sample_weight_classification():
    N, M = 10000, 5
    X1, Y1 = get_x_y_classification(N, M)
    X2, Y2 = get_x_y_classification(N, M)
    bloat_feature = np.linspace(0, 10, num=M)
    sample_weights = []
    for i in range(1, N * 2, 2):
        X2 = np.insert(X2, i, bloat_feature, axis=0)
        Y2 = np.insert(Y2, i, i)  # i is used as a bloat outcome value
        sample_weights.append(1)
        sample_weights.append(0)

    t1 = DecisionTree("Classification", criteria=Gini_index)
    t2 = DecisionTree("Classification", criteria=Gini_index)

    t1.fit(X1, Y1)
    t2.fit(X2, Y2, sample_weight=sample_weights)

    assert_tree_equality(t1, t2)

    t1 = DecisionTree("Classification", criteria=Entropy)
    t2 = DecisionTree("Classification", criteria=Entropy)

    t1.fit(X1, Y1)
    t2.fit(X2, Y2, sample_weight=sample_weights)

    assert_tree_equality(t1, t2)


def test_sample_weight_regression():
    N, M = 10000, 5
    X1, Y1 = get_x_y_regression(N, M)
    X2, Y2 = get_x_y_regression(N, M)
    bloat_feature = np.linspace(0, 10, num=M)
    sample_weights = []
    for i in range(1, N * 2, 2):
        X2 = np.insert(X2, i, bloat_feature, axis=0)
        Y2 = np.insert(Y2, i, i)  # i is used as a bloat outcome value
        sample_weights.append(1)
        sample_weights.append(0)

    t1 = DecisionTree("Regression", criteria=Squared_error)
    t2 = DecisionTree("Regression", criteria=Squared_error)

    t1.fit(X1, Y1)
    t2.fit(X2, Y2, sample_weight=sample_weights)

    assert_tree_equality(t1, t2)

    t1 = DecisionTree("Regression", criteria=Linear_regression)
    t2 = DecisionTree("Regression", criteria=Linear_regression)

    t1.fit(X1, Y1)
    t2.fit(X2, Y2, sample_weight=sample_weights)

    assert_tree_equality(t1, t2)


def test_quantile_predict():
    X, Y = uniform_x_y(10000, 5)
    tree = DecisionTree(
        "Quantile",
        criteria=Squared_error,
        predict=PredictQuantile,
        leaf_builder=LeafBuilderRegression,
        max_depth=0,
    )
    tree.fit(X, Y)
    pred = tree.predict(
        X[0], quantile=0.95
    )  # As we are never splitting, we can just check a single data point
    np_quantile = np.quantile(Y, 0.95)
    assert (
        pred == np_quantile
    ), f"Quantile predict failed with {pred} - should be {np_quantile}"


def test_quantile_predict_array():
    X, Y = uniform_x_y(10000, 5)
    tree = DecisionTree(
        "Quantile",
        criteria=Squared_error,
        predict=PredictQuantile,
        leaf_builder=LeafBuilderRegression,
        max_depth=0,
    )
    tree.fit(X, Y)
    pred = tree.predict(
        X[0], quantile=[0.95, 0.1]
    )  # As we are never splitting, we can just check a single data point
    np_quantile = np.quantile(Y, [0.95, 0.1])
    assert (
        np.array_equal(pred, [np_quantile])
    ), f"Quantile predict failed with {pred} - should be {np_quantile}"


def test_linear_predict():
    """
    As the Linear Regression is fitted on the index 0 of the X training data,
    we can validate the new Prediction by first creating some "noise" data,
    and then create some data on the same line.
    Then with new values that should be on the same line,
    we can make sure that the predicted Y values,
    indeed are on the line.
    """
    np.random.seed(2024)
    X = np.random.uniform(1000, 100000, (1000, 5))  # noise
    Y = np.random.uniform(1000, 100000, (1000))

    X_Y_corr = np.arange(0, 50, step=1)
    idx = np.random.randint(0, 1000, 50)

    # Replace some indices with correlated data
    X[idx, 0] = X_Y_corr
    Y[idx] = X_Y_corr

    tree = DecisionTree(
        "LinearRegression",
        criteria=Linear_regression,
        predict=PredictLinearRegression,
        leaf_builder=LeafBuilderLinearRegression,
    )
    tree.fit(X, Y)

    X = np.random.uniform(1, 100, (50, 5))
    corr_data = np.arange(50, 100, step=1)
    X[:, 0] = corr_data
    prediction = tree.predict(X)
    assert (
        np.corrcoef(prediction) == 1.0
    ), "Linear Prediction didn't predict with perfect correlation"


if __name__ == "__main__":
    test_quantile_predict_array()
