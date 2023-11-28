from adaXT.decision_tree import DecisionTree
from adaXT.decision_tree.criteria import Gini_index, Squared_error
import numpy as np


def test_predict_leaf_matrix_classification():
    X = np.array([[1, -1],
                  [-0.5, -2],
                  [-1, -1],
                  [-0.5, -0.5],
                  [1, 0],
                  [-1, 1],
                  [1, 1],
                  [-0.5, 2]])
    Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])

    tree = DecisionTree("Classification", Gini_index)
    tree.fit(X, Y_cla)
    res1 = tree.get_leaf_matrix()
    res2 = tree.predict_leaf_matrix(X)
    assert res1.shape == res2.shape
    row, col = res1.shape
    for i in range(row):
        for j in range(col):
            assert res1[i, j] == res2[i, j]


def test_predict_leaf_matrix_regression():
    X = np.array([[1, -1],
                  [-0.5, -2],
                  [-1, -1],
                  [-0.5, -0.5],
                  [1, 0],
                  [-1, 1],
                  [1, 1],
                  [-0.5, 2]])
    Y_reg = np.array([2.2, -0.5, 0.5, -0.5, 2, -3, 2.2, -3])
    tree = DecisionTree("Regression", Squared_error)
    tree.fit(X, Y_reg)

    res1 = tree.get_leaf_matrix()
    res2 = tree.predict_leaf_matrix(X)
    assert (res1.shape == res2.shape)
    row, col = res1.shape
    for i in range(row):
        for j in range(col):
            assert res1[i, j] == res2[i, j]


def test_predict_leaf_matrix_regression_with_scaling():
    X = np.array([[1, -1],
                  [-0.5, -2],
                  [-1, -1],
                  [-0.5, -0.5],
                  [1, 0],
                  [-1, 1],
                  [1, 1],
                  [-0.5, 2]])
    Y_reg = np.array([2.2, -0.5, 0.5, -0.5, 2, -3, 2.2, -3])
    tree = DecisionTree("Regression", Squared_error)
    tree.fit(X, Y_reg)

    res1 = tree.get_leaf_matrix()
    for i in range(res1.shape[0]):
        res1[i] = res1[i] / np.sum(res1[i])
    res2 = tree.predict_leaf_matrix(X, scale=True)
    assert (res1.shape == res2.shape)
    row, col = res1.shape
    for i in range(row):
        for j in range(col):
            assert res1[i, j] == res2[i, j]


def test_prediction():
    X = np.array([[1, -1],
                  [-0.5, -2],
                  [-1, -1],
                  [-0.5, -0.5],
                  [1, 0],
                  [-1, 1],
                  [1, 1],
                  [-0.5, 2]])
    Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    tree = DecisionTree("Classification", Gini_index)
    tree.fit(X, Y_cla)
    prediction = tree.predict(X)
    for i in range(len(Y_cla)):
        assert Y_cla[i] == prediction[
            i], f"incorrect prediction at {i}, expected {Y_cla[i]} got {prediction[i]}"


def test_predict_proba_probability():
    X = np.array([[1, -1],
                  [-0.5, -2],
                  [-1, -1],
                  [-0.5, -0.5],
                  [1, 0],
                  [-1, 1],
                  [1, 1],
                  [-0.5, 2]])
    Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    tree = DecisionTree("Classification", Gini_index)
    tree.fit(X, Y_cla)
    classes, prediction = tree.predict_proba(X)
    assert prediction.shape[0] == X.shape[0]
    for i in range(len(Y_cla)):
        assert Y_cla[i] == classes[np.argmax(
            prediction[i, :])], f"incorrect prediction at {i}, expected {Y_cla[i]} got {classes[np.argmax(prediction[i, :])]}"


def test_predict_proba_against_predict():
    X = np.random.uniform(0, 100, (10000, 5))
    Y = np.random.randint(0, 5, 10000)

    tree = DecisionTree("Classification", Gini_index)
    tree.fit(X, Y)

    predict = tree.predict(X)
    classes, predict_proba = tree.predict_proba(X)

    for i in range(predict.shape[0]):
        assert predict[i] == classes[np.argmax(
            predict_proba[i, :])], f"incorrect prediction at {i}, expected {predict[i]} got {classes[np.argmax(predict_proba[i, :])]}"


def test_NxN_matrix():
    X = np.array([[1, -1],
                  [-0.5, -2],
                  [-1, -1],
                  [-0.5, -0.5],
                  [1, 0],
                  [-1, 1],
                  [1, 1],
                  [-0.5, 2]])
    Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    tree = DecisionTree("Classification", Gini_index)
    tree.fit(X, Y_cla)
    leaf_matrix = tree.get_leaf_matrix()
    true_weight = np.array([
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 1]
    ])
    for i in range(len(true_weight)):
        for j in range(len(true_weight[0])):
            assert leaf_matrix[i, j] == true_weight[i,
                                                    j], f"Failed on ({i}, {j}), should be {true_weight[i, j]} was {leaf_matrix[i, j]}"


def test_max_depth_setting():
    X = np.random.uniform(0, 100, (10000, 5))
    Y = np.random.randint(0, 5, 10000)
    max_depth_desired = 5

    tree = DecisionTree(
        "Classification",
        criteria=Gini_index,
        max_depth=max_depth_desired)
    tree.fit(X, Y)

    for node in tree.leaf_nodes:
        assert node.depth <= max_depth_desired, f"Failed as node depth was,{node.depth} but should be at the most {max_depth_desired}"


def test_impurity_tol_setting():
    X = np.random.uniform(0, 100, (10000, 5))
    Y = np.random.randint(0, 5, 10000)
    impurity_tol_desired = 0.5

    tree = DecisionTree(
        "Classification",
        criteria=Gini_index,
        impurity_tol=impurity_tol_desired)
    tree.fit(X, Y)

    for node in tree.leaf_nodes:
        assert node.impurity < impurity_tol_desired, f"Failed as node impurity was,{node.impurity} but should be at the most {impurity_tol_desired}"


def test_min_samples_split_setting():
    X = np.random.uniform(0, 100, (10000, 5))
    Y = np.random.randint(0, 5, 10000)
    min_samples_split_desired = 1000

    tree = DecisionTree(
        "Classification",
        criteria=Gini_index,
        min_samples=min_samples_split_desired)
    tree.fit(X, Y)

    for node in tree.leaf_nodes:
        assert min_samples_split_desired <= node.parent.n_samples, f"Failed as node had a parent with {min_samples_split_desired}, but which should have been a lead node"


def test_min_improvement_setting():
    np.random.seed(2023)  # Set seed such that each run is the same
    X = np.random.randint(0, 10000, (10000, 5))
    Y = np.random.randint(0, 100, 10000)
    min_improvement_desired = 0.2

    tree = DecisionTree(
        "Classification",
        criteria=Gini_index,
        min_improvement=min_improvement_desired)

    tree.fit(X, Y)

    print(max(tree.leaf_nodes, key=lambda x: x.depth).depth)

    for node in tree.leaf_nodes:
        assert abs(node.parent.impurity -
                   node.impurity) < min_improvement_desired, f"Failed as node had an impurity improvement greater than {abs(node.parent.impurity - node.impurity)}"


if __name__ == "__main__":
    test_predict_leaf_matrix_classification()
    test_predict_leaf_matrix_regression()
    test_predict_leaf_matrix_regression_with_scaling()
    test_prediction()
    test_predict_proba_probability()
    test_predict_proba_against_predict()
    test_NxN_matrix()
    test_max_depth_setting()
    test_impurity_tol_setting()
    test_min_samples_split_setting()
    test_min_improvement_setting()
    print("Done.")
