from adaXT.decision_tree.tree import Tree
from adaXT.decision_tree.criteria import Gini_index, Squared_error
import numpy as np

def test_predict_matrix_classification():
    X = np.array([[1, -1],
                  [-0.5, -2],
                  [-1, -1],
                  [-0.5, -0.5],
                  [1, 0],
                  [-1, 1],
                  [1, 1],
                  [-0.5, 2]])
    Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])

    tree = Tree("Classification")
    tree.fit(X, Y_cla, Gini_index)
    res1 = tree.weight_matrix()
    res2 = tree.predict_matrix(X)
    assert res1.shape == res2.shape
    row, col = res1.shape
    for i in range(row):
        for j in range(col):
            assert res1[i, j] == res2[i, j]

def test_predict_matrix_regression():
    X = np.array([[1, -1],
                  [-0.5, -2],
                  [-1, -1],
                  [-0.5, -0.5],
                  [1, 0],
                  [-1, 1],
                  [1, 1],
                  [-0.5, 2]])
    Y_reg = np.array([2.2, -0.5, 0.5, -0.5, 2, -3, 2.2, -3])
    tree = Tree("Regression")
    tree.fit(X, Y_reg, Squared_error)

    res1 = tree.weight_matrix()
    res2 = tree.predict_matrix(X)
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
    tree = Tree("Classification")
    tree.fit(X, Y_cla, Gini_index)
    prediction = tree.predict(X)
    for i in range(len(Y_cla)):
        assert Y_cla[i] == prediction[
            i], f"incorrect prediction at {i}, expected {Y_cla[i]} got {prediction[i]}"


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
    tree = Tree("Classification")
    tree.fit(X, Y_cla, Gini_index)
    weight_matrix = tree.weight_matrix()
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
            assert weight_matrix[i, j] == true_weight[i,
                                                      j], f"Failed on ({i}, {j}), should be {true_weight[i, j]} was {weight_matrix[i, j]}"