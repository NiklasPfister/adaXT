from adaXT.decision_tree import *
from adaXT.decision_tree.criteria import Gini_index, Squared_error, Entropy
from adaXT.decision_tree.tree_utils import print_tree, pre_sort, plot_tree

import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import cProfile
from pstats import Stats
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

def predict_run_time_full_dataset(tree_type, crit_func):
    max_depth = 10
    min_samples = 2
    if tree_type == "Classification":
        data = pd.read_csv(r'classification_data.csv')
    elif tree_type == "Regression":
        data = pd.read_csv(r'regression_data.csv')

    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]

    tree = DecisionTree(tree_type, max_depth=max_depth, min_samples=min_samples, criteria=crit_func)
    tree.fit(X, Y)

    start_time = time.perf_counter()
    pred = tree.predict(X)
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000 # elapsed time in ms

    np.savetxt("predictions.csv", pred, delimiter=",")

    return elapsed


def predict_run_time_full_dataset_sklearn(tree_type):
    max_depth = 10
    min_samples = 2

    if tree_type == "Classification":
        data = pd.read_csv(r'classification_data.csv')
    elif tree_type == "Regression":
        data = pd.read_csv(r'regression_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]

    if tree_type == "Classification":
        tr_cla = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples)
    elif tree_type == "Regression":
        tr_cla = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples)
    tr_cla.fit(X, Y)

    start_time = time.perf_counter()
    pred = tr_cla.predict(X)
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000 # elapsed time in ms
    np.savetxt("predictions_sklearn.csv", pred, delimiter=",")

    return elapsed

def compare_pred_sklearn_adaxt(tree_type, crit_func):
    max_depth = 5
    min_samples = 2

    # Setup data
    if tree_type == "Classification":
        data = pd.read_csv(r'classification_data.csv')
    elif tree_type == "Regression":
        data = pd.read_csv(r'regression_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]

    # Fit SKlearn
    if tree_type == "Classification":
        sk_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples)
    elif tree_type == "Regression":
        sk_tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples)
    sk_tree.fit(X, Y)

    # Fit adaXT
    adaxt_tree = DecisionTree(tree_type, max_depth=max_depth, min_samples=min_samples, criteria=crit_func)
    adaxt_tree.fit(X, Y)

    # predict with sklearn and adaxt
    sklearn_pred = sk_tree.predict(X)
    adaxt_pred = adaxt_tree.predict(X)

    np.savetxt("predictions.csv", adaxt_pred, delimiter=",")

    # Assert same shape so nothing has gone wrong in predict method
    assert(sklearn_pred.shape == adaxt_pred.shape)

    # Calculate squared sum of error between sklearn and adaXT
    for i in range(sklearn_pred.shape[0]):
        print("AdaXT", adaxt_pred[i])
        print("SKlearn", sklearn_pred[i])
        print("True", Y[i], "\n")


def compare_pred_sklearn_adaxt_alt(tree_type, crit_func):
    max_depth = 100
    min_samples = 2

    # Setup data
    if tree_type == "Classification":
        data = pd.read_csv(r'classification_data.csv')
    elif tree_type == "Regression":
        data = pd.read_csv(r'regression_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]

    # Fit adaXT
    adaxt_tree = DecisionTree(tree_type, min_samples=min_samples, criteria=crit_func)
    adaxt_tree.fit(X, Y)
    adaxt_pred = adaxt_tree.predict(X)
    np.savetxt("predictions.csv", adaxt_pred, delimiter=",")

    # Fit SKlearn
    if tree_type == "Classification":
        sk_tree = DecisionTreeClassifier(min_samples_split=min_samples)
    elif tree_type == "Regression":
        sk_tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples)
    sk_tree.fit(X, Y)
    sklearn_pred = sk_tree.predict(X)
    np.savetxt("predictions_sklearn.csv", sklearn_pred, delimiter=",")

    assert(sklearn_pred.shape == adaxt_pred.shape)

    num_y = Y.shape[0]
    sk_same_ada = 0
    sk_same_true = 0
    ada_same_true = 0
    for i in range(num_y):
        if abs(adaxt_pred[i] - sklearn_pred[i]) < 0.1:
            sk_same_ada += 1
        if abs(Y[i] - sklearn_pred[i]) < 0.1:
            sk_same_true += 1
        if abs(adaxt_pred[i] - Y[i]) < 0.1:
            ada_same_true += 1

    print("a same sk", sk_same_ada / num_y)
    print("sk same true", sk_same_true / num_y)
    print("a same true", ada_same_true / num_y, "\n")




if __name__ == "__main__":
    #profiler = cProfile.Profile()
    #profiler.enable()
    ## Code to run
    #predict_run_time_full_dataset("Classification", Gini_index)
    #predict_run_time_full_dataset("Classification", Entropy)
    #predict_run_time_full_dataset("Regression", Squared_error)
    #profiler.disable()
    #stats = Stats(profiler)
    #stats.sort_stats('tottime').print_stats(10)

    #print("runtime prediction gini:", predict_run_time_full_dataset("Classification", Gini_index))
    #print("runtime prediction entropy:", predict_run_time_full_dataset("Classification", Entropy))
    #print("runtime prediction squared:", predict_run_time_full_dataset("Regression", Squared_error))

    #print("runtime prediction sklearn classification:", predict_run_time_full_dataset_sklearn("Classification"))
    #print("runtime prediction sklearn regression:", predict_run_time_full_dataset_sklearn("Regression"))

    compare_pred_sklearn_adaxt_alt("Classification", Entropy)
    compare_pred_sklearn_adaxt_alt("Classification", Gini_index)
    #print("Correct percentage was", correct_per)