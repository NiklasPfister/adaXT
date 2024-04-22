import numpy as np
import adaXT.criteria as crit
from adaXT.decision_tree import DecisionTree
import time
from cProfile import Profile
from pstats import SortKey, Stats


def run_classification_tree(X, Y, criteria):
    tree = DecisionTree("Classification", criteria=criteria)
    st = time.time()
    tree.fit(X, Y)
    et = time.time()
    ada_time = et - st

    return ada_time


def run_regression_tree(X, Y, criteria):
    tree = DecisionTree("Regression", criteria=criteria)
    st = time.time()
    tree.fit(X, Y)
    et = time.time()
    ada_time = et - st

    return ada_time


if __name__ == "__main__":
    np.random.seed(2024)
    n = 10000
    m = 5
    X = np.random.uniform(0, 100, (n, m))
    Y_cla = np.random.randint(0, 5, n)
    Y_reg = np.random.uniform(0, 5, n)
    gini = DecisionTree("Classification", criteria=crit.Gini_index)
    entropy = DecisionTree("Classification", criteria=crit.Entropy)
    squared_error = DecisionTree("Regression", criteria=crit.Squared_error)
    with Profile() as profile:
        gini.fit(X, Y_cla)
        (Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats())
    with Profile() as profile:
        entropy.fit(X, Y_cla)
        (Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats())
    with Profile() as profile:
        squared_error.fit(X, Y_reg)
        (Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats())
