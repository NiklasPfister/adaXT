import numpy as np
import adaXT.criteria as crit
from adaXT.decision_tree import DecisionTree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import time
import matplotlib.pyplot as plt
import json
import sys


def run_classification_tree(X, Y, criteria):
    sk_time = 0
    if criteria.__name__ == "Gini_index":
        tree = DecisionTreeClassifier(criterion="gini")
        st = time.time()
        tree.fit(X, Y)
        et = time.time()
        sk_time = et - st
    elif criteria.__name__ == "Entropy":
        tree = DecisionTreeClassifier(criterion="entropy")
        st = time.time()
        tree.fit(X, Y)
        et = time.time()
        sk_time = et - st
    else:
        raise Exception("Tree neither Entropy nor Gini_index")

    tree = DecisionTree("Classification", criteria=criteria)
    st = time.time()
    tree.fit(X, Y)
    et = time.time()
    ada_time = et - st

    return (ada_time / sk_time, ada_time)


def run_regression_tree(X, Y, criteria):
    sk_time = 0
    if criteria.__name__ == "Squared_error":
        tree = DecisionTreeRegressor(criterion="squared_error")
        st = time.time()
        tree.fit(X, Y)
        et = time.time()
        sk_time = et - st
    else:
        raise Exception("Tree not a Squared_error")

    tree = DecisionTree("Regression", criteria=criteria)
    st = time.time()
    tree.fit(X, Y)
    et = time.time()
    ada_time = et - st

    return (ada_time / sk_time, ada_time)


def run_num_iterations(n, m, x=[0, 100], y=[0, 5], num_trees=10):
    X = np.random.uniform(x[0], x[1], (n, m))
    y_regression = np.random.uniform(y[0], y[1], n)
    y_classification = np.random.randint(y[0], y[1], n)
    run_times = np.empty(shape=(num_trees, 6))
    for i in range(num_trees):
        gini_diff, gini_time = run_classification_tree(
            X, y_classification, crit.Gini_index
        )
        entropy_diff, entropy_time = run_classification_tree(
            X, y_classification, crit.Entropy
        )
        squared_diff, square_time = run_regression_tree(
            X, y_regression, crit.Squared_error
        )
        run_times[i] = [
            gini_diff,
            entropy_diff,
            squared_diff,
            gini_time,
            entropy_time,
            square_time,
        ]
    return np.mean(run_times, axis=0)


def plot_subplot(ax, X, Ydiff, title):
    ax.set_title(title)
    ax.yaxis.set_ticks(np.arange(0, 20, 2))
    ax.xaxis.set_ticks(np.arange(1, 30, 3))
    ax.axvline(x=1, color="red")
    ax.plot(X, Ydiff)


def plot_new_data():
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
    i = 0
    j = 0
    data = {}
    for n in range(1000, 10000, 1000):
        X = []
        Ydiff = []
        Yrun = []
        d_entry = []
        for m in range(1, 30, 1):
            mean_run_times = run_num_iterations(n, m, num_trees=10)
            X.append(m)
            Ydiff.append(mean_run_times[:3])
            Yrun.append(mean_run_times[3:])
            d_entry.append({m: mean_run_times.tolist()})
        data[n] = d_entry
        plot_subplot(axs[i, j], X, Ydiff, n)
        if j == 2:
            i += 1
            j = 0
        else:
            j += 1
    fig.tight_layout()
    fig.legend(
        loc="outside upper center",
        labels=["x=1", "Gini diff", "Entropy diff", "Squared error diff"],
        ncols=3,
    )
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.xlabel("Number of features")
    plt.ylabel("adaXT/sklearn time")
    plt.savefig("decision_tree_time.svg", format="svg")
    with open("data.json", "w") as f:
        json.dump(data, f, indent=6)
    plt.show()


def plot_old_data():
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
    with open("data.json", "r") as f:
        data_json = json.load(f)
        i = 0
        j = 0
        for n in range(1000, 10000, 1000):
            X = []
            Ydiff = []
            Yrun = []
            for m in range(1, 30, 1):
                mean_run_times = data_json[str(n)][m - 1][str(m)]
                X.append(m)
                Ydiff.append(mean_run_times[:3])
                Yrun.append(mean_run_times[3:])
            plot_subplot(axs[i, j], X, Ydiff, n)
            if j == 2:
                i += 1
                j = 0
            else:
                j += 1
    fig.tight_layout()
    fig.legend(
        loc="outside upper center",
        labels=["x=1", "Gini diff", "Entropy diff", "Squared error diff"],
        ncols=3,
    )
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.xlabel("Number of features")
    plt.ylabel("adaXT/sklearn time")
    plt.savefig("decision_tree_time.svg", format="svg")
    plt.show()


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "old":
        plot_old_data()
    else:
        plot_new_data()
