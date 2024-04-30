from multiprocessing import cpu_count
from adaXT.random_forest import RandomForest
from adaXT.criteria import Gini_index, Squared_error, Entropy
import matplotlib.pyplot as plt
import time
import numpy as np


def plot_running_time(n_jobs, running_time, ax, title):
    ax.plot(n_jobs, running_time, title=title)


def get_regression_data(
    n, m, random_state: np.random.RandomState, lowx=0, highx=100, lowy=0, highy=5
):
    X = random_state.uniform(lowx, highx, (n, m))
    Y = random_state.uniform(lowy, highy, n)
    return (X, Y)


def get_classification_data(
    n, m, random_state: np.random.RandomState, lowx=0, highx=100, lowy=0, highy=5
):
    X = random_state.uniform(lowx, highx, (n, m))
    Y = random_state.randint(lowy, highy, n)
    return (X, Y)


def run_gini_index(X, Y, n_jobs, n_estimators):
    forest = RandomForest(
        forest_type="Classification",
        criteria=Gini_index,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    st = time.time()
    forest.fit(X, Y)
    et = time.time()
    return et - st


def run_entropy(X, Y, n_jobs, n_estimators):
    forest = RandomForest(
        forest_type="Classification",
        criteria=Entropy,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    st = time.time()
    forest.fit(X, Y)
    et = time.time()
    return et - st


def run_squared_error(X, Y, n_jobs, n_estimators):
    forest = RandomForest(
        forest_type="Regression",
        criteria=Squared_error,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    st = time.time()
    forest.fit(X, Y)
    et = time.time()
    return et - st


def running_time(n, m, random_state, n_jobs, n_estimators):
    X_cla, Y_cla = get_classification_data(n, m, random_state=random_state)
    X_reg, Y_reg = get_regression_data(n, m, random_state=random_state)
    return [
        run_entropy(X_cla, Y_cla, n_jobs=n_jobs, n_estimators=n_estimators),
        run_gini_index(X_cla, Y_cla, n_jobs=n_jobs, n_estimators=n_estimators),
        run_squared_error(X_reg, Y_reg, n_jobs=n_jobs, n_estimators=n_estimators),
    ]


if __name__ == "__main__":
    random_state = np.random.RandomState(2024)
    n_jobs = []
    mean_running_times = []
    n = 10000
    m = 4
    n_estimators = 100
    for i in range(1, cpu_count()):
        print(f"njobs = {i}")
        n_jobs.append(i)
        running_times = []
        for _ in range(1):
            running_times.append(running_time(n, m, random_state, i, n_estimators))
        mean_running_times.append(np.mean(running_times, axis=0))
    plt.plot(n_jobs, mean_running_times)
    plt.show()
