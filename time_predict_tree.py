import numpy as np
import adaXT.criteria as crit
from adaXT.decision_tree import DecisionTree
from adaXT.random_forest import RandomForest
from sklearn.tree import DecisionTreeClassifier
import time

from memory_profiler import profile

low_X = 0
high_X = 10_000
N_TRAIN = 100_000
M = 5
N_PREDICT = N_TRAIN
NUM_TREES = 1
NUM_PREDICT = 1


def predict_n_times(tree):
    """
    predicts NUM_PREDICT number of times using the tree on randomly generated X
    data.

    Returns the mean predict time of NUM_PREDICT randomly generated values.
    """
    times = []
    for _ in range(NUM_PREDICT):
        X = np.random.uniform(low_X, high_X, (N_PREDICT, M))
        st = time.time()
        x = tree.predict(X)
        et = time.time()
        times.append(et - st)
    return np.mean(times)


def main():
    X = np.random.uniform(low_X, high_X, (N_TRAIN, M))
    Y = np.random.randint(0, M, N_TRAIN)

    times = []
    for _ in range(NUM_TREES):
        tree = RandomForest(forest_type="Classification", criteria=crit.Gini_index)
        # tree = DecisionTreeClassifier(criterion="gini")
        tree.fit(X, Y)
        times.append(predict_n_times(tree))

    mean_predict_time = np.mean(times)
    print(
        f"Mean predict times for {NUM_TREES} predicting {NUM_PREDICT} times: ",
        mean_predict_time,
    )


if __name__ == "__main__":
    main()
