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
import numpy as np


def test_run_time_single_tree_classification(crit_func):
    max_depth = 5
    min_samples = 2

    data = pd.read_csv(r'classification_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]

    start_time = time.perf_counter()
    tree = DecisionTree(
        "Classification",
        max_depth=max_depth,
        min_samples=min_samples,
        criteria=crit_func)
    tree.fit(X, Y)
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000  # elapsed time in ms

    new_time_entry = {
        "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num features": data.shape[1],
        "num rows": data.shape[0],
        "max depth": max_depth,
        "min samples": min_samples,
        "num trees": 1,
        "run time": elapsed
    }

    # add_time_entry(new_time_entry, "classification on a single tree")
    return elapsed


def test_run_time_single_tree_regression():
    max_depth = 5
    min_samples = 2

    data = pd.read_csv(r'regression_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]

    start_time = time.perf_counter()
    tree = DecisionTree(
        "Regression",
        max_depth=max_depth,
        min_samples=min_samples,
        criteria=Squared_error)
    tree.fit(X, Y)
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000  # elapsed time in ms

    new_time_entry = {
        "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num features": data.shape[1],
        "num rows": data.shape[0],
        "max depth": max_depth,
        "min samples": min_samples,
        "num trees": 1,
        "run time": elapsed
    }

    # add_time_entry(new_time_entry, "regression on a single tree")
    return elapsed


def test_run_time_multiple_tree_classification(num_trees_to_build, pre_sorted):
    max_depth = 5
    min_samples = 2

    data = pd.read_csv(r'classification_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]

    start_time = time.perf_counter()
    if pre_sorted:
        pre_sorted_data = pre_sort(X)
    else:
        pre_sorted_data = None

    for i in range(num_trees_to_build):
        tree = DecisionTree(
            "Classification",
            max_depth=max_depth,
            min_samples=min_samples,
            pre_sort=pre_sorted_data,
            criteria=Gini_index)
        tree.fit(X, Y)
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000  # elapsed time in ms

    new_time_entry = {
        "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num features": data.shape[1],
        "num rows": data.shape[0],
        "max depth": max_depth,
        "min samples": min_samples,
        "num trees": num_trees_to_build,
        "pre sorted": pre_sorted,
        "run time": elapsed
    }

    add_time_entry(new_time_entry, "classification on multiple trees")


def test_run_time_multiple_tree_regression(num_trees_to_build, pre_sorted):
    max_depth = 5
    min_samples = 2

    data = pd.read_csv(r'regression_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]

    start_time = time.perf_counter()

    if pre_sorted:
        pre_sorted_data = pre_sort(X)
    else:
        pre_sorted_data = None

    for i in range(num_trees_to_build):
        tree = DecisionTree(
            "Regression",
            max_depth=max_depth,
            min_samples=min_samples,
            pre_sort=pre_sorted_data,
            criteria=Squared_error)
        tree.fit(X, Y)
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000  # elapsed time in ms

    new_time_entry = {
        "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num features": data.shape[1],
        "num rows": data.shape[0],
        "max depth": max_depth,
        "min samples": min_samples,
        "num trees": num_trees_to_build,
        "pre sorted": pre_sorted,
        "run time": elapsed
    }

    add_time_entry(new_time_entry, "regression on multiple trees")


def add_time_entry(new_time_entry, key_to_write_to):
    # Load existing JSON data (if any)
    existing_json_data = {}
    try:
        with open("running_time.json", "r") as file:
            existing_json_data = json.load(file)
    except FileNotFoundError:
        pass

    # Get the data list or create a new list if it doesn't exist
    data_list = existing_json_data.get(key_to_write_to, [])

    # Append the new entry to the list
    data_list.append(new_time_entry)

    # Update the "Key1" in the JSON data with the new list
    existing_json_data[key_to_write_to] = data_list

    # Save the updated JSON data back to the file
    with open("running_time.json", "w") as file:
        json.dump(existing_json_data, file, indent=4)


def update_data_set(type, num_rows, num_features, num_classes):
    if type == "Classification":
        X = np.random.normal(loc=10, scale=5, size=(num_rows, num_features))
        Y = np.random.randint(num_classes, size=num_rows)
        data = np.hstack((X, Y.reshape(-1, 1)))
        # Save the new array to a CSV file
        np.savetxt("classification_data.csv", data, delimiter=",")

    if type == "Regression":
        X = np.random.normal(loc=10, scale=5, size=(num_rows, num_features))
        Y = np.random.normal(loc=10, scale=5, size=num_rows)
        data = np.hstack((X, Y.reshape(-1, 1)))
        # Save the new array to a CSV file
        np.savetxt("regression_data.csv", data, delimiter=",")


def run_sklearn_regression():
    max_depth = 5
    min_samples = 2
    data = pd.read_csv(r'regression_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]

    start_time = time.perf_counter()
    tr_cla = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples)
    tr_cla.fit(X, Y)
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000  # elapsed time in ms

    return elapsed


def run_sklearn_classification():
    max_depth = 5
    min_samples = 2
    data = pd.read_csv(r'classification_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]

    start_time = time.perf_counter()
    tr_cla = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples)
    tr_cla.fit(X, Y)
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000  # elapsed time in ms

    return elapsed


def test_run_time_single_tree_classification_presort():
    max_depth = 5
    min_samples = 2

    data = pd.read_csv(r'classification_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]

    start_time = time.perf_counter()
    pre_sorted = pre_sort(X).astype(int)
    tree = DecisionTree(
        "Classification",
        max_depth=max_depth,
        min_samples=min_samples,
        pre_sort=pre_sorted,
        criteria=Gini_index)
    tree.fit(X, Y)
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000  # elapsed time in ms

    new_time_entry = {
        "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num features": data.shape[1],
        "num rows": data.shape[0],
        "max depth": max_depth,
        "min samples": min_samples,
        "num trees": 1,
        "run time": elapsed
    }

    # add_time_entry(new_time_entry, "classification on a single tree with presort")
    return elapsed


def regression_table():
    data = []

    d = [5, 15, 25]
    n = [100, 1000, 2500]

    # For regression
    for d_el in d:
        for n_el in n:
            update_data_set("Regression", n_el, d_el, -1)
            sk_list = []
            ada_list = []

            for i in range(10):
                sk_list.append(run_sklearn_regression())
                ada_list.append(test_run_time_single_tree_regression())

            sk_time = sum(sk_list) / len(sk_list)

            ada_time = sum(ada_list) / len(ada_list)

            # print("With", n_el, "rows and", d_el, "features:")
            # print("sklearn runtime:", sk_time)
            # print("our runtime:", ada_time)
            # print("They are", ada_time / sk_time, "times faster than us.\n\n")

            data.append([f'{n_el} by {d_el}',
                         f'{ada_time:.1f} ms',
                         f'{sk_time:.1f} ms',
                         f'{ada_time / sk_time:.2f}'])

    # Create a Pandas DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            'rows by features',
            'us',
            'sklearn',
            'us / sklearn'])

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')  # Turn off axis lines and labels

    # Create a table from the DataFrame and display it
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2'] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.5)

    # Save the table as an image (e.g., PNG)
    plt.savefig('regression.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def classification_table_gini():
    data = []

    d = [5, 15, 25]
    n = [100, 1000, 2500]

    # For classification
    for d_el in d:
        for n_el in n:
            update_data_set("Classification", n_el, d_el, 3)
            sk_list = []
            ada_list = []

            for i in range(10):
                sk_list.append(run_sklearn_classification())
                ada_list.append(
                    test_run_time_single_tree_classification(Gini_index))

            sk_time = sum(sk_list) / len(sk_list)

            ada_time = sum(ada_list) / len(ada_list)

            # print("With", n_el, "rows and", d_el, "features:")
            # print("sklearn runtime:", sk_time)
            # print("our runtime:", ada_time)
            # print("They are", ada_time / sk_time, "times faster than us.\n\n")

            data.append([f'{n_el} by {d_el}',
                         f'{ada_time:.1f} ms',
                         f'{sk_time:.1f} ms',
                         f'{ada_time / sk_time:.2f}'])

    # Create a Pandas DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            'rows by features',
            'us',
            'sklearn',
            'us / sklearn'])

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')  # Turn off axis lines and labels

    # Create a table from the DataFrame and display it
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2'] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.5)

    # Save the table as an image (e.g., PNG)
    plt.savefig('classification_gini.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def classification_table_entropy():
    data = []

    d = [5, 15, 25]
    n = [100, 1000, 2500]

    # For classification
    for d_el in d:
        for n_el in n:
            update_data_set("Classification", n_el, d_el, 3)
            sk_list = []
            ada_list = []

            for i in range(10):
                sk_list.append(run_sklearn_classification())
                ada_list.append(
                    test_run_time_single_tree_classification(Entropy))

            sk_time = sum(sk_list) / len(sk_list)

            ada_time = sum(ada_list) / len(ada_list)

            # print("With", n_el, "rows and", d_el, "features:")
            # print("sklearn runtime:", sk_time)
            # print("our runtime:", ada_time)
            # print("They are", ada_time / sk_time, "times faster than us.\n\n")

            data.append([f'{n_el} by {d_el}',
                         f'{ada_time:.1f} ms',
                         f'{sk_time:.1f} ms',
                         f'{ada_time / sk_time:.2f}'])

    # Create a Pandas DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            'rows by features',
            'us',
            'sklearn',
            'us / sklearn'])

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')  # Turn off axis lines and labels

    # Create a table from the DataFrame and display it
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2'] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.5)

    # Save the table as an image (e.g., PNG)
    plt.savefig(
        'classification_entropy.png',
        bbox_inches='tight',
        pad_inches=0.1)
    plt.show()


def classification_table_change_num_classes():
    data = []

    # Classification with changing num classes
    n = [1000, 2000]
    number_of_classes = [5, 7, 10, 13]

    for n_el in n:
        for n_class in number_of_classes:
            update_data_set("Classification", n_el, 15, n_class)
            sk_list = []
            ada_list = []

            for i in range(10):
                sk_list.append(run_sklearn_classification())
                ada_list.append(
                    test_run_time_single_tree_classification(Gini_index))

            sk_time = sum(sk_list) / len(sk_list)

            ada_time = sum(ada_list) / len(ada_list)

            # print("With", n_el, "rows and", 15, "features, and", n_class, "classes:")
            # print("sklearn runtime:", sk_time)
            # print("our runtime:", ada_time)
            # print("They are", ada_time / sk_time, "times faster than us.\n\n")

            data.append([f'{n_el} by {n_class}',
                         f'{ada_time:.1f} ms',
                         f'{sk_time:.1f} ms',
                         f'{ada_time / sk_time:.2f}'])

    # Create a Pandas DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            'rows by num_classes',
            'us',
            'sklearn',
            'us / sklearn'])

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')  # Turn off axis lines and labels

    # Create a table from the DataFrame and display it
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2'] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.5)

    # Save the table as an image (e.g., PNG)
    plt.savefig(
        'classification_change_num_classes.png',
        bbox_inches='tight',
        pad_inches=0.1)
    plt.show()


if __name__ == "__main__":
    # remember to create datasets for time testing, if they have not been
    # previously created:
    update_data_set("Classification", 1000, 15, 3)
    update_data_set("Regression", 1000, 15, -1)
    # test_run_time_multiple_tree_classification(num_trees_to_build=20, pre_sorted=False)
    # test_run_time_multiple_tree_classification(num_trees_to_build=20, pre_sorted=True)

    # profiler = cProfile.Profile()
    # profiler.enable()
    # Code to run
    # test_run_time_single_tree_regression()
    # lst = []
    # for i in range(10):
    #    lst.append(test_run_time_single_tree_classification_presort())
    # print(sum(lst) / len(lst))
    # test_run_time_single_tree_classification()
    # profiler.disable()
    # stats = Stats(profiler)
    # stats.sort_stats('tottime').print_stats(20)

    # profiler = cProfile.Profile()
    # profiler.enable()
    # Code to run
    # print("ms", test_run_time_single_tree_regression())
    # print(test_run_time_single_tree_classification(Gini_index))
    # profiler.disable()
    # stats = Stats(profiler)
    # stats.sort_stats('tottime').print_stats(10)

    # test_run_time_single_tree_classification()
    # print("Sklearn time regression:", run_sklearn_regression())
    # print("Sklearn time classification:", run_sklearn_classification())

    # regression_table()
    # classification_table_gini()
    # classification_table_entropy()
    # classification_table_change_num_classes()

    # lst = []
    # for i in range(100):
    #    lst.append(test_run_time_single_tree_classification())
    # print("entropy", sum(lst) / len(lst))

    # lst_new = []
    # lst_old = []
    # for i in range(100):
    #    lst_new.append(test_run_time_single_tree_classification(gini_index_new()))
    #    lst_old.append(test_run_time_single_tree_classification(Gini_index))
    # print("old gini", sum(lst_old) / len(lst_old))
    # print("new gini", sum(lst_new) / len(lst_new))
