from adaXT.decision_tree.tree import *
#from adaXT.decision_tree.criteria import *
from adaXT.decision_tree.criteria_cy import gini_index, variance
from adaXT.decision_tree.tree_utils import print_tree, pre_sort, plot_tree

import time
import json
import pandas as pd
import matplotlib.pyplot as plt

def test_run_time_single_tree_classification():
    max_depth = 5
    min_samples = 2

    data = pd.read_csv(r'classification_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1].astype(np.int_)
    
    start_time = time.perf_counter()
    tree = Tree("Classification", max_depth=max_depth, min_samples=min_samples)
    tree.fit(X, Y, gini_index)
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000 # elapsed time in ms

    new_time_entry = {
        "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num features": data.shape[1],
        "num rows": data.shape[0],
        "max depth": max_depth,
        "min samples": min_samples,
        "num trees": 1,
        "run time": elapsed
    }

    add_time_entry(new_time_entry, "classification on a single tree")

def test_run_time_single_tree_regression():
    max_depth = 5
    min_samples = 2

    data = pd.read_csv(r'regression_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]
    
    start_time = time.perf_counter()
    tree = Tree("Regression", max_depth=max_depth, min_samples=min_samples)
    tree.fit(X, Y, variance)
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000 # elapsed time in ms

    new_time_entry = {
        "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num features": data.shape[1],
        "num rows": data.shape[0],
        "max depth": max_depth,
        "min samples": min_samples,
        "num trees": 1,
        "run time": elapsed
    }

    add_time_entry(new_time_entry, "regression on a single tree")

def test_run_time_multiple_tree_classification(num_trees_to_build, pre_sorted):
    max_depth = 5
    min_samples = 2

    data = pd.read_csv(r'classification_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1].astype(int)
    
    start_time = time.perf_counter()
    if pre_sorted:
        pre_sorted_data = pre_sort(X)
    else:
        pre_sorted_data = None
    
    for i in range(num_trees_to_build):
        tree = Tree("Classification", max_depth=max_depth, min_samples=min_samples, pre_sort=pre_sorted_data)
        tree.fit(X, Y, gini_index)
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000 # elapsed time in ms

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
        tree = Tree("Regression", max_depth=max_depth, min_samples=min_samples, pre_sort=pre_sorted_data)
        tree.fit(X, Y, variance)
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000 # elapsed time in ms

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

def update_data_set(type, num_rows, num_features):
    if type == "Classification":
        X = np.random.normal(loc=10, scale=5, size=(num_rows, num_features))
        Y = np.random.randint(2, size=num_rows)
        data = np.hstack((X, Y.reshape(-1, 1)))
        # Save the new array to a CSV file
        np.savetxt("classification_data.csv", data, delimiter=",")

    if type == "Regression":
        X = np.random.normal(loc=10, scale=5, size=(num_rows, num_features))
        Y = np.random.normal(loc=10, scale=5, size=num_rows)
        data = np.hstack((X, Y.reshape(-1, 1)))
        # Save the new array to a CSV file
        np.savetxt("regression_data.csv", data, delimiter=",")


if __name__ == "__main__":
    # remember to create datasets for time testing, if they have not been previously created:
    #update_data_set("Classification", 1000, 10)
    #update_data_set("Regression", 1000, 10)
    test_run_time_multiple_tree_classification(num_trees_to_build=20, pre_sorted=False)
    test_run_time_multiple_tree_classification(num_trees_to_build=20, pre_sorted=True)
    test_run_time_single_tree_regression()
    test_run_time_single_tree_classification()