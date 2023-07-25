from adaXT.decision_tree.tree import *
from adaXT.decision_tree.criteria import *
from adaXT.decision_tree.criteria import gini_index
from adaXT.decision_tree.tree_utils import print_tree, pre_sort, plot_tree

import time
import json
import pandas as pd
import matplotlib.pyplot as plt

def rec_node(node: LeafNode|DecisionNode|None, depth: int) -> None:
    """
    Used to check the depth value associated with nodes

    Parameters
    ----------
    node : LeafNode | DecisionNode | None
        node to recurse on
    depth : int
        expected depth of the node
    """
    if type(node) == LeafNode or type(node) == DecisionNode:
        assert node.depth == depth, f'Incorrect depth, expected {depth} got {node.depth}'
        if type(node) == DecisionNode:
            rec_node(node.left_child, depth+1)

def test_single_class():
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
    tree.fit(X, Y_cla, gini_index)
    root = tree.root
    exp_val = [0.25, -0.75, 0]
    spl_idx = [0, 0, 1]
    assert type(root) == LeafNode or type(root) == DecisionNode, f"root is not a node but {type(root)}"
    queue = [root]
    i = 0
    # Loop over all the nodes
    while len(queue) > 0:
        cur_node = queue.pop()
        if type(cur_node) == DecisionNode: # Check threshold and idx of decision node
            assert cur_node.threshold == exp_val[i], f'Expected threshold {exp_val[i]} on i={i}, got {cur_node.threshold}'
            assert cur_node.split_idx == spl_idx[i], f'Expected split idx {spl_idx[i]} on i={i}, got {cur_node.split_idx}'
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
            i += 1
        elif type(cur_node) == LeafNode: # Check that the value is of length 2
            assert len(cur_node.value) == 2, f'Expected 2 mean values, one for each class, but got: {len(cur_node.value)}'
        
    rec_node(root, 0)

def test_multi_class():
    X = np.array([[1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2]])
    Y_multi = np.array([1, 2, 1, 0, 1, 0, 1, 0])
    Y_unique = len(np.unique(Y_multi))
    tree = Tree("Classification")
    tree.fit(X, Y_multi, gini_index)
    root = tree.root
    exp_val = [0.25, -0.75, -0.75] # DIFFERENT FROM SKLEARN THEIRS IS: [0.25, -0.75, -1.5], both give pure leaf node
    spl_idx = [0, 1, 0] # DIFFERENT FROM SKLEARN THEIRS IS: [0, 1, 1], both give pure leaf node
    assert type(root) == LeafNode or type(root) == DecisionNode, f"root is not a node but {type(root)}"
    queue = [root]
    i = 0
    while len(queue) > 0:
        cur_node = queue.pop()
        if type(cur_node) == DecisionNode:
            assert cur_node.threshold == exp_val[i], f'Expected threshold {exp_val[i]}, got {cur_node.threshold}'
            assert cur_node.split_idx == spl_idx[i], f'Expected split idx {spl_idx[i]}, got {cur_node.split_idx}'
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
            i += 1
        elif type(cur_node) == LeafNode:
            assert len(cur_node.value) == Y_unique, f'Expected {Y_unique} mean values, one for each class, but got: {len(cur_node.value)}'
            
    rec_node(root, 0)

def test_regression():
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
    tree.fit(X, Y_reg, variance)
    root = tree.root
    exp_val2 = [0.25, -0.5, 0.5, 0.25, -0.75]
    spl_idx2 = [0, 1, 1, 1, 0]
    assert type(root) == LeafNode or type(root) == DecisionNode, f"root is not a node but {type(root)}"
    queue = [root]
    i = 0
    while len(queue) > 0:
        cur_node = queue.pop()
        if type(cur_node) == DecisionNode:
            assert cur_node.threshold == exp_val2[i], f'Expected threshold {exp_val2[i]}, got {cur_node.threshold}'
            assert cur_node.split_idx == spl_idx2[i], f'Expected split idx {spl_idx2[i]}, got {cur_node.split_idx}'
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
            i += 1
        elif type(cur_node) == LeafNode:
            assert len(cur_node.value) == 1, f'Expected {1} mean values, but got: {len(cur_node.value)}'
    rec_node(root, 0)


def test_pre_sort():
    X = np.array([[1, -1],
                [-0.5, -2],
                [-1, -1],
                [-0.5, -0.5],
                [1, 0],
                [-1, 1],
                [1, 1],
                [-0.5, 2]])
    Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    pre_sorted = pre_sort(X)
    tree = Tree("Classification", pre_sort=pre_sorted)
    print(pre_sorted)
    tree.fit(X, Y_cla, gini_index)
    root = tree.root
    exp_val = [0.25, -0.75, 0]
    spl_idx = [0, 0, 1]
    assert type(root) == LeafNode or type(root) == DecisionNode, f"root is not a node but {type(root)}"
    queue = [root]
    i = 0
    # Loop over all the nodes
    while len(queue) > 0:
        cur_node = queue.pop()
        if type(cur_node) == DecisionNode: # Check threshold and idx of decision node
            assert cur_node.threshold == exp_val[i], f'Expected threshold {exp_val[i]} on i={i}, got {cur_node.threshold}'
            assert cur_node.split_idx == spl_idx[i], f'Expected split idx {spl_idx[i]} on i={i}, got {cur_node.split_idx}'
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
            i += 1
        elif type(cur_node) == LeafNode: # Check that the value is of length 2
            assert len(cur_node.value) == 2, f'Expected 2 mean values, one for each class, but got: {len(cur_node.value)}'
        
    rec_node(root, 0)

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
    tree.fit(X, Y_cla, gini_index)
    prediction = tree.predict(np.array(X))
    print(prediction)
    for i in range(len(Y_cla)):
        assert Y_cla[i] == prediction[i], f"incorrect prediction at {i}, expected {Y_cla[i]} got {prediction[i]}"

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
    tree.fit(X, Y_cla, gini_index)
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
            assert weight_matrix[i, j] == true_weight[i, j], f"Failed on ({i}, {j}), should be {true_weight[i, j]} was {weight_matrix[i, j]}"

def test_run_time_single_tree_classification():
    max_depth = 5
    min_samples = 2

    data = pd.read_csv(r'tests\classification_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]
    
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

    data = pd.read_csv(r'tests\regression_data.csv')
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

    data = pd.read_csv(r'tests\classification_data.csv')
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]
    
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

    data = pd.read_csv(r'tests\regression_data.csv')
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
        with open("tests\\running_time.json", "r") as file:
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
    with open("tests\\running_time.json", "w") as file:
        json.dump(existing_json_data, file, indent=4)

def update_data_set(type, num_rows, num_features):
    if type == "Classification":
        X = np.random.normal(loc=10, scale=5, size=(num_rows, num_features))
        Y = np.random.randint(2, size=num_rows)
        data = np.hstack((X, Y.reshape(-1, 1)))
        # Save the new array to a CSV file
        np.savetxt("tests\\classification_data.csv", data, delimiter=",")

    if type == "Regression":
        X = np.random.normal(loc=10, scale=5, size=(num_rows, num_features))
        Y = np.random.normal(loc=10, scale=5, size=num_rows)
        data = np.hstack((X, Y.reshape(-1, 1)))
        # Save the new array to a CSV file
        np.savetxt("tests\\regression_data.csv", data, delimiter=",")




if __name__ == "__main__":
    # remember to create datasets for time testing, if they have not been previously created:
    #update_data_set("Classification", 10000, 5)
    #update_data_set("Regression", 200, 10)
    #test_run_time_multiple_tree_classification(num_trees_to_build=20, pre_sorted=False)
    #test_run_time_multiple_tree_classification(num_trees_to_build=20, pre_sorted=True)
    #test_run_time_multiple_tree_regression(num_trees_to_build=50, pre_sorted=False)
    #test_run_time_multiple_tree_regression(num_trees_to_build=50, pre_sorted=True)
    test_single_class()
    test_multi_class()
    test_regression()
    test_pre_sort()
    test_prediction()
    test_NxN_matrix()
    #print("All tests passed succesfully")