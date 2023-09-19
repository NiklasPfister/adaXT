from adaXT.decision_tree._tree import *
#from adaXT.decision_tree.criteria import *
from adaXT.decision_tree._criteria import gini_index_wrapped, variance_wrapped
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
    tree.fit(X, Y_cla, gini_index_wrapped())
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
            assert cur_node.threshold == exp_val[i], f'Expected threshold {exp_val[i]} on node={i}, got {cur_node.threshold} on split_idx {cur_node.split_idx} exp: {spl_idx[i]}'
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
    tree.fit(X, Y_multi, gini_index_wrapped())
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
    tree.fit(X, Y_reg, variance_wrapped())
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
    pre_sorted = pre_sort(X).astype(int)
    tree = Tree("Classification", pre_sort=pre_sorted)
    tree.fit(X, Y_cla, gini_index_wrapped())
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
    tree.fit(X, Y_cla, gini_index_wrapped())
    prediction = tree.predict(X)
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
    tree.fit(X, Y_cla, gini_index_wrapped())
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




if __name__ == "__main__":
    # test_single_class()
    test_multi_class()
    # test_regression()
    # test_pre_sort()
    # test_prediction()
    # test_NxN_matrix()
    # print("done")