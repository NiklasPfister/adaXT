from adaXT.decision_tree import DecisionTree, LeafNode, DecisionNode
from adaXT.criteria import Gini_index, Squared_error, Entropy, Linear_regression

import numpy as np
import scipy


def rec_node(node: LeafNode | DecisionNode | None, depth: int) -> None:
    """
    Used to check the depth value associated with nodes

    Parameters
    ----------
    node : LeafNode | DecisionNode | None
        node to recurse on
    depth : int
        expected depth of the node
    """
    if isinstance(node, LeafNode) or isinstance(node, DecisionNode):
        assert (
            node.depth == depth
        ), f"Incorrect depth, expected {depth} got {node.depth}"
        if isinstance(node, DecisionNode):
            rec_node(node.left_child, depth + 1)


def test_gini_single():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    tree = DecisionTree("Classification", criteria=Gini_index)
    tree.fit(X, Y_cla)
    root = tree.root
    exp_val = [0.25, -0.75, 0]
    spl_idx = [0, 0, 1]
    assert isinstance(root, LeafNode) or isinstance(
        root, DecisionNode
    ), f"root is not a node but {type(root)}"
    queue = [root]
    i = 0

    # Loop over all the nodes
    while len(queue) > 0:
        cur_node = queue.pop()
        if isinstance(
            cur_node, DecisionNode
        ):  # Check threshold and idx of decision node
            assert (
                cur_node.threshold == exp_val[i]
            ), f"Expected threshold {exp_val[i]} on node={i}, got {cur_node.threshold} on split_idx {cur_node.split_idx} exp: {spl_idx[i]}"
            assert (
                cur_node.split_idx == spl_idx[i]
            ), f"Expected split idx {spl_idx[i]} on i={i}, got {cur_node.split_idx}"
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
            i += 1
        elif isinstance(cur_node, LeafNode):  # Check that the value is of length 2
            assert (
                len(cur_node.value) == 2
            ), f"Expected 2 mean values, one for each class, but got: {len(cur_node.value)}"

    rec_node(root, 0)


def test_gini_multi():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_multi = np.array([1, 2, 1, 0, 1, 0, 1, 0])
    Y_unique = len(np.unique(Y_multi))
    tree = DecisionTree("Classification", criteria=Gini_index)
    tree.fit(X, Y_multi)
    root = tree.root
    # DIFFERENT FROM SKLEARN THEIRS IS: [0.25, -0.75, -1.5], both give pure
    # leaf node
    exp_val = [0.25, -0.75, -0.75]
    # DIFFERENT FROM SKLEARN THEIRS IS: [0, 1, 1], both give pure leaf node
    spl_idx = [0, 1, 0]
    assert isinstance(root, LeafNode) or isinstance(
        root, DecisionNode
    ), f"root is not a node but {type(root)}"
    queue = [root]
    i = 0
    while len(queue) > 0:
        cur_node = queue.pop()
        if isinstance(cur_node, DecisionNode):
            assert (
                cur_node.threshold == exp_val[i]
            ), f"Expected threshold {exp_val[i]}, got {cur_node.threshold}"
            assert (
                cur_node.split_idx == spl_idx[i]
            ), f"Expected split idx {spl_idx[i]}, got {cur_node.split_idx}"
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
            i += 1
        elif isinstance(cur_node, LeafNode):
            assert (
                len(cur_node.value) == Y_unique
            ), f"Expected {Y_unique} mean values, one for each class, but got: {len(cur_node.value)}"

    rec_node(root, 0)


def test_regression():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_reg = np.array([2.2, -0.5, 0.5, -0.5, 2, -3, 2.2, -3])
    tree = DecisionTree("Regression", criteria=Squared_error)
    tree.fit(X, Y_reg)
    root = tree.root
    exp_val2 = [0.25, -0.5, 0.5, 0.25, -0.75]
    spl_idx2 = [0, 1, 1, 1, 0]
    assert isinstance(root, LeafNode) or isinstance(
        root, DecisionNode
    ), f"root is not a node but {type(root)}"
    queue = [root]
    i = 0
    while len(queue) > 0:
        cur_node = queue.pop()
        if isinstance(cur_node, DecisionNode):
            assert (
                cur_node.threshold == exp_val2[i]
            ), f"Expected threshold {exp_val2[i]}, got {cur_node.threshold}"
            assert (
                cur_node.split_idx == spl_idx2[i]
            ), f"Expected split idx {spl_idx2[i]}, got {cur_node.split_idx}"
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
            i += 1
        elif isinstance(cur_node, LeafNode):
            assert (
                len(cur_node.value) == 1
            ), f"Expected {1} mean values, but got: {len(cur_node.value)}"
    rec_node(root, 0)


def test_entropy_single():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    tree = DecisionTree("Classification", criteria=Entropy)
    tree.fit(X, Y_cla)
    root = tree.root
    exp_val = [0.25, -0.75, 0]
    spl_idx = [0, 0, 1]
    assert isinstance(root, LeafNode) or isinstance(
        root, DecisionNode
    ), f"root is not a node but {type(root)}"
    queue = [root]
    i = 0
    # Loop over all the nodes
    while len(queue) > 0:
        cur_node = queue.pop()
        if isinstance(
            cur_node, DecisionNode
        ):  # Check threshold and idx of decision node
            assert (
                cur_node.threshold == exp_val[i]
            ), f"Expected threshold {exp_val[i]} on node={i}, got {cur_node.threshold} on split_idx {cur_node.split_idx} exp: {spl_idx[i]}"
            assert (
                cur_node.split_idx == spl_idx[i]
            ), f"Expected split idx {spl_idx[i]} on i={i}, got {cur_node.split_idx}"
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
            i += 1
        elif isinstance(cur_node, LeafNode):  # Check that the value is of length 2
            assert (
                len(cur_node.value) == 2
            ), f"Expected 2 mean values, one for each class, but got: {len(cur_node.value)}"

    rec_node(root, 0)


def test_entropy_multi():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_multi = np.array([1, 2, 1, 0, 1, 0, 1, 0])
    Y_unique = len(np.unique(Y_multi))
    tree = DecisionTree("Classification", criteria=Entropy)
    tree.fit(X, Y_multi)
    root = tree.root
    # DIFFERENT FROM SKLEARN THEIRS IS: [0.25, -0.75, -1.5], both give pure
    # leaf node
    exp_val = [0.25, -0.75, -0.75]
    # DIFFERENT FROM SKLEARN THEIRS IS: [0, 1, 1], both give pure leaf node
    spl_idx = [0, 1, 0]
    assert isinstance(root, LeafNode) or isinstance(
        root, DecisionNode
    ), f"root is not a node but {type(root)}"
    queue = [root]
    i = 0
    while len(queue) > 0:
        cur_node = queue.pop()
        if isinstance(cur_node, DecisionNode):
            assert (
                cur_node.threshold == exp_val[i]
            ), f"Expected threshold {exp_val[i]}, got {cur_node.threshold}"
            assert (
                cur_node.split_idx == spl_idx[i]
            ), f"Expected split idx {spl_idx[i]}, got {cur_node.split_idx}"
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
            i += 1
        elif isinstance(cur_node, LeafNode):
            assert (
                len(cur_node.value) == Y_unique
            ), f"Expected {Y_unique} mean values, one for each class, but got: {len(cur_node.value)}"

    rec_node(root, 0)


def sanity_regression(n, m):
    X = np.random.uniform(0, 100, (n, m))
    Y1 = np.random.randint(0, 5, n)
    Y2 = np.random.uniform(0, 5, n)

    tree1 = DecisionTree("Regression", criteria=Squared_error)
    tree2 = DecisionTree("Regression", criteria=Squared_error)
    tree1.fit(X, Y1)
    tree2.fit(X, Y2)
    pred1 = tree1.predict(X)
    pred2 = tree2.predict(X)
    for i in range(n):
        assert (
            abs(Y1[i] - pred1[i]) < 0.00001
        ), f"Square: Expected {Y1[i]} Got {pred1[i]}"
        assert (
            abs(Y2[i] - pred2[i]) < 0.00001
        ), f"Square: Expected {Y2[i]} Got {pred2[i]}"


def sanity_gini(n, m):
    X = np.random.uniform(0, 100, (n, m))
    Y = np.random.randint(0, 5, n)

    tree = DecisionTree("Classification", criteria=Gini_index)
    tree.fit(X, Y)

    pred = tree.predict(X)
    for i in range(n):
        assert Y[i] == pred[i], f"Gini: Expected {Y[i]} Got {pred[i]}"


def sanity_entropy(n, m):
    X = np.random.uniform(0, 100, (n, m))
    Y = np.random.randint(0, 5, n)

    tree = DecisionTree("Classification", criteria=Entropy)
    tree.fit(X, Y)

    pred = tree.predict(X)
    for i in range(n):
        assert Y[i] == pred[i], f"Entropy: Expected {Y[i]} Got {pred[i]}"


def sanity_linear_regression(n, m):
    X = np.random.uniform(0, 100, (n, m))
    Y = np.random.uniform(0, 10, n)
    tree = DecisionTree("Regression", criteria=Linear_regression)
    tree.fit(X, Y)

    for node in tree.leaf_nodes:
        assert isinstance(node, LeafNode)
        # Linear regression fits on the X[:, 0] values,
        # so for the final node X[:, 0] should have a correlation of 1 with Y
        # in the node
        if node.indices.shape[0] > 1:
            corr = scipy.stats.pearsonr(X[node.indices, 0], Y[node.indices])[0]
            assert abs(corr) == 1.0


def test_sanity():
    n = 10000
    m = 5
    sanity_regression(n, m)
    sanity_gini(n, m)
    sanity_entropy(n, m)
    sanity_linear_regression(n, m)


if __name__ == "__main__":
    # test_gini_single()
    # test_gini_multi()
    # test_entropy_single()
    # test_entropy_multi()
    test_regression()
    # test_sanity()
    # print("Done.")
