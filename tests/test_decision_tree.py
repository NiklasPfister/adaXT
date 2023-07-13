from adaXT.decision_tree.tree import *
from adaXT.decision_tree.criteria import *
from adaXT.decision_tree.criteria import gini_index
from adaXT.decision_tree.tree_utils import print_tree

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
    print_tree(tree)
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
    print_tree(tree)
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

if __name__ == '__main__':
    test_regression()