from adaXT.decision_tree.tree import *
from adaXT.decision_tree.criteria import *
from adaXT.decision_tree.criteria import gini_index
from adaXT.decision_tree.tree_utils import print_tree

def rec_node(node: LeafNode|DecisionNode|None, depth):
    if issubclass(type(node), Node):
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
    while len(queue) > 0:
        cur_node = queue.pop()
        if type(cur_node) == DecisionNode:
            assert cur_node.threshold == exp_val[i], f'Expected threshold {exp_val[i]}, got {cur_node.threshold}'
            assert cur_node.split_idx == spl_idx[i], f'Expected split idx {spl_idx[i]}, got {cur_node.split_idx}'
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
        elif type(cur_node) == LeafNode:
            continue
        i += 1
    rec_node(root, 0)



if __name__ == '__main__':
    test_single_class()