from . import criteria
from . import splitter
from typing import Callable
import numpy as np
import numpy.typing as npt


crit = criteria.gini_index # default criteria function


class Node: # should just be a ctype struct in later implementation
    """
    Node in a tree

    Attributes 
    ----------
    indices : list[int]
        indices within the data, which are apart of the node
    depth : int
        depth of the node
    parent : Node, optional
        parent node, by default None
    impurity : float
        impurity of the node
    n_samples : int
        number of samples within the node
    threshold : float, optional
        threshold value of a decision node, by default None
    value : float, optional
        mean outcome value of datapoints in leaf node, by default None
    left_child : Node
        left child, by default None
    right_child : Node
        right child, by default None
    """
    def __init__(self, indices: list[int], depth: int, is_leaf: bool, impurity: float, n_samples: int, parent = None, threshold: float|None = None, split_idx: int|None = None, value: float|None = None) -> None:
        """
        Parameters
        ----------
        indices : list[int]
            indices within the data, which are apart of the node
        depth : int
            depth of the node
        is_leaf : bool
            whether or not the node is a leafnode
        parent : Node, optional
            parent node, by default None
        threshold : float, optional
            threshold value of a decision node, by default None
        split_idx : int
            index which to split on if it is a decision node, by default None
        value : float, optional
            mean outcome value of datapoints in leaf node, by default None
        """
        self.indices = indices # indices of values within the node
        self.depth = depth
        self.is_leaf = is_leaf
        self.impurity = impurity
        self.parent = parent
        self.n_samples = n_samples
        self.threshold = threshold # None for leaf nodes.
        self.split_idx = split_idx # None for leaf nodes.
        self.value = value # None for decision nodes
        self.left_child = None
        self.right_child = None

class Tree:
    """
    Tree object built by the tree builder class
    """
    def __init__(self, max_depth: int, root: Node|None = None, n_nodes: int|None = None, n_features: int|None = None, n_obs: int|None = None) -> None:
        """
        Parameters
        ----------
        root : Node
            root node of the tree
        n_nodes : int
            number of nodes in the tree
        n_features : int
            number of features of the data
        n_obs : int
            number of observations in the data
        max_depth : int
            maximum depth of the tree
        list_leaf_nodes : list[Node]
            list of leaf nodes #TODO not yet implemented
        """
        self.root = root
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.n_obs = n_obs
        self.max_depth = max_depth
    
    def print_tree(self):
        queue = []
        queue.append(self.root)
        while len(queue) > 0:
            node = queue.pop()
            if node:
                print(f"Depth: {node.depth}")
                print(f"Impurity: {node.impurity}")
                print(f"samples: {node.n_samples}")
                if node.is_leaf:
                    print(f"LEAF WITH VAL: {node.value}")
                else:
                    print(f"Decision WITH x{node.split_idx} <= {node.threshold}")
                print("") # spacing
                queue.append(node.left_child)
                queue.append(node.right_child)

    #TODO: prediction of which leaf a new sample would end up in
    #TODO: Implement NxN matrix which weighs the data in nodes, for example x_ij is true if index i and j are in the same leaf node(theory currently) seperate function.

class queue_obj:
    def __init__(self, indices : list, depth : int, impurity: float, parent: Node|None = None, is_left : int|None = None) -> None:
        """
        queue object

        Parameters
        ----------
        indices : list
            indices used to calculate node
        depth : int
            depth of the computed node
        idx : int
            index of where the node should be saved in the final nodes of the tree
        parent : Node, optional
            parent of the computed node, by default None
        is_left : int
        """
        self.indices = indices
        self.depth = depth
        self.impurity = impurity
        self.parent = parent
        self.is_left = is_left
class DepthTreeBuilder:
    """
    Depth first tree builder
    """
    def __init__(self, X: npt.NDArray, Y: npt.NDArray, criterion: Callable | None = None, tol : float = 1e-9) -> None:
        """
        Parameters
        ----------
        data : np.dtype
            data used to create the tree
        splitter : Splitter
            the splitter class used to split the data
        max_depth : int
            the maximum depth of the tree
        criterion : Callable
            the function to calculate the criteria used for splitting, 
            by default the one specied at the top of the file.
        tol : float
            tolerance for impurity of leaf nodes
        """
        self.features = X
        self.outcomes = Y
        self.criteria = crit
        if criterion:
            self.criteria = criterion
        self.splitter = splitter.Splitter(X, Y, self.criteria)
        self.tol = tol
    
    def build_tree(self, tree: Tree) -> Tree:
        """
        Builds the tree 

        Parameters
        ----------
        tree : Tree
            the tree to build

        Returns
        -------
        Tree
            the tree object built
        """
        splitter = self.splitter
        max_depth = tree.max_depth
        criteria = self.criteria
        features  = self.features
        outcomes = self.outcomes
        root = None
    
        max_depth_seen = 0
        
        n_obs = len(outcomes)
        queue = [] # built of lists, where each list is the indices of samples in a given node
        
        all_idx = [*range(n_obs)] # root node contains all indices
        queue.append(queue_obj(all_idx, 0, criteria(features[all_idx], outcomes[all_idx])))
        n_nodes = 0
        while len(queue) > 0:
            obj = queue.pop()
            indices, depth, impurity, parent, is_left = obj.indices, obj.depth, obj.impurity, obj.parent, obj.is_left
            is_leaf = (depth >= max_depth or impurity <= self.tol)

            if depth > max_depth_seen: # keep track of the max depth seen
                max_depth_seen = depth

            if not is_leaf:
                split, best_threshold, best_index, best_score, chil_imp = splitter.get_split(indices)
                # Add the decision node to the list of nodes
                new_node = Node(indices, depth, is_leaf, impurity, len(indices), parent, best_threshold, best_index)
                if is_left and parent: # if there is a parent
                    parent.left_child = new_node
                elif parent:
                    parent.right_child = new_node

                left, right = split
                # Add the left node to the queue of nodes yet to be computed
                queue.append(queue_obj(left, depth+1, chil_imp[0], new_node, 1))
                # Add the right node to the queue of nodes yet to be computed
                queue.append(queue_obj(right, depth+1, chil_imp[1], new_node, 0))
            else:
                mean_value = float(np.mean(outcomes[indices])) # calculate the mean outcome value of the nodes in the leaf
                new_node = Node(indices, depth, is_leaf, impurity, len(indices), parent, value = mean_value)
                if is_left and parent: # if there is a parent
                    parent.left_child = new_node
                elif parent:
                    parent.right_child = new_node
            if n_nodes == 0:
                root = new_node
            n_nodes += 1 # number of nodes increase by 1

        tree.n_nodes = n_nodes
        tree.max_depth = max_depth_seen
        tree.n_features = splitter.n_features
        tree.n_obs = n_obs
        tree.root = root
        return tree

    

    