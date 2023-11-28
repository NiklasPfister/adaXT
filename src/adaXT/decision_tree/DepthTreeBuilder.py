# General
import numpy as np
from typing import List

# Custom
from .splitter import Splitter
from .criteria import Criteria
from .Nodes import Node, DecisionNode, LeafNode
from . import DecisionTree


EPSILON = np.finfo('double').eps


class queue_obj:
    """
    Queue object for the splitter depthtree builder class
    """

    def __init__(
            self,
            indices: np.ndarray,
            depth: int,
            impurity: float,
            parent: Node | None = None,
            is_left: bool | None = None) -> None:
        """

        Parameters
        ----------
        indices : np.ndarray
            indicies waiting to be split
        depth : int
            depth of the node which is to be made
        impurity : float
            impurity in the node to be made
        parent : Node | None, optional
            parent of the node to be made, by default None
        is_left : bool | None, optional
            whether the object is left or right, by default None
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

    def __init__(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            feature_indices: np.ndarray,
            sample_indices: np.ndarray,
            criteria: Criteria,
            splitter: Splitter | None = None,
            min_impurity: float = 0.0,
            min_improvement: float = EPSILON,
            pre_sort: np.ndarray | None = None) -> None:
        """
        Parameters
        ----------
        X : np.ndarray
            The feature values
        Y : np.ndarray
            The response values
        feature_indices : np.ndarray
            Which features to use
        sample_indices : np.ndarray
            Indicies of the samples to use.
        criteria : FuncWrapper
            Criteria function used for impurity calculations, wrapped in FuncWrapper class
        splitter : Splitter | None, optional
            Splitter class used to split data, by default None
        min_impurity : float, optional
            Tolerance in impurity of leaf node, by default 0
        pre_sort : np.ndarray | None, optional
            Pre_sorted indicies in regards to features, by default None
        """
        self.features = X[np.ix_(sample_indices, feature_indices)]
        self.response = Y[sample_indices]
        self.feature_indices = feature_indices
        self.sample_indices = sample_indices
        self.criteria = criteria

        if splitter:
            self.splitter = splitter
        else:
            self.splitter = Splitter(self.features, self.response, criteria)

        if isinstance(pre_sort, np.ndarray):
            if pre_sort.dtype != np.int32:
                pre_sort = np.ascontiguousarray(pre_sort, np.int32)
            self.splitter.set_pre_sort(pre_sort)
        self.min_impurity = min_impurity
        self.min_improvement = min_improvement

    def get_mean(
            self,
            tree: object,
            node_response: np.ndarray,
            n_samples: int) -> List[float]:
        """
        Calculates the mean of a leafnode

        Parameters
        ----------
        tree : DecisionTree
            The fille tree object
        node_response : np.ndarray
            outcome values in the node
        n_samples : int
            number of samples in the node
        n_classes : int
            number of different classes in the node

        Returns
        -------
        List[float]
            A List of mean values for each class in the node
        """
        if tree.tree_type == "Regression":
            return [float(np.mean(node_response))]
        # create an empty List for each class type
        lst = [0.0 for _ in range(tree.n_classes)]
        for i in range(tree.n_classes):
            for idx in range(n_samples):
                if node_response[idx] == self.classes[i]:
                    lst[i] += 1  # add 1, if the value is the same as class value
            # weight by the number of total samples in the leaf
            lst[i] = lst[i] / n_samples
        return lst

    def build_tree(self, tree: DecisionTree):
        """
        Builds the tree

        Parameters
        ----------
        tree : DecisionTree
            the tree to build
        Returns
        -------
        int :
            returns 0 on succes
        """
        features = self.features
        response = self.response
        splitter = self.splitter
        criteria = self.criteria

        n_classes = 0
        if tree.tree_type == "Classification":
            self.classes = np.unique(response)
            tree.classes = self.classes
            n_classes = self.classes.shape[0]

        tree.n_classes = n_classes
        min_samples = tree.min_samples
        max_depth = tree.max_depth
        root = None

        leaf_node_list = []
        max_depth_seen = 0

        n_obs = len(response)
        queue = []  # queue of elements queue objects that need to be built

        # root node contains all indices
        all_idx = np.arange(n_obs, dtype=np.int32)
        queue.append(
            queue_obj(
                all_idx,
                0,
                criteria.impurity(all_idx)))
        n_nodes = 0
        leaf_count = 0  # Number of leaf nodes
        while len(queue) > 0:
            obj = queue.pop()
            indices, depth, impurity, parent, is_left = obj.indices, obj.depth, obj.impurity, obj.parent, obj.is_left
            n_samples = len(indices)
            # bool used to determine wheter a node is a leaf or not
            # additional stopping criteria can be added with 'or' statements
            # here
            is_leaf = ((depth >= max_depth) or
                       (abs(impurity - self.min_impurity) <= EPSILON) or
                       (n_samples <= min_samples))
            # Check improvement
            # if parent != None:
            #     is_leaf = (abs(parent.impurity - impurity) <= self.min_improvement) or is_leaf

            # TODO: possible impurity improvement tolerance.
            if depth > max_depth_seen:  # keep track of the max depth seen
                max_depth_seen = depth
            if not is_leaf:
                split, best_threshold, best_index, _, child_imp = splitter.get_split(
                    indices)

                # Add the decision node to the List of nodes
                new_node = DecisionNode(
                    indices,
                    depth,
                    impurity,
                    n_samples,
                    best_threshold,
                    best_index,
                    parent=parent)
                if is_left and parent:  # if there is a parent
                    parent.left_child = new_node
                elif parent:
                    parent.right_child = new_node

                left, right = split
                # Add the left node to the queue of nodes yet to be computed
                queue.append(
                    queue_obj(
                        left,
                        depth + 1,
                        child_imp[0],
                        new_node,
                        1))
                # Add the right node to the queue of nodes yet to be computed
                queue.append(
                    queue_obj(
                        right,
                        depth + 1,
                        child_imp[1],
                        new_node,
                        0))

            else:
                mean_value = self.get_mean(tree, response[indices], n_samples)
                new_node = LeafNode(
                    leaf_count,
                    indices,
                    depth,
                    impurity,
                    n_samples,
                    mean_value,
                    parent=parent)
                if is_left and parent:  # if there is a parent
                    parent.left_child = new_node
                elif parent:
                    parent.right_child = new_node
                leaf_node_list.append(new_node)
                leaf_count += 1
            if n_nodes == 0:
                root = new_node
            n_nodes += 1  # number of nodes increase by 1

        tree.n_nodes = n_nodes
        tree.max_depth = max_depth_seen
        tree.n_features = features.shape[0]
        tree.n_obs = n_obs
        tree.root = root
        tree.leaf_nodes = leaf_node_list
        return 0
