# General
import numpy as np
from typing import List

# Custom
from .splitter import Splitter
from ..criteria import Criteria
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
            sample_weight: np.ndarray,
            criteria: Criteria,
            splitter: Splitter | None = None,
            sample_indices: np.ndarray | None = None) -> None:
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
        criteria : Criteria
            Criteria class used for impurity calculations
        splitter : Splitter | None, optional
            Splitter class used to split data, by default None
        """
        self.features = X
        self.response = Y
        self.feature_indices = feature_indices
        self.sample_indices = sample_indices
        self.criteria = criteria
        self.sample_weight = sample_weight

        if splitter:
            self.splitter = splitter
        else:
            self.splitter = Splitter(self.features, self.response, criteria)

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
            The filled tree object
        node_response : np.ndarray
            response values in the node
        n_samples : int
            number of samples in the node

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

        min_samples_split = tree.min_samples_split
        min_samples_leaf = tree.min_samples_leaf
        max_depth = tree.max_depth
        impurity_tol = tree.impurity_tol
        min_improvement = tree.min_improvement

        root = None

        leaf_node_list = []
        max_depth_seen = 0

        queue = []  # queue for objects that need to be built

        all_idx = np.arange(features.shape[0])
        if self.sample_indices is not None:
            all_idx = np.array(self.sample_indices)

        all_idx = np.array(
            [x for x in all_idx if self.sample_weight[x] != 0], dtype=np.int32)

        # Update the tree now that we have the correct samples
        n_classes = 0
        if tree.tree_type == "Classification":
            self.classes = np.unique(response[all_idx])
            tree.classes = self.classes
            n_classes = self.classes.shape[0]
        tree.n_classes = n_classes

        n_obs = all_idx.shape[0]

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
            # Stopping Conditions - BEFORE:
            # boolean used to determine wheter 'current node' is a leaf or not
            # additional stopping criteria can be added with 'or' statements
            is_leaf = ((depth >= max_depth) or
                       (impurity <= impurity_tol + EPSILON) or
                       (n_samples <= min_samples_split))

            if depth > max_depth_seen:  # keep track of the max depth seen
                max_depth_seen = depth

            # If it is not a leaf, find the best split
            if not is_leaf:
                split, best_threshold, best_index, _, child_imp = splitter.get_split(
                    indices, self.feature_indices)
                # If we were unable to find a split, this must be a leaf.
                if len(split) == 0:
                    is_leaf = True
                else:
                    # Stopping Conditions - AFTER:
                    # boolean used to determine wheter 'parent node' is a leaf or not
                    # additional stopping criteria can be added with 'or'
                    # statements
                    N_t_L = len(split[0])
                    N_t_R = len(split[1])
                    is_leaf = ((n_samples / n_obs *
                                (impurity -
                                 (N_t_L / n_samples) * child_imp[0] -
                                    (N_t_R / n_samples) * child_imp[1]) <
                                min_improvement + EPSILON) or
                               (N_t_L < min_samples_leaf) or
                               (N_t_R < min_samples_leaf))

            if not is_leaf:
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
        tree.n_features = features.shape[1]
        tree.n_obs = n_obs
        tree.root = root
        tree.leaf_nodes = leaf_node_list

        return 0
