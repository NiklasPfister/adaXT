import numpy as np
import sys

cimport numpy as cnp
ctypedef cnp.float64_t DOUBLE_t
ctypedef cnp.int64_t LONG_t
from libcpp cimport bool


# Custom
from .splitter import Splitter
from ..predictor import Predictor
from ..criteria import Criteria
from ..leaf_builder import LeafBuilder
from .nodes import DecisionNode

# for c level definitions

cimport cython
from .nodes cimport DecisionNode, Node

from ..utils cimport dsum

cdef double EPSILON = np.finfo('double').eps

# Pseudo Node class, which will get replaced after refitting leaf nodes.
cdef class refit_object(Node):
    cdef public:
        list list_idx
        bint is_left

    def __init__(
            self,
            idx: int,
            depth: int,
            parent: DecisionNode,
            is_left: bool) -> None:

        self.list_idx = [idx]
        self.depth = depth
        self.parent = parent
        self.is_left = is_left

    def add_idx(self, idx: int) -> None:
        self.list_idx.append(idx)

cdef class _DecisionTree:
    cdef public:
        object criteria
        object splitter
        object predictor
        object leaf_builder
        object leaf_nodes, predictor_instance, root
        long max_depth, min_samples_leaf, max_features
        long min_samples_split, n_nodes, n_features
        long n_rows_fit, n_rows_predict, X_n_rows
        float impurity_tol, min_improvement

    def __init__(
            self,
            criteria: type[Criteria],
            leaf_builder: type[LeafBuilder],
            predictor: type[Predictor],
            splitter: type[Splitter],
            max_depth: long = sys.maxsize,
            impurity_tol: float = 0.0,
            min_samples_split: int = 1,
            min_samples_leaf: int = 1,
            min_improvement: float = 0.0,
            max_features: int | float | Literal["sqrt", "log2"] | None = None) -> None:

        self.criteria = criteria
        self.predictor = predictor
        self.leaf_builder = leaf_builder
        self.splitter = splitter
        self.max_features = max_features

        self.max_depth = max_depth
        self.impurity_tol = impurity_tol
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_improvement = min_improvement

        self.leaf_nodes = None
        self.predictor_instance = None
        self.root = None
        self.n_nodes = -1
        self.n_features = -1

    def predict(self, double[:, ::1] X, **kwargs) -> np.ndarray:
        return self.predictor_instance.predict(X, **kwargs)

    cdef dict __get_leaf(self, bool scale = False):
        if self.root is None:
            raise ValueError("The tree has not been trained before trying to predict")

        leaf_nodes = self.leaf_nodes
        if (not leaf_nodes):  # make sure that there are calculated observations
            raise ValueError("The tree has no leaf nodes")

        ht = {}
        for node in leaf_nodes:
            ht[node.id] = node.indices
        return ht

    def predict_weights(self, X: np.ndarray | None = None,
                        bool scale = True) -> np.ndarray:
        if X is None:
            size_0 = self.n_rows_predict
            new_hash_table = self.__get_leaf()
        else:
            size_0 = X.shape[0]
            new_hash_table = self.predict_leaf(X)
        if scale:
            scaling = "row"
        else:
            scaling = "none"
        default_hash_table = self.__get_leaf()
        return self._tree_based_weights(new_hash_table, default_hash_table,
                                        size_0, self.n_rows_predict,
                                        scaling=scaling)

    def predict_leaf(self, X: np.ndarray | None = None) -> dict:
        if X is None:
            return self.__get_leaf()
        if self.predictor_instance is None:
            raise ValueError("The tree has not been trained before trying to predict")
        return self.predictor_instance.predict_leaf(X)

    def _tree_based_weights(self, hash0: dict, hash1: dict, size_X0: int,
                            size_X1: int, scaling: str) -> np.ndarray:
        cdef:
            int xi, ind2
            cnp.ndarray[DOUBLE_t, ndim=2] matrix

        matrix = np.zeros((size_X0, size_X1))
        hash0_keys = hash0.keys()
        hash1_keys = hash1.keys()
        for xi in hash0_keys:
            if xi in hash1_keys:
                indices_1 = hash0[xi]
                indices_2 = hash1[xi]
                if scaling == "row":
                    val = 1.0/len(indices_2)
                    for ind2 in indices_2:
                        matrix[indices_1, ind2] += val
                elif scaling == "similarity":
                    val = 1.0
                    matrix[np.ix_(indices_1, indices_2)] = val
                elif scaling == "none":
                    val = 1.0
                    for ind2 in indices_2:
                        matrix[indices_1, ind2] += val
        return matrix

    def similarity(self, cnp.ndarray[DOUBLE_t, ndim=2] X0,
                   cnp.ndarray[DOUBLE_t, ndim=2] X1):
        hash0 = self.predict_leaf(X0)
        hash1 = self.predict_leaf(X1)
        return self._tree_based_weights(hash0, hash1, X0.shape[0], X1.shape[0],
                                        scaling="similarity")

    cdef void __remove_leaf_nodes(self):
        cdef:
            int i, n_nodes
            object parent
        n_nodes = len(self.leaf_nodes)
        for i in range(n_nodes):
            parent = self.leaf_nodes[i].parent
            if parent is None:
                self.root = None
            elif parent.left_child == self.leaf_nodes[i]:
                parent.left_child = None
            else:
                parent.right_child = None
            self.leaf_nodes[i] = None

    cdef void __fit_new_leaf_nodes(self, cnp.ndarray[DOUBLE_t, ndim=2] X,
                                   cnp.ndarray[DOUBLE_t, ndim=2] Y,
                                   cnp.ndarray[DOUBLE_t, ndim=1] sample_weight,
                                   cnp.ndarray[LONG_t, ndim=1] sample_indices):
        cdef:
            int idx, n_objs, depth, cur_split_idx
            double cur_threshold
            object cur_node
            int[::1] all_idx
            int[::1] leaf_indices

        # Set all_idx to contain only sample_indices with a positive weight
        all_idx = np.array(
            [x for x in sample_indices if sample_weight[x] != 0], dtype=np.int32
        )

        if self.root is not None:
            refit_objs = []
            for idx in all_idx:
                cur_node = self.root
                depth = 0
                while isinstance(cur_node, DecisionNode) :
                    # Mark cur_node as visited
                    cur_node.visited = 1
                    cur_split_idx = cur_node.split_idx
                    cur_threshold = cur_node.threshold

                    # Check if X should go to the left or right
                    if X[idx, cur_split_idx] < cur_threshold:
                        # If the left or right is none, then there previously was a
                        # leaf node, and we create a new refit object
                        if cur_node.left_child is None:
                            cur_node.left_child = refit_object(idx, depth,
                                                               cur_node, True)
                            refit_objs.append(cur_node.left_child)
                        # If there already is a refit object, add this new index to
                        # the refit object
                        elif isinstance(cur_node.left_child, refit_object):
                            cur_node.left_child.add_idx(idx)

                        cur_node = cur_node.left_child
                    else:
                        if cur_node.right_child is None:
                            cur_node.right_child = refit_object(idx, depth,
                                                                cur_node, False)
                            refit_objs.append(cur_node.right_child)
                        elif isinstance(cur_node.right_child, refit_object):
                            cur_node.right_child.add_idx(idx)

                        cur_node = cur_node.right_child
                    depth += 1

        leaf_builder_instance = self.leaf_builder(X, Y, all_idx)
        criteria_instance = self.criteria(X, Y, sample_weight)
        # Make refit objects into leaf_nodes
        # Two cases:
        # (1) Only a single root node (n_objs == 0)
        # (2) At least one split (n_objs > 0)
        if self.root is None:
            weighted_samples = dsum(sample_weight, all_idx)
            self.root = leaf_builder_instance.build_leaf(
                    leaf_id=0,
                    indices=all_idx,
                    depth=0,
                    impurity=criteria_instance.impurity(all_idx),
                    weighted_samples=weighted_samples,
                    parent=None)
            self.leaf_nodes = [self.root]
        else:
            n_objs = len(refit_objs)
            nodes = []
            for i in range(n_objs):
                obj = refit_objs[i]
                leaf_indices = np.array(obj.list_idx, dtype=np.int32)
                weighted_samples = dsum(sample_weight, leaf_indices)
                new_node = leaf_builder_instance.build_leaf(
                        leaf_id=i,
                        indices=leaf_indices,
                        depth=obj.depth,
                        impurity=criteria_instance.impurity(leaf_indices),
                        weighted_samples=weighted_samples,
                        parent=obj.parent,
                        )
                new_node.visited = 1
                nodes.append(new_node)
                if obj.is_left:
                    obj.parent.left_child = new_node
                else:
                    obj.parent.right_child = new_node
            self.leaf_nodes = nodes

    # Assumes that each visited node is marked during __fit_new_leaf_nodes
    cdef void __squash_tree(self):

        decision_queue = []
        decision_queue.append(self.root)
        while len(decision_queue) > 0:
            cur_node = decision_queue.pop(0)
            # If we don't have a decision node, just continue
            if not isinstance(cur_node, DecisionNode):
                continue

            # If left child was not visited, then squash current node and right
            # child
            if (cur_node.left_child is None) or (cur_node.left_child.visited ==
                                                 0):
                parent = cur_node.parent
                # Root node
                if parent is None:
                    self.root = cur_node.right_child
                # if current node is left child
                elif parent.left_child == cur_node:
                    # update parent to point to the child that has been visited
                    # instead
                    parent.left_child = cur_node.right_child
                else:
                    parent.right_child = cur_node.right_child

                cur_node.right_child.parent = parent

                # Only add this squashed child to the queue
                decision_queue.append(cur_node.right_child)

            # Same for the right
            elif (cur_node.right_child is None) or (cur_node.right_child.visited
                                                    == 0):
                parent = cur_node.parent
                # Root node
                if parent is None:
                    self.root = cur_node.left_child
                # if current node is left child
                elif parent.left_child == cur_node:
                    # update parent to point to the child that has been visited
                    # instead
                    parent.left_child = cur_node.left_child
                else:
                    parent.right_child = cur_node.left_child

                cur_node.left_child.parent = parent

                # Only add this squashed child to the queue
                decision_queue.append(cur_node.left_child)
            else:
                # Neither need squashing, add both to the queue
                decision_queue.append(cur_node.left_child)
                decision_queue.append(cur_node.right_child)

    def refit_leaf_nodes(self,
                         cnp.ndarray[DOUBLE_t, ndim=2] X,
                         cnp.ndarray[DOUBLE_t, ndim=2] Y,
                         cnp.ndarray[DOUBLE_t, ndim=1] sample_weight,
                         cnp.ndarray[LONG_t, ndim=1] sample_indices) -> None:

        if self.root is None:
            raise ValueError("The tree has not been trained before trying to\
                             refit leaf nodes")

        # Remove current leaf nodes
        self.__remove_leaf_nodes()

        # Find the leaf node, all samples would have been placed in
        self.__fit_new_leaf_nodes(X, Y, sample_weight, sample_indices)

        # Now squash all the DecisionNodes not visited
        self.__squash_tree()

        # Make sure that predictor_instance points to the same root, if we have
        # changed it
        self.predictor_instance.root = self.root


# From below here, it is the DepthTreeBuilder
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
        is_left: bool | None = None,
    ) -> None:
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
        max_features: int | None,
        sample_weight: np.ndarray,
        sample_indices: np.ndarray | None,
        criteria: Criteria,
        splitter: Splitter,
        leaf_builder: LeafBuilder,
        predictor: Predictor,
    ) -> None:
        """
        Parameters
        ----------
        X : np.ndarray
            The feature values
        Y : np.ndarray
            The response values
        max_featueres : int | None
            Max number of features in a leaf node
        sample_weight : np.ndarray
            The weight of all samples
        sample_indices : np.ndarray
            The sample indices to use of the total data
        criteria : Criteria
            Criteria class used for impurity calculations
        splitter : Splitter | None, optional
            Splitter class used to split data, by default None
        leaf_builder : LeafBuilder
            The LeafBuilder class to use
        predictor
            The Predictor class to use
        """
        self.X = X
        self.Y = Y
        self.sample_indices = sample_indices
        self.sample_weight = sample_weight
        self.max_features = max_features

        self.splitter = splitter
        self.criteria = criteria
        self.predictor = predictor
        self.leaf_builder = leaf_builder

    def __get_feature_indices(self) -> np.ndarray:
        if self.max_features == -1:
            return self.feature_indices
        else:
            return np.random.choice(
                self.feature_indices,
                size=self.max_features,
                replace=False)

    def build_tree(self, tree: _DecisionTree) -> None:
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
        # initialization
        _, col = self.X.shape
        self.max_features = tree.max_features

        self.feature_indices = np.arange(col, dtype=np.int32)

        criteria_instance = self.criteria(self.X, self.Y, self.sample_weight)
        splitter_instance = self.splitter(self.X, self.Y, criteria_instance)

        min_samples_split = tree.min_samples_split
        min_samples_leaf = tree.min_samples_leaf
        max_depth = tree.max_depth
        impurity_tol = tree.impurity_tol
        min_improvement = tree.min_improvement

        root = None

        leaf_node_list = []
        max_depth_seen = 0

        queue = []  # queue for objects that need to be built

        all_idx = np.array([
            x for x in self.sample_indices if self.sample_weight[x] != 0
        ], dtype=np.int32)

        # Update the tree now that we have the correct samples
        leaf_builder_instance = self.leaf_builder(self.X, self.Y, all_idx)
        weighted_total = dsum(self.sample_weight, all_idx)

        queue.append(queue_obj(all_idx, 0, criteria_instance.impurity(all_idx)))
        n_nodes = 0
        leaf_count = 0  # Number of leaf nodes
        while len(queue) > 0:
            obj = queue.pop()
            indices, depth, impurity, parent, is_left = (
                obj.indices,
                obj.depth,
                obj.impurity,
                obj.parent,
                obj.is_left,
            )
            weighted_samples = dsum(self.sample_weight, indices)
            # Stopping Conditions - BEFORE:
            # boolean used to determine wheter 'current node' is a leaf or not
            # additional stopping criteria can be added with 'or' statements
            is_leaf = (
                (depth >= max_depth)
                or (impurity <= impurity_tol + EPSILON)
                or (weighted_samples <= min_samples_split)
            )

            if depth > max_depth_seen:  # keep track of the max depth seen
                max_depth_seen = depth

            # If it is not a leaf, find the best split
            if not is_leaf:
                split, best_threshold, best_index, _, child_imp = splitter_instance.get_split(
                    indices, self.__get_feature_indices()
                )
                # If we were unable to find a split, this must be a leaf.
                if len(split) == 0:
                    is_leaf = True
                else:
                    # Stopping Conditions - AFTER:
                    # boolean used to determine wheter 'parent node' is a leaf or not
                    # additional stopping criteria can be added with 'or'
                    # statements
                    weight_left = dsum(
                        self.sample_weight, split[0]
                    )
                    weight_right = dsum(
                        self.sample_weight, split[1]
                    )
                    is_leaf = (
                        (
                            weighted_samples
                            / weighted_total
                            * (
                                impurity
                                - (weight_left / weighted_samples) * child_imp[0]
                                - (weight_right / weighted_samples) * child_imp[1]
                            )
                            < min_improvement + EPSILON
                        )
                        or (weight_left < min_samples_leaf)
                        or (weight_right < min_samples_leaf)
                    )

            if not is_leaf:
                # Add the decision node to the List of nodes
                new_node = DecisionNode(
                    indices=indices,
                    depth=depth,
                    impurity=impurity,
                    threshold=best_threshold,
                    split_idx=best_index,
                    parent=parent,
                )
                if is_left and parent:  # if there is a parent
                    parent.left_child = new_node
                elif parent:
                    parent.right_child = new_node

                left, right = split
                # Add the left node to the queue of nodes yet to be computed
                queue.append(queue_obj(left, depth + 1,
                             child_imp[0], new_node, 1))
                # Add the right node to the queue of nodes yet to be computed
                queue.append(queue_obj(right, depth + 1,
                             child_imp[1], new_node, 0))

            else:
                new_node = leaf_builder_instance.build_leaf(
                        leaf_id=leaf_count,
                        indices=indices,
                        depth=depth,
                        impurity=impurity,
                        weighted_samples=weighted_samples,
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
        tree.root = root
        tree.leaf_nodes = leaf_node_list
        tree.predictor_instance = self.predictor(self.X, self.Y, root)
