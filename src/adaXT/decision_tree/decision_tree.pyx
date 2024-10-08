# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False


# General
import numpy as np
import sys
from numpy.typing import ArrayLike

cimport numpy as cnp
ctypedef cnp.float64_t DOUBLE_t


# Custom
from .splitter import Splitter
from ..predict import Predict
from ..criteria import Criteria
from .nodes import DecisionNode
from ..leaf_builder import LeafBuilder
from ..base_model import BaseModel

cdef double EPSILON = np.finfo('double').eps


class refit_object():
    def __init__(
            self,
            idx: int,
            depth: int,
            parent: DecisionNode,
            is_left: bool) -> None:

        self.indices = [idx]
        self.depth = depth
        self.parent = parent
        self.is_left = is_left

    def add_idx(self, idx: int) -> None:
        self.indices.append(idx)


class DecisionTree(BaseModel):
    def __init__(
            self,
            tree_type: str | None = None,
            skip_check_input: bool = False,
            max_depth: int = sys.maxsize,
            impurity_tol: float = 0,
            min_samples_split: int = 1,
            min_samples_leaf: int = 1,
            min_improvement: float = 0,
            max_features: int | float | Literal["sqrt", "log2"] | None = None,
            criteria: Criteria | None = None,
            leaf_builder: LeafBuilder | None = None,
            predict: Predict | None = None,
            splitter: Splitter | None = None) -> None:

        if skip_check_input:
            self.criteria_class = criteria
            self.predict_class = predict
            self.leaf_builder_class = leaf_builder
            self.splitter_class = splitter
            self.max_features = max_features
        else:
            self._check_tree_type(tree_type, criteria, splitter, leaf_builder, predict)
            self.max_features = self._check_max_features(max_features)

        self.skip_check_input = skip_check_input
        self.max_depth = max_depth
        self.impurity_tol = impurity_tol
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_improvement = min_improvement
        self.tree_type = tree_type
        self.leaf_nodes = None
        self.root = None
        self.predictor = None
        self.n_nodes = -1
        self.n_features = -1

    def fit(self,
            X: ArrayLike,
            Y: ArrayLike,
            sample_indices: ArrayLike | None = None,
            sample_weight: ArrayLike | None = None) -> None:

        # Check inputs
        if not self.skip_check_input:
            X, Y = self._check_input(X, Y)

        # These values are used when checking sample_indices and sample_weight,
        # so they have to be updated after checking X and Y
        self.n_rows_fit = X.shape[0]
        self.n_rows_predict = X.shape[0]
        self.X_n_rows = X.shape[0]
        self.n_features = X.shape[1]

        if not self.skip_check_input:
            sample_weight = self._check_sample_weight(sample_weight=sample_weight)
            sample_indices = self._check_sample_indices(sample_indices=sample_indices)

        builder = DepthTreeBuilder(
            X=X,
            Y=Y,
            sample_indices=sample_indices,
            max_features=self.max_features,
            sample_weight=sample_weight,
            criteria_class=self.criteria_class,
            leaf_builder_class=self.leaf_builder_class,
            predict_class=self.predict_class,
            splitter_class=self.splitter_class)
        builder.build_tree(self)

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        if not self.predictor:
            raise AttributeError("The tree has not been fitted before trying to call predict")
        if not self.skip_check_input:
            X, _ = self._check_input(X)
            self._check_dimensions(X)
        return self.predictor.predict(X, **kwargs)

    def __get_leaf(self, scale: bool = False) -> dict:
        if self.root is None:
            raise ValueError("The tree has not been trained before trying to predict")

        leaf_nodes = self.leaf_nodes
        if (not leaf_nodes):  # make sure that there are calculated observations
            raise ValueError("The tree has no leaf nodes")

        ht = {}
        for node in leaf_nodes:
            ht[node.id] = node.indices
        return ht

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

    def similarity(self, X0: ArrayLike, X1: ArrayLike):
        if not self.skip_check_input:
            X0, _ = self._check_input(X0)
            self._check_dimensions(X0)
            X1, _ = self._check_input(X1)
            self._check_dimensions(X1)

        hash0 = self.predict_leaf(X0)
        hash1 = self.predict_leaf(X1)
        return self._tree_based_weights(hash0, hash1, X0.shape[0], X1.shape[0],
                                        scaling="similarity")

    def predict_weights(self, X: ArrayLike | None = None,
                        scale: bool = True) -> np.ndarray:
        if X is None:
            size_0 = self.n_rows_predict
            new_hash_table = self.__get_leaf()
        else:
            if not self.skip_check_input:
                X, _ = self._check_input(X)
                self._check_dimensions(X)
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

    def predict_leaf(self, X: ArrayLike | None = None) -> dict:
        if X is None:
            return self.__get_leaf()
        else:
            if not self.skip_check_input:
                X, _ = self._check_input(X)
                self._check_dimensions(X)
        if not self.predictor:
            raise ValueError("The tree has not been trained before trying to predict")
        return self.predictor.predict_leaf(X)

    def __remove_leaf_nodes(self) -> None:
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

    def __fit_new_leaf_nodes(self, X: np.ndarray, Y: np.ndarray, sample_weight:
                             np.ndarray, sample_indices: np.ndarray) -> None:
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

        leaf_builder = self.leaf_builder_class(X, Y, all_idx)
        criteria = self.criteria_class(X, Y, sample_weight)
        # Make refit objects into leaf_nodes
        # Two cases:
        # (1) Only a single root node (n_objs == 0)
        # (2) At least one split (n_objs > 0)
        if self.root is None:
            weighted_samples = np.sum([sample_weight[x] for x in all_idx])
            self.root = leaf_builder.build_leaf(
                    leaf_id=0,
                    indices=all_idx,
                    depth=0,
                    impurity=criteria.impurity(all_idx),
                    weighted_samples=weighted_samples,
                    parent=None)
            self.leaf_nodes = [self.root]
        else:
            n_objs = len(refit_objs)
            nodes = []
            for i in range(n_objs):
                obj = refit_objs[i]
                leaf_indices = np.array(obj.indices, dtype=np.int32)
                weighted_samples = np.sum([sample_weight[x] for x in leaf_indices])
                new_node = leaf_builder.build_leaf(
                        leaf_id=i,
                        indices=leaf_indices,
                        depth=obj.depth,
                        impurity=criteria.impurity(leaf_indices),
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
    def __squash_tree(self) -> None:

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
                         X: ArrayLike,
                         Y: ArrayLike,
                         sample_weight: ArrayLike | None = None,
                         sample_indices: ArrayLike | None = None,
                         **kwargs) -> None:
        if self.root is None:
            raise ValueError("The tree has not been trained before trying to\
                             refit leaf nodes")
        if not self.skip_check_input:
            X, Y = self._check_input(X, Y)
            self._check_dimensions(X)
            sample_weight = self._check_sample_weight(sample_weight)
            sample_indices = self._check_sample_indices(sample_indices)

        # Remove current leaf nodes
        self.__remove_leaf_nodes()

        # Find the leaf node, all samples would have been placed in
        self.__fit_new_leaf_nodes(X, Y, sample_weight, sample_indices)

        # Now squash all the DecisionNodes not visited
        self.__squash_tree()
        return


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
        criteria_class: Criteria,
        splitter_class: Splitter,
        leaf_builder_class: LeafBuilder,
        predict_class: Predict,
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
        criteria_class : Criteria
            Criteria class used for impurity calculations
        splitter_class : Splitter | None, optional
            Splitter class used to split data, by default None
        leaf_builder_class : LeafBuilder
            The LeafBuilder class to use
        predict_class
            The Predict class to use
        """
        self.X = X
        self.Y = Y
        self.sample_indices = sample_indices
        self.sample_weight = sample_weight

        _, col = X.shape
        self.int_max_features = self.__parse_max_features(max_features)

        self.feature_indices = np.arange(col, dtype=np.int32)
        self.num_features = col

        self.criteria = criteria_class(self.X, self.Y, self.sample_weight)
        self.splitter = splitter_class(self.X, self.Y, self.criteria)

        # These can not yet be initialized, as they depend on the all_idx
        # parameter calculated in build_tree
        self.predict_class = predict_class
        self.leaf_builder_class = leaf_builder_class

    def __get_feature_indices(self) -> np.ndarray:
        if self.int_max_features is None:
            return self.feature_indices
        else:
            return np.random.choice(
                self.feature_indices,
                size=self.int_max_features,
                replace=False)

    def __parse_max_features(self,
                             max_features: int|str|float|None
                             ) -> int:

        if max_features is None:
            return None
        elif isinstance(max_features, int):
            return min(max_features, self.num_features)
        elif isinstance(max_features, float):
            return min(self.num_features, int(max_features * self.num_features))
        elif isinstance(max_features, str):
            if max_features == "sqrt":
                return int(np.sqrt(self.num_features))
            elif max_features == "log2":
                return int(np.log2(self.num_features))
        else:
            raise ValueError("Unable to parse max_features")

    def build_tree(self, tree: DecisionTree) -> None:
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
        X = self.X
        Y = self.Y
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

        all_idx = np.array(
            [x for x in self.sample_indices if self.sample_weight[x] != 0], dtype=np.int32
        )

        # Update the tree now that we have the correct samples
        leaf_builder = self.leaf_builder_class(X, Y, all_idx)
        weighted_total = np.sum(self.sample_weight)

        queue.append(queue_obj(all_idx, 0, criteria.impurity(all_idx)))
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
            weighted_samples = np.sum([self.sample_weight[x] for x in indices])
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
                split, best_threshold, best_index, _, child_imp = splitter.get_split(
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
                    weight_left = np.sum(list(map(lambda x:
                                                  self.sample_weight[x],
                                                  split[0])))
                    weight_right = np.sum(list(map(lambda x:
                                                   self.sample_weight[x],
                                                   split[1])))
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
                new_node = leaf_builder.build_leaf(
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
        tree.predictor = self.predict_class(self.X, self.Y, root)
