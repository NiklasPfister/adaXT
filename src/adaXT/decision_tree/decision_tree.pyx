# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

# General
import numpy as np
from numpy import float64 as DOUBLE
import sys

# Custom
from .splitter import Splitter
from ..predict import Predict
from ..criteria import Criteria
from .nodes import DecisionNode
from ..leaf_builder import LeafBuilder
from ..base_model import BaseModel


cdef double EPSILON = np.finfo('double').eps


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

        # Function defined in BaseModel
        if skip_check_input:
            self.criteria_class = criteria
            self.predict_class = predict
            self.leaf_builder_class = leaf_builder
            self.splitter = splitter
        else:
            self.check_tree_type(tree_type, criteria, splitter, leaf_builder, predict)

        self.max_depth = max_depth
        self.impurity_tol = impurity_tol
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_improvement = min_improvement
        self.max_features = self.__error_check_max_features(max_features)
        self.tree_type = tree_type
        self.leaf_nodes = None
        self.root = None
        self.predictor = None
        self.n_nodes = -1
        self.n_features = -1
        self.n_obs = -1
        self.skip_check_input = skip_check_input

    def __error_check_max_features(self, max_features):
        if max_features is None:
            return max_features
        elif isinstance(max_features, int):
            if max_features < 1:
                raise ValueError("max_features can not be less than 1")
            else:
                return max_features
        elif isinstance(max_features, float):
            return max_features
        elif isinstance(max_features, str):
            if max_features == "sqrt":
                return max_features
            elif max_features == "log2":
                return max_features
            else:
                raise ValueError("The only string options available for max_features are \"sqrt\", \"log2\"")
        else:
            raise ValueError("max_features can only be int, float, or in {\"sqrt\", \"log2\"}")

    def __parse_max_features(self, max_features, num_features):
        if max_features is None:
            return None
        elif isinstance(max_features, int):
            return min(max_features, num_features)
        elif isinstance(max_features, float):
            return min(num_features, int(max_features * num_features))
        elif isinstance(max_features, str):
            if max_features == "sqrt":
                return int(np.sqrt(num_features))
            elif max_features == "log2":
                return int(np.log2(num_features))
        else:
            raise ValueError("Unable to parse max_features")

    def __check_input(self, X: object, Y: object):
        # If the skip_check_input option is set to True in the initialization of a tree, then simply return X and Y
        if self.skip_check_input:
            return X, Y

        # Make sure input arrays are c contigous
        X = np.ascontiguousarray(X, dtype=DOUBLE)
        Y = np.ascontiguousarray(Y, dtype=DOUBLE)

        # Check if X and Y has same number of rows
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y should have the same number of rows")

        # Check if Y has dimensions (n, 1) or (n,)
        if 2 < Y.ndim:
            raise ValueError("Y should have dimensions (n,1) or (n,)")
        elif 2 == Y.ndim:
            if 1 < Y.shape[1]:
                raise ValueError("Y should have dimensions (n,1) or (n,)")
            else:
                Y = Y.reshape(-1)

        return X, Y

    def __check_sample_weight(self, sample_weight: np.ndarray, n_samples):

        if sample_weight is None:
            return np.ones(n_samples, dtype=np.double)
        sample_weight = np.array(sample_weight, dtype=np.double)
        if sample_weight.shape[0] != n_samples:
            raise ValueError("sample_weight should have as many elements as X and Y")
        if sample_weight.ndim > 1:
            raise ValueError("sample_weight should have dimension (n_samples,)")
        return sample_weight

    def fit(
            self,
            X,
            Y,
            sample_indices: np.ndarray | None = None,
            sample_weight: np.ndarray | None = None) -> None:

        if not self.skip_check_input:
            X, Y = self.__check_input(X, Y)
        row, col = X.shape
        self.int_max_features = self.__parse_max_features(self.max_features, col)

        # If sample_weight is valid it is simply passed through check_sample_weight, if it is None all entries are set to 1
        sample_weight = self.__check_sample_weight(sample_weight=sample_weight, n_samples=row)

        builder = DepthTreeBuilder(
            X=X,
            Y=Y,
            sample_indices=sample_indices,
            int_max_features=self.int_max_features,
            sample_weight=sample_weight,
            criteria=self.criteria_class(X, Y, sample_weight),
            leaf_builder_class = self.leaf_builder_class,
            predict_class=self.predict_class,
            splitter=self.splitter)
        builder.build_tree(self)

    def predict(self, X: np.ndarray, **kwargs):
        if not self.predictor:
            raise AttributeError("The tree has not been fitted before trying to call predict")
        return np.asarray(self.predictor.predict(X, **kwargs))

    def predict_proba(self, X: np.ndarray):
        if not self.predictor:
            raise AttributeError("The tree has not been fitted before trying to call predict_proba")
        return np.asarray(self.predictor.predict_proba(X))

    def __get_leaf_matrix(self, scale: bool = False) -> np.ndarray:
        cdef:
            int i, j
            int[::1] indices
            int n_node

        if not self.root:
            raise ValueError("The tree has not been trained before trying to predict")

        leaf_nodes = self.leaf_nodes
        if (not leaf_nodes):  # make sure that there are calculated observations
            raise ValueError("The tree has no leaf nodes")

        n_obs = self.n_rows
        matrix = np.zeros((n_obs, n_obs))
        for node in leaf_nodes:
            indices = node.indices
            for i in indices:
                for j in indices:
                    if scale:
                        n_node = len(node.indices)
                        matrix[i, j] += 1/n_node
                    else:
                        matrix[i, j] += 1
        return matrix

    def predict_leaf_matrix(self, X: np.ndarray|None, scale: bool = False):
        if X is None:
            return self.__get_leaf_matrix(scale=scale)
        if not self.predictor:
            raise ValueError("The tree has not been trained before trying to predict")
        return self.predictor.predict_leaf_matrix(X, scale)


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
        int_max_features: int | None,
        sample_weight: np.ndarray,
        criteria: Criteria,
        splitter: Splitter,
        leaf_builder_class: LeafBuilder,
        predict_class: Predict,
        sample_indices: np.ndarray | None = None,
    ) -> None:
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
        self.X = X
        self.Y = Y
        self.int_max_features = int_max_features
        self.sample_indices = sample_indices
        self.criteria = criteria
        self.sample_weight = sample_weight

        self.splitter = splitter(self.X, self.Y, criteria)
        self.predict_class = predict_class
        self.leaf_builder_class = leaf_builder_class

        _, col = X.shape
        self.feature_indices = np.arange(col, dtype=np.int32)
        self.num_features = col

    def __get_feature_indices(self):
        if self.int_max_features is None:
            return self.feature_indices
        else:
            return np.random.choice(
                self.feature_indices,
                size=self.int_max_features,
                replace=False)

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

        all_idx = np.arange(X.shape[0])
        if self.sample_indices is not None:
            all_idx = np.array(self.sample_indices)

        all_idx = np.array(
            [x for x in all_idx if self.sample_weight[x] != 0], dtype=np.int32
        )

        # Update the tree now that we have the correct samples
        leaf_builder = self.leaf_builder_class(X, Y, all_idx)
        n_obs = all_idx.shape[0]

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
            n_samples = len(indices)
            # Stopping Conditions - BEFORE:
            # boolean used to determine wheter 'current node' is a leaf or not
            # additional stopping criteria can be added with 'or' statements
            is_leaf = (
                (depth >= max_depth)
                or (impurity <= impurity_tol + EPSILON)
                or (n_samples <= min_samples_split)
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
                    N_t_L = len(split[0])
                    N_t_R = len(split[1])
                    is_leaf = (
                        (
                            n_samples
                            / n_obs
                            * (
                                impurity
                                - (N_t_L / n_samples) * child_imp[0]
                                - (N_t_R / n_samples) * child_imp[1]
                            )
                            < min_improvement + EPSILON
                        )
                        or (N_t_L < min_samples_leaf)
                        or (N_t_R < min_samples_leaf)
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

        tree.n_rows = X.shape[0]
        tree.n_nodes = n_nodes
        tree.max_depth = max_depth_seen
        tree.n_features = X.shape[1]
        tree.n_obs = n_obs
        tree.root = root
        tree.leaf_nodes = leaf_node_list
        tree.predictor = self.predict_class(self.X, self.Y, root)
        return 0
