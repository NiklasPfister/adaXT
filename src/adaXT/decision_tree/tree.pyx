
# General
import numpy as np
from numpy import float64 as DOUBLE
import sys

# Custom
from .splitter import Splitter
from .criteria import Criteria

cdef double EPSILON = np.finfo('double').eps


class Node:  # should just be a ctype struct in later implementation
    def __init__(
            self,
            indices: np.ndarray,
            depth: int,
            impurity: float,
            n_samples: int) -> None:
        """
        Parameters
        ----------
        indices : np.ndarray
            indices in node
        depth : int
            depth of noe
        impurity : float
            impurity of node
        n_samples : int
            number of samples in node
        """
        self.indices = indices  # indices of values within the node
        self.depth = depth
        self.impurity = impurity
        self.n_samples = n_samples


class DecisionNode(Node):
    def __init__(
            self,
            indices: np.ndarray,
            depth: int,
            impurity: float,
            n_samples: int,
            threshold: float,
            split_idx: int,
            left_child: "DecisionNode|LeafNode|None" = None,
            right_child: "DecisionNode|LeafNode|None" = None,
            parent: "DecisionNode|None" = None) -> None:
        """
        Parameters
        ----------
        indices : np.ndarray
            indices ni node
        depth : int
            depth of node
        impurity : float
            impurity in node
        n_samples : int
            number of samples in node
        threshold : float
            the threshold value of a split
        split_idx : int
            the feature index to split on
        left_child : DecisionNode|LeafNode|None, optional
            the left child, by default None
        right_child : DecisionNode|LeafNode|None, optional
            the right child, by default None
        parent : DecisionNode|None, optional
            the parent node, by default None
        """
        super().__init__(indices, depth, impurity, n_samples)
        self.threshold = threshold
        self.split_idx = split_idx
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent


class LeafNode(Node):
    def __init__(
            self,
            id: int,
            indices: np.ndarray,
            depth: int,
            impurity: float,
            n_samples: int,
            value: list[float],
            parent: DecisionNode) -> None:
        """

        Parameters
        ----------
        indices : np.ndarray
            Indices of leaf node
        depth : int
            depth the leaf node is located at
        impurity : float
            Impurity of leaf node
        n_samples : int
            Number of samples in leaf node
        value : list[float]
            The mean values of classes in leaf node
        parent : DecisionNode
            The parent node
        """
        super().__init__(indices, depth, impurity, n_samples)
        self.value = value
        self.parent = parent
        self.id = id


class Tree:
    """
    Tree object
    """

    def __init__(
            self,
            tree_type: str,
            max_depth: int = sys.maxsize,
            impurity_tol: float = 1e-20,
            min_samples: int = 1,
            root: Node | None = None,
            n_nodes: int = -1,
            n_features: int = -1,
            n_classes: int = -1,
            n_obs: int = -1,
            leaf_nodes: list[Node] | None = None,
            pre_sort: None | np.ndarray = None,
            classes: np.ndarray | None = None) -> None:
        """
        Parameters
        ----------
        tree_type : str
            Classification or Regression
        max_depth : int
            maximum depth of the tree, by default int(np.inf)
        impurity_tol : float
            the tolerance of impurity in a leaf node, by default 1e-20
        min_samples : int
            the minimum amount of samples in a leaf node, by deafult 2
        root : Node | None
            root node, by default None, added after fitting
        n_nodes : int | None
            number of nodes in the tree, by default -1, added after fitting
        n_features : int | None
            number of features in the dataset, by default -1, added after fitting
        n_classes : int | None
            number of classes in the dataset, by default -1, added after fitting
        n_obs : int | None
            number of observations in the dataset, by default -1, added after fitting
        leaf_nodes : list[Node] | None
            number of leaf nodes in the tree, by default None, added after fitting
        pre_sort: np.ndarray | None
            a sorted index matrix for the dataset
        classes : np.ndarray | None
            the different classes in response, by default None, added after fitting
        """
        tree_types = ["Classification", "Regression"]
        assert tree_type in tree_types, f"Expected Classification or Regression as tree type, got: {tree_type}"
        self.max_depth = max_depth
        self.impurity_tol = impurity_tol
        self.min_samples = min_samples
        self.tree_type = tree_type
        self.leaf_nodes = leaf_nodes
        self.root = root
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_obs = n_obs
        self.pre_sort = pre_sort
        self.classes = classes

    def check_input(self, X: object, Y: object):
        # Make sure input arrays are c contigous
        X = np.ascontiguousarray(X, dtype=DOUBLE)
        Y = np.ascontiguousarray(Y, dtype=DOUBLE)

        return X, Y

    def fit(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            criteria: Criteria,
            splitter: Splitter | None = None,
            feature_indices: np.ndarray | None = None,
            sample_indices: np.ndarray | None = None) -> None:
        """
        Function used to fit the data on the tree using the DepthTreeBuilder

        Parameters
        ----------
        X : np.ndarray
            feature values
        Y : np.ndarray
            outcome values
        criteria : FuncWrapper
            Callable criteria function used to calculate impurity wrapped in Funcwrapper class.
        splitter : Splitter | None, optional
            Splitter class if None uses premade Splitter class
        feature_indices : np.ndarray | None, optional
            which features to use from the data X, by default uses all
        sample_indices : np.ndarray | None, optional
            which samples to use from the data X and Y, by default uses all
        """
        # TODO: test feature and sample indexing
        X, Y = self.check_input(X, Y)
        row, col = X.shape
        if sample_indices is None:
            sample_indices = np.arange(row)
        if feature_indices is None:
            feature_indices = np.arange(col)
        builder = DepthTreeBuilder(
            X,
            Y,
            feature_indices,
            sample_indices,
            criteria(X, Y),
            splitter,
            self.impurity_tol,
            pre_sort=self.pre_sort)
        builder.build_tree(self)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts a y-value for given X values

        Parameters
        ----------
        X : np.ndarray
            (N, M) numpy array with features to predict

        Returns
        -------
        np.ndarray
            (N) numpy array with the prediction
        """
        # Check if node exists
        row, _ = X.shape
        Y = np.empty(row)
        if not self.root:
            return Y
        for i in range(row):
            cur_node = self.root
            while isinstance(cur_node, DecisionNode):
                if X[i, cur_node.split_idx] < cur_node.threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            if isinstance(
                    cur_node,
                    LeafNode) and self.tree_type == "Regression":
                Y[i] = cur_node.value[0]
            elif isinstance(cur_node, LeafNode) and self.tree_type == "Classification":
                values = np.array(cur_node.value)
                idx = np.argmax(values)
                if isinstance(self.classes, np.ndarray):
                    Y[i] = self.classes[idx]

        return Y

    def weight_matrix(self) -> np.ndarray:
        """
        Creates NxN matrix,
        where N is the number of observations.
        If a given value is 1, then they are in the same leaf,
        otherwise it is 0

        Returns
        -------
        np.ndarray
            NxN matrix
        """
        leaf_nodes = self.leaf_nodes
        n_obs = self.n_obs

        data = np.zeros((n_obs, n_obs))
        if (not leaf_nodes):  # make sure that there are calculated observations
            return data
        for node in leaf_nodes:
            data[np.ix_(node.indices, node.indices)] = 1

        # TODO scale
        return data

    def predict_matrix(self, X: np.ndarray):
        cdef:
            int row, i
        row = X.shape[0]
        Y = np.empty(row)
        ht = {}
        if not self.root:
            return Y
        for i in range(row):
            cur_node = self.root
            while isinstance(cur_node, DecisionNode):
                if X[i, cur_node.split_idx] < cur_node.threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child

            # Add to the dict
            if isinstance(cur_node, LeafNode):
                if cur_node.id not in ht.keys():
                    ht[cur_node.id] = [i]
                else:
                    ht[cur_node.id] += [i]
        matrix = np.zeros((row, row))
        for key in ht.keys():
            indices = ht[key]
            matrix[np.ix_(indices, indices)] = 1

        # TODO scale
        return matrix

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
            min_impurity: float = 0,
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
            tree: Tree,
            node_response: np.ndarray,
            n_samples: int) -> list[float]:
        """
        Calculates the mean of a leafnode

        Parameters
        ----------
        tree : Tree
            The fille tree object
        node_response : np.ndarray
            outcome values in the node
        n_samples : int
            number of samples in the node
        n_classes : int
            number of different classes in the node

        Returns
        -------
        list[float]
            A list of mean values for each class in the node
        """
        if tree.tree_type == "Regression":
            return [float(np.mean(node_response))]
        # create an empty list for each class type
        lst = [0.0 for _ in range(tree.n_classes)]
        for i in range(tree.n_classes):
            for idx in range(n_samples):
                if node_response[idx] == self.classes[i]:
                    lst[i] += 1  # add 1, if the value is the same as class value
            # weight by the number of total samples in the leaf
            lst[i] = lst[i] / n_samples
        return lst

    def build_tree(self, tree: Tree):
        """
        Builds the tree

        Parameters
        ----------
        tree : Tree
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
            # bool used to determine wheter a node is a leaf or not, feel free
            # to add or statements
            is_leaf = ((depth >= max_depth) or
                       (abs(impurity - self.min_impurity) < EPSILON) or
                       (n_samples <= min_samples))
            # Check improvement
            # if parent != None:
            #     is_leaf = (abs(parent.impurity - impurity) <= self.min_improvement) or is_leaf

            # TODO: possible impurity improvement tolerance.
            if depth > max_depth_seen:  # keep track of the max depth seen
                max_depth_seen = depth

            if not is_leaf:
                split, best_threshold, best_index, _, chil_imp = splitter.get_split(
                    indices)

                # Add the decision node to the list of nodes
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
                        chil_imp[0],
                        new_node,
                        1))
                # Add the right node to the queue of nodes yet to be computed
                queue.append(
                    queue_obj(
                        right,
                        depth + 1,
                        chil_imp[1],
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
