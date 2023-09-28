# General
from typing import Callable, List
import numpy as np
import numpy.typing as npt
import sys
from scipy.sparse import issparse
from numpy import float64 as DOUBLE

# Custom
from ._func_wrapper import FuncWrapper
from ._criteria import gini_index_wrapped
from ._splitter import Splitter

crit = gini_index_wrapped # default criteria function


class Node: # should just be a ctype struct in later implementation
    def __init__(self, indices: List[int], depth: int, impurity: float, n_samples: int) -> None:
        """
        Node parent class

        Parameters
        ----------
        indices : list[int]
            indices within the data, which are apart of the node
        depth : int
            depth of the node
        impurity : float
            impurity of the node
        n_samples : int
            number of samples within a node
        """
        self.indices = indices # indices of values within the node
        self.depth = depth
        self.impurity = impurity
        self.n_samples = n_samples

class DecisionNode(Node):
    def __init__(self, indices: List[int], depth: int, impurity: float, n_samples: int, threshold: float, split_idx: int, left_child: "DecisionNode|LeafNode|None" = None, right_child: "DecisionNode|LeafNode|None"= None, parent: "DecisionNode|None" = None) -> None:
        """
        Decision node class

        Parameters
        ----------
        indices : list[int]
            indices within the data, which are apart of the node
        depth : int
            depth of the node
        impurity : float
            impurity of the node
        n_samples : int
            number of samples within a node
        threshold : float
            threshold value for a given split
        split_idx : int
            feature index of the split
        left_child : Node | None, optional
            left child, by default None
        right_child : Node | None, optional
            right child, by default None
        parent : Node | None, optional
            parent node, by default None
        """
        super().__init__(indices, depth, impurity, n_samples)
        self.threshold = threshold
        self.split_idx = split_idx
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent

class LeafNode(Node):
    def __init__(self, indices: List[int], depth: int, impurity: float, n_samples: int, value: list[float], parent: DecisionNode) -> None:
        """
        Leaf Node class

        Parameters
        ----------
        indices : List[int]
            _description_
        depth : int
            _description_
        impurity : float
            _description_
        n_samples : int
            _description_
        value : list[float]
            mean value of the outcomes in the leaf node
        parent : _type_, optional
            _description_, by default None
        """
        super().__init__(indices, depth, impurity, n_samples)
        self.value = value 
        self.parent = parent


class Tree:
    """
    Tree object
    """
    def __init__(self, tree_type: str, max_depth: int = sys.maxsize, impurity_tol: float = 1e-20, min_samples: int = 2,
                root : Node|None = None, n_nodes: int=-1, n_features: int=-1, n_classes: int=-1, n_obs: int=-1,
                leaf_nodes: list[Node]|None = None, pre_sort: None| npt.NDArray = None, classes : npt.NDArray|None = None) -> None:
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
        pre_sort: npt.NDArray | None
            a sorted index matrix for the dataset
        classes : npt.NDArray | None
            the different classes in outcomes, by default None, added after fitting
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
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DOUBLE:
                X.data = np.ascontiguousarray(X, dtype=DOUBLE)

        elif X.dtype != DOUBLE:
            X = np.asfortranarray(X, dtype=DOUBLE)
        
        if Y.dtype != DOUBLE:
            Y = np.ascontiguousarray(Y, dtype=DOUBLE)
    
        return X, Y
    
    def fit(self, X: npt.NDArray, Y:npt.NDArray, criteria: FuncWrapper, splitter:Splitter | None = None, 
            feature_indices: npt.NDArray|None = None, sample_indices: npt.NDArray|None = None) -> None:
        """
        Function used to fit the data on the tree using the DepthTreeBuilder

        Parameters
        ----------
        X : npt.NDArray
            feature values
        Y : npt.NDArray
            outcome values
        criteria : Callable
            Callable criteria function used to calculate
        splitter : splitter.Splitter | None, optional
            Splitter class if None uses premade Splitter class
        feature_indices : npt.NDArray | None, optional
            which features to use from the data X, by default uses all
        sample_indices : npt.NDArray | None, optional
            which samples to use from the data X and Y, by default uses all
        """
        # TODO: test feature and sample indexing
        X, Y = self.check_input(X, Y)
        row, col = X.shape
        if sample_indices is None:
            sample_indices = np.arange(row)
        if feature_indices is None:
            feature_indices = np.arange(col)
    
        builder = DepthTreeBuilder(X, Y, feature_indices, sample_indices, criteria, splitter, self.impurity_tol, pre_sort=self.pre_sort)
        builder.build_tree(self)


    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """
        Predicts a y-value for given X values

        Parameters
        ----------
        X : npt.NDArray
            (N, M) numpy array with features to predict

        Returns
        -------
        npt.NDArray
            (N, M+1) numpy array with last column being the predicted y-values, or empty on fail
        """
        # Check if node exists
        row, _ = X.shape
        Y = np.empty(row)

        if not self.root: 
            return Y
        for i in range(row):
            cur_node = self.root
            while type(cur_node) == DecisionNode:
                if X[i, cur_node.split_idx] < cur_node.threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            if type(cur_node) == LeafNode and self.tree_type == "Regression":
                Y[i] = cur_node.value[0]
            elif type(cur_node) == LeafNode and self.tree_type == "Classification":
                idx = np.argmax(cur_node.value)
                if type(self.classes) == np.ndarray:
                    Y[i] = self.classes[idx]

        return Y
    
    def weight_matrix(self) -> npt.NDArray:
        """
        Creates NxN matrix, where N is the number of observations. If a given value is 1, then they are in the same leaf, otherwise it is 0

        Returns
        -------
        npt.NDArray|int
            NxN matrix, or -1 if there no observations or leaf_nodes in the tree.
        """
        leaf_nodes = self.leaf_nodes
        n_obs = self.n_obs

        data = np.zeros((n_obs, n_obs))
        if (not leaf_nodes): # make sure that there are calculated observations
            return data
        for node in leaf_nodes: 
            data[np.ix_(node.indices, node.indices)] = 1
        return data

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
    def __init__(self, X: npt.NDArray, Y: npt.NDArray, feature_indices: npt.NDArray, sample_indices: npt.NDArray, criteria: FuncWrapper|None = None, splitter: Splitter|None = None, tol : float = 1e-9,
                pre_sort:npt.NDArray|None = None) -> None:
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
        Splitter : Splitter
            optional splitter class, uses standard implementation by default
        tol : float
            tolerance for impurity of leaf nodes
        """
        self.features = X[np.ix_(sample_indices, feature_indices)]
        self.outcomes = Y[sample_indices]
        self.feature_indices = feature_indices
        self.sample_indices = sample_indices
        if criteria:
            self.criteria = criteria
        else:
            self.criteria = crit()

        if splitter:
            self.splitter = splitter
        else:
            self.splitter = Splitter(self.features, self.outcomes, criteria)

        if type(pre_sort) == np.ndarray:
            if pre_sort.dtype != np.int32:
                pre_sort = np.ascontiguousarray(pre_sort, np.int32)
            self.splitter.set_pre_sort(pre_sort)
        self.tol = tol

    def get_mean(self, tree: Tree, node_outcomes: npt.NDArray, n_samples: int, n_classes: int) -> list[float]:
        if tree.tree_type == "Regression":
            return [float(np.mean(node_outcomes))]
        lst = [0.0 for _ in range(n_classes)] # create an empty list for each class type   
        classes = self.classes
        for i in range(n_classes):
            for idx in range(n_samples):
                if node_outcomes[idx] == classes[i]:
                    lst[i] += 1 # add 1, if the value is the same as class value
            lst[i] = lst[i]/n_samples # weight by the number of total samples in the leaf
        return lst

        
    def build_tree(self, tree: Tree) -> Tree:
        """
        Builds the tree 

        Parameters
        ----------
        tree : Tree
            the tree to build
        sample_indices : npt.NDArray
            which samples to use from the total dataset
        feature_indices : npt.NDArray
            which features to use from the total dataset
        Returns
        -------
        Tree
            the tree object built
        """
        features = self.features
        outcomes = self.outcomes
        splitter = self.splitter
        min_samples = tree.min_samples
        criteria = self.criteria
        max_depth = tree.max_depth
        classes = np.unique(outcomes)
        self.classes = classes
        n_classes = len(self.classes)
        tree.n_classes = n_classes
        tree.classes = classes
        
        if tree.tree_type == "Classification":
            # Initialize c lists in splitter class
            splitter.make_c_lists(n_classes)

        root = None
    
        leaf_node_list = []
        max_depth_seen = 0
        
        n_obs = len(outcomes)
        queue = [] # queue of elements queue objects that need to be built
        
        all_idx = np.arange(n_obs, dtype=np.int32) # root node contains all indices
        queue.append(queue_obj(all_idx, 0, criteria.crit_func(features[all_idx], outcomes[all_idx], np.array(all_idx))))
        n_nodes = 0
        while len(queue) > 0:
            obj = queue.pop()
            indices, depth, impurity, parent, is_left = obj.indices, obj.depth, obj.impurity, obj.parent, obj.is_left
            n_samples = len(indices)
            is_leaf = (depth >= max_depth or impurity <= self.tol
                       or n_samples < min_samples) # bool used to determine wheter a node is a leaf or not, feel free to add or statements
            #TODO: possible impurity improvement tolerance.
            if depth > max_depth_seen: # keep track of the max depth seen
                max_depth_seen = depth

            if not is_leaf:
                split, best_threshold, best_index, best_score, chil_imp = splitter.get_split(indices)

                # Add the decision node to the list of nodes
                new_node = DecisionNode(indices, depth, impurity, n_samples, best_threshold, best_index, parent = parent)
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
                mean_value = self.get_mean(tree, outcomes[indices], n_samples, n_classes)
                new_node = LeafNode(indices, depth, impurity, n_samples, mean_value, parent=parent)
                if is_left and parent: # if there is a parent
                    parent.left_child = new_node
                elif parent:
                    parent.right_child = new_node
                leaf_node_list.append(new_node)
            if n_nodes == 0:
                root = new_node
            n_nodes += 1 # number of nodes increase by 1

        splitter.free_c_lists()
        tree.n_nodes = n_nodes
        tree.max_depth = max_depth_seen
        tree.n_features = features.shape[0]
        tree.n_obs = n_obs
        tree.root = root
        tree.leaf_nodes = leaf_node_list
        tree.n_classes = n_classes
        tree.classes = classes
        return tree