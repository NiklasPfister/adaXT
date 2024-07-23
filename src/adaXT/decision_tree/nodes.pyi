import numpy as np

class Node:
    indices: np.ndarray
    depth: int
    impurity: float

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
        """
        pass

class DecisionNode(Node):
    threshold: float
    split_indx: int
    left_child: Node|None
    right_child: Node|None
    parent: DecisionNode|None
    visited: int

    def __init__(
            self,
            indices: np.ndarray,
            depth: int,
            impurity: float,
            threshold: float,
            split_idx: int,
            left_child: "DecisionNode|LeafNode|None" = None,
            right_child: "DecisionNode|LeafNode|None" = None,
            parent: "DecisionNode|None" = None) -> None:
        """
        Parameters
        ----------
        indices : np.ndarray
            indices in node
        depth : int
            depth of node
        impurity : float
            impurity in node
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
        pass

class LeafNode(Node):
    value: list[float]
    parent: DecisionNode|None
    id: int
    weighted_samples: float
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
        weighted_samples : float
            Weight of the samples in the LeafNode
        value : list[float]
            The mean values of classes in leaf node
        parent : DecisionNode
            The parent node
        """
        pass
