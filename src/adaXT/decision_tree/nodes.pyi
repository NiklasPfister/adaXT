import numpy as np

class Node:
    """
    The base Node from which all other nodes must inherit.
    """

    indices: np.ndarray
    depth: int
    impurity: float

    def __init__(self, indices: np.ndarray, depth: int, impurity: float) -> None:
        """
        Parameters
        ----------
        indices : np.ndarray
            indices in node
        depth : int
            depth of node
        impurity : float
            impurity of node
        """
        pass

class DecisionNode(Node):
    threshold: float
    split_indx: int
    left_child: Node | None
    right_child: Node | None
    parent: DecisionNode | None
    split_idx: int
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
        parent: "DecisionNode|None" = None,
    ) -> None:
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
            threshold value of a split
        split_idx : int
            feature index to split on
        left_child : DecisionNode | LeafNode | None
            left child
        right_child : DecisionNode | LeafNode | None
            right child
        parent : DecisionNode | None
            parent node
        """
        pass

class LeafNode(Node):
    value: list[float]
    parent: DecisionNode | None
    id: int
    weighted_samples: float

    def __init__(
        self,
        id: int,
        indices: np.ndarray,
        depth: int,
        impurity: float,
        weighted_samples: float,
        value: np.ndarray,
        parent: DecisionNode,
    ) -> None:
        """
        Parameters
        ----------
        id : int
            unique identifier of leaf node
        indices : np.ndarray
            indices in leaf node
        depth : int
            depth of leaf node
        impurity : float
            impurity of leaf node
        weighted_samples : float
            summed weight of all samples in leaf node
        value : np.ndarray
            value of leaf node (depends on LeafBuilder)
        parent : DecisionNode
            parent node
        """
        pass

class LinearPolynomialLeafNode(LeafNode):
    def __init__(
        self,
        id: int,
        indices: np.ndarray,
        depth: int,
        impurity: float,
        weighted_samples: float,
        value: np.ndarray,
        parent: object,
        theta0: float,
        theta1: float,
        theta2: float,
    ) -> None:
        """
        Parameters
        ----------
        id : int
            unique identifier of leaf node
        indices : np.ndarray
            indices in leaf node
        depth : int
            depth of leaf node
        impurity : float
            impurity of leaf node
        weighted_samples : float
            summed weight of all samples in leaf node
        value : np.ndarray
            value of leaf node (depends on LeafBuilder)
        parent : DecisionNode
            parent node
        theta0 : float
            theta0 parameter corresponding to intercept term
        theta1 : float
            theta1 parameter correponding to linear term
        theta2 : float
            theta2 parameter correponding to quadratic term
        """
        pass
