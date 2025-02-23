# cython: embedsignature=True
import numpy as np
cimport numpy as cnp

cdef class Node:
    def __init__(
            self,
            indices: np.ndarray,
            depth: int,
            impurity: float,
            parent=None) -> None:

        self.is_leaf = 0
        self.visited = 0
        self.parent = None
        self.indices = np.asarray(indices, np.int32)
        self.depth = depth
        self.impurity = impurity

cdef class DecisionNode(Node):
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
            is_leaf: int=0,
            visited: int=0) -> None:
        super().__init__(indices, depth, impurity)
        self.threshold = threshold
        self.split_idx = split_idx
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent

cdef class LeafNode(Node):
    def __init__(
            self,
            id: int,
            indices: np.ndarray,
            depth: int,
            impurity: float,
            weighted_samples: float,
            value: np.ndarray,
            parent: object) -> None:
        super().__init__(indices, depth, impurity, parent)
        self.weighted_samples = weighted_samples
        self.parent = parent
        self.id = id
        self.value = np.asarray(value, dtype=np.float64)
        self.is_leaf = 1


cdef class LocalPolynomialLeafNode(LeafNode):
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
            theta2: float) -> None:
        super().__init__(id, indices, depth, impurity, weighted_samples, value,
                         parent)
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta2
