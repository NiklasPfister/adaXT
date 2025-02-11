import numpy as np


cdef class Node:
    def __cinit__(self):
        self.is_leaf = 0
        self.visited = 0
    
    def __init__(
            self,
            indices: np.ndarray,
            depth: int,
            impurity: float) -> None:
        self.indices = indices
        self.depth = depth
        self.impurity = impurity

    def __reduce__(self):
        return (
                self.__class__, # Callable object that will be called ot create
                                # initial state upon pickle
                (self.indices, self.depth, self.impurity), # Input to Callable
                {
                    "is_leaf": self.is_leaf,
                    "visited": self.visited,
                    "indices": self.indices.base
                } # Current state of variables that can not be passed to init
                )
    # This function is passed the state provided above
    def __setstate__(self, d: dict):
        self.is_leaf = d["is_leaf"]
        self.visited = d["visited"]
        self.indices = d["indices"]

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
            parent: "DecisionNode|None" = None) -> None:
        super().__init__(indices, depth, impurity)
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
            weighted_samples: float,
            value: np.ndarray,
            parent: object) -> None:
        super().__init__(indices, depth, impurity)
        self.weighted_samples = weighted_samples
        self.parent = parent
        self.id = id
        self.value = np.asarray(value)
        self.is_leaf = 1


class LocalPolynomialLeafNode(LeafNode):
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
        super().__init__(id, indices, depth, impurity, weighted_samples, value, parent)
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta2
