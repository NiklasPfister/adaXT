# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as cnp
from typing import List


cdef class Node:
    def __init__(
            self,
            indices: np.ndarray,
            depth: int,
            impurity: float,
            n_samples: int) -> None:
        self.indices = indices
        self.depth = depth
        self.impurity = impurity
        self.n_samples = n_samples


cdef class DecisionNode(Node):
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
        super().__init__(indices, depth, impurity, n_samples)
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
            n_samples: int,
            value: np.ndarray,
            parent: object) -> None:
        super().__init__(indices, depth, impurity, n_samples)
        self.parent = parent
        self.id = id
        self.value = np.asarray(value)



