# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

# General
import numpy as np
from typing import List

# Custom
from .splitter import Splitter
from .criteria import Criteria

class Node:
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
            value: List[float],
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
        value : List[float]
            The mean values of classes in leaf node
        parent : DecisionNode
            The parent node
        """
        super().__init__(indices, depth, impurity, n_samples)
        self.value = value
        self.parent = parent
        self.id = id