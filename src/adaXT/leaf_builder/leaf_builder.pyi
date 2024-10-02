import numpy as np
from adaXT.decision_tree.nodes import Node

class LeafBuilder:
    """
    The base LeafBuilder class from which all other leaf builders must inherit.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        all_idx: np.ndarray,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        X : np.ndarray
            The feature values used for training.
        Y : np.ndarray
            The response values used for training.
        all_idx : np.ndarray
            A vector specifying samples of the training data that should be
            considered by the LeafBuilder. 
        """
        pass

    def build_leaf(
        self,
        leaf_id: int,
        indices: np.ndarray,
        depth: int,
        impurity: float,
        weighted_samples: float,
        parent: Node,
    ) -> Node:
        """
        Builds a leaf node.

        Parameters
        ----------
        leaf_id : int
            unique identifier of leaf node
        indices : np.ndarray
            indices in leaf node
        depth : int
            depth of leaf node
        impurity : float
            impurity of leaf node
        weighted_samples : float
            summed weight of all samples in the LeafNode
        parent : DecisionNode
            parent node

        Returns
        -----------
        Node
            built leaf node
        """
        pass

class LeafBuilderClassification(LeafBuilder):
    pass

class LeafBuilderRegression(LeafBuilder):
    pass

class LeafBuilderPartialLinear(LeafBuilderRegression):
    pass

class LeafBuilderPartialQuadratic(LeafBuilderRegression):
    pass
