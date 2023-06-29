class Tree:
    pass


class Node: # should just be a ctype struct in later implementation
    """
    Node in a tree
    """
    def __init__(self, indices: list[int], gini_val: float, parent = None, threshold: float = None, value: float = None) -> None:
        """
        Parameters
        ----------
        indices : list[int]
            indices within the data, which are apart of the node
        gini_val : float
            gini value of the node
        parent : Node, optional
            parent node, by default None
        threshold : float, optional
            threshold value of a decision node, by default None
        value : float, optional
            mean outcome value of datapoints in leaf node, by default None
        """
        self.indices = indices # indices of values within the node
        self.parent = parent
        self.threshold = threshold # None for leaf nodes.
        self.value = value # None for decision nodes
