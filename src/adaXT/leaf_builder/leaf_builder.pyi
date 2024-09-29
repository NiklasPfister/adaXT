from adaXT.decision_tree.nodes import Node

class LeafBuilder:
    def build_leaf(
        self,
        leaf_id: int,
        indices: int,
        depth: int,
        impurity: float,
        weighted_samples: float,
        parent: Node,
    ) -> Node:
        pass

class LeafBuilderClassification(LeafBuilder):
    pass

class LeafBuilderRegression(LeafBuilder):
    pass

class LeafBuilderPartialLinear(LeafBuilderRegression):
    pass

class LeafBuilderPartialQuadratic(LeafBuilderRegression):
    pass
