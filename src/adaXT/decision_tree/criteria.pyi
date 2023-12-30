class Criteria:
    """
    Criteria abstract class used to calculate splits
    """
    pass


class Gini_index(Criteria):
    r"""
    Calculates the gini index given:
    $$
    Gini Index = 1 - \sum_{i=1}^n (P_i)^2
    $$
    Where $P_i$ denotes the probability of an element
    being classified for a distinct class.
    """
    pass
class Entropy(Criteria):
    r"""
    Calculates the Entropy given:
    $$
    E = - \sum_{i = 1}^n p_i \log_2 (p_i)
    $$
    Where $p_i$ denotes the probability of randomly selecting an example
    in class i.
    """
    pass
class Squared_error(Criteria):
    pass
class Linear_regression(Criteria):
    r"""
    Calculates the impurity of a Node by:
    $$
    L = \sum_{i \in indices} (Y[i] - \\theta_0 - \\theta_1 X[i, 0])^2
    $$
    """
    pass