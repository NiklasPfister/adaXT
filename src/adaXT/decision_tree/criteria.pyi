class Criteria:
    """
    Criteria abstract class used to calculate splits
    """
    pass


class Gini_index(Criteria):
    """
    Gini index calculation.

    Calculates the gini index given:
    $$
    Gini Index = 1 - \sum_{i=1}^n (P_i)^2
    $$
    Where $P_i$ denotes the probability of an element
    being classified for a distinct class.
    """

    pass
class Entropy(Criteria):
    pass
class Squared_error(Criteria):
    pass
class Linear_regression(Criteria):
    pass