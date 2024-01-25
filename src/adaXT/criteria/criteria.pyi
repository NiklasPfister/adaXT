class Criteria:
    """
    Abstract class used as a parent for new Criteria functions
    """
    pass


class Gini_index(Criteria):
    r"""
    Calculates the gini index given by
    $$
    \text{Gini Index} = 1 - \sum_{i=1}^n (P[i])^2,
    $$
    where $P[i]$ denotes the probability of an element
    being classified for a distinct class.
    """
    pass


class Entropy(Criteria):
    r"""
    Calculates the Entropy given by
    $$
    \text{Entropy} = - \sum_{i = 1}^n P[i] \log_2 (P[i]),
    $$
    where $P[i]$ denotes the probability of randomly selecting an example
    in class i.
    """
    pass


class Squared_error(Criteria):
    r"""
    Calculates the Squared error given by
    $$
    \text{Squared Error} = \frac{\sum_{i = 1}^n (Y[i] - \mu_Y)^2}{n_{obs}},
    $$
    where $Y$ denotes the outcome values in a node. $\mu_Y$ denotes the mean value in the node,
    and $n_{obs}$ denotes the number of observations in the node.

    For a faster, but equivalent calculation, it is calculated using the formula
    $$
    \text{Squared Error} = \frac{\sum_{i = 1}^n Y[i]^2}{n_obs} - \mu_Y^2
    $$
    """
    pass


class Linear_regression(Criteria):
    r"""
    Calculates the impurity given by
    $$
    \text{Linear Regression} = \sum_{i \in indices} (Y[i] - \\theta_0 - \\theta_1 X[i, 0])^2.
    $$
    """
    pass
