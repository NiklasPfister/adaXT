class Criteria:
    """
    Abstract class used as a parent for new Criteria classes.
    """
    pass


class Gini_index(Criteria):
    r"""
    Gini index based criteria, which can be used for classification.
    Formally, given class labels $\mathcal{L}$, the Gini index in a node
    consisting of samples $I$, is given by
    $$
    \text{Gini\_index} = 1 - \sum_{k\in \mathcal{L}} P[k]^2,
    $$
    where $P[k]$ denotes the fraction of samples in $I$ with class
    label $k$.
    """
    pass


class Entropy(Criteria):
    r"""
    Entropy based criteria, which can be used for classification.
    Formally, given class labels $\mathcal{L}$, the entropy in a node
    consisting of samples $I$, is given by
    $$
    \text{Entropy} = - \sum_{k\in\mathcal{L}} P[k] \log_2 (P[k]),
    $$
    where $P[k]$ denotes the fraction of samples in $I$ with class
    label $k$.
    """
    pass


class Squared_error(Criteria):
    r"""
    Squared error based criteria, which can be used for regression and
    leads to standard CART splits. Formally, the squared error in a node
    consisting of samples $I$, is given by
    $$
    \text{Squared\_error} = \tfrac{1}{|I|}\sum_{i\in I}
    \Big(Y[i] - \tfrac{1}{|I|}\sum_{i\in I} Y[i]\Big)^2,
    $$
    where $Y[i]$ denotes the response value at sample $i$.

    For a faster, but equivalent calculation, it is computed by
    $$
    \text{Squared\_error} = \tfrac{1}{|I|}\sum_{i\in I} Y[i]^2
    - \Big(\tfrac{1}{|I|}\sum_{i\in I} Y[i]\Big)^2
    $$
    """
    pass


class Linear_regression(Criteria):
    r"""
    Linear regression based criteria, which adapts the Squared_error
    criteria by fitting a linear regression in the first coordinate.
    Formally, in a node consisting of samples $I$, it is given by
    $$
    \text{Linear\_regression} = \sum_{i \in I}
    (Y[i] - \widehat{\theta}_0 - \widehat{\theta}_1 X[i, 0])^2,
    $$
    where $Y[i]$ and $X[i, 0]$ denote the response value and
    the value of the first feature at sample $i$, respectively, and
    $(\widehat{\theta}_0, \widehat{\theta}_1)$ are ordinary
    least squares regression estimates when regressing
    $Y[i]$ on $X[i, 0]$ using the samples in $I$.
    """
    pass
