class Criteria:
    """
    The base Criteria class from which all other criteria need to inherit.
    """

    pass

class ClassificationCriteria(Criteria):
    """
    Parent class for Criteria used in the Classification Tree Type. Can not be
    used as a standalone Criteria.
    """

    pass

class GiniIndex(ClassificationCriteria):
    r"""
    Gini index based criteria, which can be used for classification.
    Formally, given class labels $\mathcal{L}$, the Gini index in a node
    consisting of samples $I$, is given by
    $$
    \text{GiniIndex} = 1 - \sum_{k\in \mathcal{L}} P[k]^2,
    $$
    where $P[k]$ denotes the fraction of samples in $I$ with class
    label $k$.
    """

    pass

class Entropy(ClassificationCriteria):
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

class RegressionCriteria(Criteria):
    """
    Parent class for criteria used in Regression Tree Type. Can not be used as a
    standalone Criteria.
    """

    pass

class SquaredError(RegressionCriteria):
    r"""
    Squared error based criteria, which can be used for regression and
    leads to standard CART splits. Formally, the squared error in a node
    consisting of samples $I$, is given by
    $$
    \text{SquaredError} = \tfrac{1}{|I|}\sum_{i\in I}
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

class MultiSquaredError(RegressionCriteria):
    r"""
    Multi dimensional squared error criteria. With Y-values in one-dimension, it
    is equivalent to the SquaredError criteria. However, this criteria is able
    to function with Y-values in multiple dimensions. Formally, the
    MultiSquaredError in a node consisting of samples $I$ and Y-values of $D$
    dimensions, is given by:
    $$
    \text{MultiSquaredError} = \tfrac{1}{|I|} \sum^D_{j = 1} \sum_{i \in I}
    \Left(Y[i, j] - \tfrac{1}{|I|\sum_{i \in I} Y[I] \Right)^2
    $$

    For a faster, but equivalent calculation, it is computed as:
    $$
    \text{MultiSquaredError} = \tfrac{1}{|I|} \sum^D_{j = 1} \left(\sum_{i\in I} Y[i]^2
    - \Big(\tfrac{1}{|I|}\sum_{i\in I} Y[i]\Big)^2 \right)
    $$
    """

    pass

class PairwiseEuclideanDistance(RegressionCriteria):
    r"""
    Pairwise Euclidean Distance criteria. Generally performs in a similair
    fashion to the MultiSquaredError. However, instead of the squared error
    compared with the mean, it instead minimizes the individual distance between
    points in a node. Formally, the PairwiseEuclideanDistance in a node
    consisting of samples $I$ and Y-values of $D$ dimensions is given by:
    $$
    \text{PairwiseEuclideanDistance} = \tfrac{1}{|I|} \sum_{i = 1}^{|I| -1}
    \sum_{j = i}^{|I|} \sqrt{\sum_{k = 1}^{D} (Y[I[i], k] - Y[I[j], k])^2}
    $$
    """

    pass

class PartialLinear(RegressionCriteria):
    r"""
    Criteria based on fitting a linear function in the first predictor
    variable in each leaf. Formally, in a node consisting of samples $I$,
    it is given by
    $$
    \text{PartialLinear} = \tfrac{1}{|I|}\sum_{i \in I}
    (Y[i] - \widehat{\theta}_0 - \widehat{\theta}_1 X[i, 0])^2,
    $$
    where $Y[i]$ and $X[i, 0]$ denote the response value and
    the value of the first feature at sample $i$, respectively, and
    $(\widehat{\theta}_0, \widehat{\theta}_1)$ are ordinary
    least squares regression estimates when regressing
    $Y[i]$ on $X[i, 0]$ using the samples in $I$.
    """

    pass

class PartialQuadratic(RegressionCriteria):
    r"""
    Criteria based on fitting a quadratic function in the first predictor
    variable in each leaf. Formally, in a node consisting of samples $I$,
    it is given by
    $$
    \text{PartialQuadratic} = \tfrac{1}{|I|}\sum_{i \in I}
    (Y[i] - \widehat{\theta}_0 - \widehat{\theta}_1 X[i, 0] - \widehat{\theta}_2 X[i, 0]^2)^2,
    $$
    where $Y[i]$ and $X[i, 0]$ denote the response value and
    the value of the first feature at sample $i$, respectively, and
    $(\widehat{\theta}_0, \widehat{\theta}_1)$ are ordinary
    least squares regression estimates when regressing
    $Y[i]$ on $X[i, 0]$ using the samples in $I$.
    """

    pass
