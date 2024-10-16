# Tree-based weights

Decision trees divide the predictor space into regions, specified by the leaf
nodes, such that samples in the same leaf node are close in terms of the
criteria used during fitting. As first proposed by
[Lin and Jeon, 2012](https://doi.org/10.1198/016214505000001230) this
perspective can be used to view decision trees (and in extension random forests)
as adaptive nearest neighbor estimators. To make this more concrete, consider a
regression setting with training data
$(X_1, Y_1),\ldots,(X_n, Y_n)\in\mathcal{X}\times\mathbb{R}$, where $X_i$ is a
multivariate predictor in $\mathcal{X}\subseteq\mathbb{R}^d$ and $Y_i$ is a
real-valued response variable. For a fitted decision tree, we define the _leaf
function_ $\mathcal{L}:\mathcal{X}\times\mathcal{X}\rightarrow{0,1}$ which
determines for all $x,x'\in\mathcal{X}$ whether they lie in the same leaf, i.e.,
$\mathcal{L}(x, x')=1$ if $x$ and $x'$ are in the same leaf and
$\mathcal{L}(x,x')=0$ if $x$ and $x'$ are in different leafs. Furthermore,
define for all $i\in\{1,\ldots,n\}$ a weight function
$w_i:\mathcal{X}\rightarrow[0,1]$ for all $x\in\mathcal{X}$ by

$$
w_i^{\operatorname{DT}}(x):=\frac{\mathcal{L}(X_{i}, x)}{\sum_{\ell=1}^n\mathcal{L}(X_{\ell}, x)}.
$$

By construction it holds for all $x\in\mathcal{X}$ that
$\sum_{i=1}^n w_i^{\operatorname{DT}}(x)=1$. So intuitively the weights capture
how close a new observation $x$ is to each of the training samples. Importantly,
the predicted value of a regression tree at the $x$ is simply given by

$$
\sum_{i=1}^n w_i^{\operatorname{DT}}(x)Y_i,
$$

which corresponds to a weighted nearest neighbor estimator. Unlike other
classical weighted nearest neighbor estimators such as kernel smoothers or local
polynomial estimators, the weights in this case explicitly depend on the
response values making them _adaptive_.

As random forests are just averages over a collection of trees, the discussion
above naturally extends to them as well. For a fitted random forest, denote by
$\mathcal{L}_1,\ldots,\mathcal{L}_M$ denote the leaf functions for each of the
$M$ trees in the forest. Then, for all $i\in\{1,\ldots,n\}$ and all
$x\in\mathcal{X}$ define the weights by

$$
w_i^{\operatorname{RF}}(x):=\frac{1}{M}\sum_{m=1}^M\frac{\mathcal{L}_m(X_{i}, x)}{\sum_{\ell=1}^n\mathcal{L}_m(X_{\ell}, x)}.
$$

In general $w_i^{\operatorname{RF}}(x)$ and $Y_i$ are dependent since the sample
$i$ appears also in the definition of the weight $w_i^{\operatorname{RF}}(x)$.
This can be avoided using
[honest splitting](honest_splitting.md), which can be seen as
separating the estimation of the weights from the averaging of the responses.

Interpreting decision trees and random forests as adaptive nearest neighbor
estimators opens the door to applying them in a more diverse set of
applications. For example, quantile random forests were first introduced based
on this connection
[(Meinshausen, 2006)](https://jmlr.csail.mit.edu/papers/v7/meinshausen06a.html).

## Tree-based weights in adaXT

Tree-based weights can be computed in adaXT using the `predict_weights` method
in the DecisionTree and RandomForest classes. The code below illustrates the
usage.

```python
import numpy as np
import matplotlib.pyplot as plt
from adaXT.decision_tree import DecisionTree
from adaXT.random_forest import RandomForest

# Create toy data set
n = 100
X = np.random.uniform(-1, 1, n)
Y = 1.0 * (X > 0) + 2.0 * (X > 0.5) + np.random.normal(0, 0.2, n)

# Fit decision tree and random forest for regression
tree = DecisionTree("Regression", max_depth=2)
rf = RandomForest("Regression", max_depth=5)
tree.fit(X, Y)
rf.fit(X, Y)

# Extract tree-based weights at training points
# (X==None uses the training X)
W_tree = tree.predict_weights()
W_rf = rf.predict_weights()

# Computing tree-based weights at a new test points
Xtest = np.array([-1, -0.25, 0.25, 1])
Wtest_tree = tree.predict_weights(Xtest)
Wtest_rf = rf.predict_weights(Xtest)

# Rows correspond Xtest while columns always correspond to the training samples
print(Wtest_tree.shape)
print(Wtest_rf.shape)
# If scale=True the weights are scaled row-wise
print(Wtest_rf.sum(axis=1))
print(Wtest_tree.sum(axis=1))

# In the case of regression predicted values correspond to weighted averages of the Y
Yhat_tree = Wtest_tree.dot(Y)
Yhat_rf = Wtest_rf.dot(Y)
print(np.c_[Yhat_tree, tree.predict(Xtest)])
print(np.c_[Yhat_rf, rf.predict(Xtest)])
```

Using the notation above, the weights computed in this example satisfy
$\texttt{W_tree}[i, j]=w_j^{\operatorname{DT}}(\texttt{X}[i])$,
$\texttt{Wtest_tree}[i, j]=w_j^{\operatorname{DT}}(\texttt{Xtest}[i])$,
$\texttt{W_rf}[i, j]=w_j^{\operatorname{RF}}(\texttt{X}[i])$ and
$\texttt{Wtest_rf}[i, j]=w_j^{\operatorname{RF}}(\texttt{Xtest}[i])$ for both
the decision tree and random forest, respectively.

## Tree-based weight induced similarity

The tree-based weights can also be used to construct an adaptive measure of
closeness in the predictor space. Formally, for two observations
$x, x'\in\mathcal{X}$, we can define the similarity for decision trees

$$
S^{\operatorname{DT}}(x, x'):=\mathcal{L}(x, x'),
$$

and for random forests

$$
S^{\operatorname{RF}}(x, x'):=\frac{1}{M}\sum_{m=1}^M\mathcal{L}_m(x, x').
$$

This is implemented in adaXT via the `similarity` method that exists for both
the DecisionTree and RandomForest class. It allows to easily compute kernel (or
Gram) matrices for this similarity.

We illustrate this by continuing the example from above.

```{python}
# Compute similarity of x values in [-1,1] to test points
grid1d = np.linspace(-1, 1, 200)
test_points = np.array([-1, -0.25, 0.25, 1])
similarity_tree = tree.similarity(grid1d, test_points)
similarity_rf = rf.similarity(grid1d, test_points)

# Create plot
fig, ax = plt.subplots(1, 4)
for i in range(4):
    ax[i].plot(grid1d, similarity_tree[:, i], color='blue', label='decision tree', alpha=0.5)
    ax[i].plot(grid1d, similarity_rf[:, i], color='red', label='random forest', alpha=0.5)
    ax[i].set_title(f'similarity to x={test_points[i]}')
    ax[i].legend()
plt.show()
```
