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
w_i(x):=\frac{\mathcal{L}(X_{i}, x)}{\sum_{\ell=1}^n\mathcal{L}(X_{\ell}, x)}.
$$

By construction it holds for all $x\in\mathcal{X}$ that $\sum_{i=1}^n w_i(x)=1$.
So intuitively the weights capture how close a new observation $x$ is to each of
the training samples. Importantly, the predicted value of a regression tree at
the $x$ is simply given by

$$
\sum_{i=1}^n w_i(x)Y_i,
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
w_i(x):=\frac{1}{M}\sum_{m=1}^M\frac{\mathcal{L}_m(X_{i}, x)}{\sum_{\ell=1}^n\mathcal{L}_m(X_{\ell}, x)}.
$$

In general $w_i(x)$ and $Y_i$ are dependent since the sample $i$ appears also in
the definition of the weight $w_i(x)$. This can be avoided using
[honest splitting](/docs/user_guide/honest_splitting.md), which can be seen as
separating the estimation of the weights from the averaging of the responses.

Interpreting decision trees and random forests as adaptive nearest neighbor
estimators opens the door to applying them in a more diverse set of
applications. For example, quantile random forests were first introduced based
on this connection
[(Meinshausen, 2006)](https://jmlr.csail.mit.edu/papers/v7/meinshausen06a.html).

## Tree-based weights in adaXT

Tree-based weights can be computed in adaXT using the `predict_leaf_matrix`
DecisionTree class method and the `predict_forest_weights` RandomForest class
method. As a simple example we the code below illustrates how to extract the
weights evaluated at training data used during fitting.

```python
import numpy as np
import matplotlib.pyplot as plt
from adaXT.decision_tree import DecisionTree
from adaXT.random_forest import RandomForest

# Create toy data set
n = 100
X = np.random.uniform(-1, 1, n)
Y = 1.0 * (X > 0) + 2.0 * (X > 0.5) + np.random.normal(0, 0.2, n)

# Fit regression tree and random forest
tree = DecisionTree("Regression", max_depth=2)
rf = RandomForest("Regression", max_depth=5)
tree.fit(X, Y)
rf.fit(X, Y)

# Extract tree-based weights at the training data
weights_tree = tree.predict_weights()
weights_rf = rf.predict_weights()
weights_rf.sum(axis=1)
weights_tree.sum(axis=1)

# Weights can be used to compute predictions
print(weights_rf.dot(Y) - rf.predict(X))
print(weights_tree.dot(Y) - tree.predict(X))

# Compute similarity of x values in [-1,1] to test points
grid1d = np.linspace(-1, 1, 200)
test_points = np.array([-1, -0.25, 0.25, 1])
similarity_tree = tree.similarity(grid1d, test_points)
similarity_rf = rf.similarity(grid1d, test_points)

# Create plot
fig, ax = plt.subplots(1, 4)
for i in range(4):
    ax[i].plot(grid1d, similarity_tree[:, i], color='blue', label='tree', alpha=0.5)
    ax[i].plot(grid1d, similarity_rf[:, i], color='red', label='random forest', alpha=0.5)
    ax[i].set_title(f'similarity to x={test_points[i]}')
    ax[i].legend()
plt.show()
```

Using the notation above, the weights computed in this example satisfy
$\texttt{weights}[i, j]=w_i(X_j)$.

<!--TODO: Update this section-->
