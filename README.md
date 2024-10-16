## Fast Adaptable and Extendable Trees for Research

**adaXT** is a Python module for tree-based machine-learning algorithms that is
fast, adaptable and extendable. It aims to provide researchers a more flexible
workflow when developing tree-based models.

It is distributed under the
[3-Clause BSD license](https://github.com/NiklasPfister/adaXT/blob/main/LICENSE).

We encourage users and developers to report problems, request features, ask for
help, or leave general comments.

Website:
[https://NiklasPfister.github.io/adaXT](https://NiklasPfister.github.io/adaXT)

### Overview

adaXT implements several tree types that can be used out-of-the-box, both as
decision trees and random forests. Currently, the following tree types are
implemented:

- **Classification:** For prediction tasks in which the response is categorical.
- **Regression:** For prediction tasks in which the response is continuous.
- **Quantile:** For uncertainty quantification tasks in which the response is
  continuous and the goal is to estimate one or more quantiles of the
  conditional distribution of the response given the predictors.
- **Gradient:** For tasks in which one aims to estimate (directional)
  derivatives of the response given the predictors. A related tree type is used
  in the
  [Xtrapolation](https://github.com/NiklasPfister/ExtrapolationAware-Inference)
  method.

Beyond these pre-defined tree types, adaXT offers a simple interface to extend
or modify most components of the tree models. For example, it is easy to create
a [custom criteria](https://NiklasPfister.github.io/adaXT/user_guide/creatingCriteria/)
function that is used
to create splits.

### Getting started

adaXT is available on [pypi](https://pypi.org/project/adaXT) and can be
installed via pip

```bash
pip install adaXT
```

Working with any of the default tree types uses the same class-style interface
as other popular machine learning packages. The following code illustrates this
for `Regression` and `Quantile` random forests:

```python
from adaXT.random_forest import RandomForest
import numpy as np

# Create toy regression data
n = 100
X = np.random.normal(0, 1, (n, 2))
Y = X[:, 0] + np.random.normal(0, 1, n)
Xtest = np.c_[np.linspace(-1, 1, n), np.random.uniform(0, 1, n)]

# Task 1: Fit regression forest
rf = RandomForest("Regression")
rf.fit(X, Y)

# Predict on test data
Ypred = rf.predict(Xtest)

# Predict forest weight on X or Xtest
# -- can be used a similarity measure on the predictor space
weight_train = rf.predict_weights()
weight_test = rf.predict_weights(Xtest)

# Task 2: Fit a quantile regression
qf = RandomForest("Quantile")
qf.fit(X, Y)

# Predict 10% and 90% conditional quantile on test data
Ybdd = qf.predict(Xtest, quantile=[0.1, 0.9])
```

The main advantage of adaXT over existing tree-based ML packages is its
modularity and extendability, which is discussed in detail in the
[documentation](https://NiklasPfister.github.io/adaXT).

### Project goals

This project aims to provide a flexible and unified code-base for various
tree-based algorithms that strikes a balance between speed and ease with which
the code can be adapted and extended. It should provide researchers a simple
toolkit for prototyping new tree-based algorithms.

adaXT provides an intuitive user experience that is similar to the
[scikit-learn](https://scikit-learn.org) implementation of decision trees both
in terms of model-based syntax and hyperparameters. Under the hood, however, adaXT
strikes a different balance between speed and ease of adapting and extending the
code.

#### Adaptable and extendable

At the heart of any tree-based algorithm is a decision tree that can be fitted
on data and then used to perform some version of prediction. adaXT has therefore
been designed with a modular decision tree implementation that takes four input
components:

- Criteria class: Used during fitting to determine splits.

- LeafBuilder class: Used during fitting to specify what is saved on the leaf
  nodes.

- Splitter class: Used during fitting to perform the splits.

- Predict class: Used after fitting to make predictions.

By specifying these four components a range of different tree algorithms can be
created, e.g., regression trees, classification trees, quantile regression trees
and gradient trees. Additionally to this modular structure, all other operations
are kept as vanilla as possible allowing users to easily change parts of the
code (e.g., the splitting procedure).

#### Speed

As tree-based algorithms involve expensive loops over the training dataset, it
is important that these computations are implemented in a compiled language.
adaXT implements all computationally expensive operations in
[Cython](https://cython.org/). This results in speeds similar (although a few
factors slower) than the corresponding [scikit-learn](https://scikit-learn.org)
implementations. However, due to its modular structure and the avoidance of
technical speed-ups, adaXT does not intend to provide state-of-the-art speed and
users mainly concerned with speed should consider more targeted implementations.
