## Fast Adaptable and Extendable Trees for Research

**adaXT** is a Python module for tree-based machine-learning
algorithms that is fast, adaptable and extendable. It aims to provide
researchers a more flexible workflow when developing tree-based
models.

It is distributed under the [3-Clause BSD license](https://github.com/NiklasPfister/adaXT/blob/main/LICENSE).

We encourage users and developers to report problems, request
features, ask for help, or leave general comments.

Website: [https://NiklasPfister.github.io/adaXT](https://NiklasPfister.github.io/adaXT)

### Getting started

adaXT is available on [pypi](https://pypi.org/project/adaXT) and can be
installed via pip
```bash
pip install adaXT
```
It can be used directly to fit decision trees or
random forests as illustrated in the following example:
```python
from adaXT.random_forest import RandomForest
import numpy as np

# Create toy regression data
n = 100
X = np.random.normal(0, 1, (n, 2))
Y = X + np.random.normal(0, 1, n)
Xtest = np.c_[np.linspace(-1, 1, n), np.random.uniform(0, 1, n)]

# Fit regression forest
forest = RandomForest("Regression")
forest.fit(X, Y)

# Predict on test data
Ypred = forest.predict(Xtest)

# Predict forest weight on X or Xtest
# -- can be used a similarity measure on the predictor space
weight_train = forest.predict_forest_weight()
weight_test = forest.predict_forest_weight(Xtest)
```
However, the main advantage of adaXT is its modularity and
extendability, which are discussed in more detail in the
[documentation](https://NiklasPfister.github.io/adaXT).

### Goals

The goal of adaXT is to provide a flexible and unified code-base for
various tree-based algorithms that strikes a balance between speed and
ease with which the code can be adapted and extended. It intends to
provide researchers a simple toolkit for prototyping new tree-based
algorithms.

adaXT aims to provide an intuitive user experience that is similar to
the [scikit-learn](https://scikit-learn.org) implementations of
decision trees both in terms model-based syntax and
hyperparameter. Under the hood, however, adaXT strikes a different
balance between speed and ease of adapting and extending the code.

#### Adaptable and extendable

At the heart of any tree-based algorithm is a decision tree that can
be fitted on data and then used to perform some version of
prediction. adaXT has therefore been designed with a modular decision
tree implementation that takes four input components:

- Criteria class: Used during fitting to determine splits.

- LeafBuilder class: Used during fitting to specify what is saved in
  the leaf nodes.

- Splitter class: Used during fitting to perform the splits.

- Prediction class: Used after fitting to make predictions.

By specifying these three components a range of different tree
algorithms can be created, e.g., regression trees, classification
trees, quanitle regression trees and survial trees. Additionally to
this modular structure, all other operations are kept as vanilla as
possible allowing users to easily change parts of the code (e.g., the
splitting procedure).


#### Speed

As tree-based algorithms involve evaluating expensive loops over the
dataset, it is important that these computations are implemented in a
compiled language. adaXT implements all computationally expensive
operations in [Cython](https://cython.org/). This results in speeds
similar (although a few factors slower) than the corresponding
[scikit-learn](https://scikit-learn.org) implementations. However, due
to its modular structure and the avoidence of technical speed-ups
adaXT is not intended to provide state-of-the-art speed and users
mainly concerned with speed should consider more targeted
implementations.
