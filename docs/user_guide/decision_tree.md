# Decision Trees

A [decision tree](https://en.wikipedia.org/wiki/Decision_tree) is a machine
learning model, which is fitted to data and then used for prediction or data
analysis. Decisions trees have a tree structure, where each internal node splits
the dataset based on a threshold value for a given feature index.

The [DecisionTree](../api_docs/DecisionTree.md) class is used to create decision
trees in adaXT. On an abstract level a decision tree defines two procedures:

1. **Fit**: Given training data, create a list of nodes arranged in a tree
   structure consisting of three types of nodes: (i) A root node, (ii) decision
   nodes and (iii) leaf nodes. In adaXT the fit procedure is determined by the
   `criteria`, `leaf_builder` and `splitter` parameters, as well as several
   other hyperparameters that are common across all decision trees.
2. **Predict**: Given test data, create predictions for all test samples by
   propagating them through the tree structure and using the leaf node they land
   in to create a prediction. In adaXT the predict procedure is determined by
   the `predict` parameter.

For a given application, one needs to fully specify these two procedures. In
adaXT, this can either be done by specifying an existing default `tree_type` or
by directly specifying all components manually.

## Tree types

There are several default tree types implemented in adaXT.

- `Classification`: For prediction tasks in which the response is categorical.
- `Regression`: For prediction tasks in which the response is continuous.
- `Quantile`: For uncertainty quantification tasks in which the response is
  continuous and the goal is to estimate one or more quantiles of the
  conditional distribution of the response given the predictors.
- `Gradient`: For tasks in which one aims to estimate (directional) derivatives
  of the response given the predictors. A related tree type is used in the
  [Xtrapolation](https://github.com/NiklasPfister/ExtrapolationAware-Inference)
  method.

The defaults for each of these tree types is set in the `BaseModel` class, which
is extended by both the `DecisionTree` and `RandomForest` classes. Moreover, if
you want to create a custom tree type, this can be done by setting the tree type
to `None` and providing all components manually. Each of these options is
discussed in the following sections.

### Classification trees

For the `Classification` tree type, the following default components are used:

- Criteria class:
  [Entropy](../api_docs/Criteria.md#adaXT.criteria.criteria.Entropy)
- Predict class:
  [PredictorClassification](../api_docs/Predictor.md#adaXT.predictor.predictor.PredictClassification)
- LeafBuilder class:
  [LeafBuilderClassification](../api_docs/LeafBuilder.md#adaXT.leaf_builder.leaf_builder.LeafBuilderClassification)

Below is a short example that illustrates how to use a classification tree.

```py
import numpy as np
from adaXT.decision_tree import DecisionTree
from adaXT.criteria import Gini_index

X = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1],
              [1, 1], [1, -1], [-1, -1], [-1, 1]])
Xtest = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
Y = [0, 1, 0, 1, 0, 0, 1, 1]

tree = DecisionTree("Classification", criteria=Gini_index)
tree.fit(X, Y)
print(tree.predict(Xtest))
print(tree.predict(Xtest, predict_proba=True))
```

In this example we created and fit a classification tree using training data and
then used the fitted tree to predict the response at the training data. When
initializing the tree we changed the default criteria to the
[Gini Index](../api_docs/Criteria.md#adaXT.criteria.criteria.Gini_index); it is
always possible to overwrite any of the default components of a specific tree
type. Classification trees use a majority vote in each of the leaf nodes to
decide which class to predict and ties are broken by selecting the smaller
class. To predict the class probabilities instead of the class labels, one can
add `predict_proba=True` as a keyword argument.

For classification trees it is also possible, using the `predict_proba` method,
to output the proportions of each class instead of only the majority class. That
method returns an array with the probability of an element being either of the
classes. To get the list of classes (in the correct order), one can use
`.classes` attribute of a fitted decision tree.

Note that in this example, the decision tree was too constrained to fit the
data, if one chooses the `max_depth` parameter larger the tree can perfectly fit
the data.

### Regression trees

For the `Regression` tree type, the following default components are used:

- Criteria class:
  [Squared_error](../api_docs/Criteria.md#adaXT.criteria.criteria.Squared_error)
- Predict class:
  [PredictRegression](../api_docs/Predictor.md#adaXT.predict.predict.PredictRegression)
- LeafBuilder class:
  [LeafBuilderRegression](../api_docs/LeafBuilder.md#adaXT.leaf_builder.leaf_builder.LeafBuilderRegression)

Regression trees work similar to classification trees as illustrated in the
following example:

```py
import numpy as np
from adaXT.decision_tree import DecisionTree

n = 100
X = np.random.normal(0, 1, (n, 2))
Y = 2.0 * (X[:, 0] > 0) + np.random.normal(0, 0.25, n)
Xnew = np.array([[1, 0], [-1, 0]])

tree = DecisionTree("Regression", min_samples_leaf=20)
tree.fit(X, Y)
print(tree.predict(Xnew))
```

### Quantile trees

For the `Quantile` tree type, the following default components are used:

- Criteria class:
  [Squared_error](../api_docs/Criteria.md#adaXT.criteria.criteria.Squared_error)
- Predict class:
  [PredictorQuantile](../api_docs/Predictor.md#adaXT.predictor.predictor.PredictQuantile)
- LeafBuilder class:
  [LeafBuilderRegression](../api_docs/LeafBuilder.md#adaXT.leaf_builder.leaf_builder.LeafBuilderRegression)

Quantile trees are the building block for quantile random forests that were
proposed by
[Meinshausen, 2006](https://jmlr.csail.mit.edu/papers/v7/meinshausen06a.html).
They have the same interface as regression and classification trees, but the
predict method takes the additional mandatory keyword `quantile` which specifies
which quantiles to estimate. The following example illustrates this:

```py
import numpy as np
from adaXT.decision_tree import DecisionTree

n = 100
X = np.random.normal(0, 1, (n, 2))
Y = 10.0 * (X[:, 0] > 0) + np.random.normal(0, 1, n)
Xnew = np.array([[1, 0], [-1, 0]])

tree = DecisionTree("Quantile", min_samples_leaf=20)
tree.fit(X, Y)
print(tree.predict(Xnew, quantile=[0.1, 0.5, 0.9]))
```

As seen from this example, the quantiles do not need to be specified prior to
prediction and it is possible to predict several quantiles simultaneously.

### Gradient trees

For the `Gradient` tree type, the following default components are used:

- Criteria class:
  [Partial_quadratic](../api_docs/Criteria.md#adaXT.criteria.criteria.Partial_quadratic)
- Predict class:
  [PredictLocalPolynomial](../api_docs/Predictor.md#adaXT.predict.predict.PredictLocalPolynomial)
- LeafBuilder class:
  [LeafBuilderLocalPolynomial](../api_docs/LeafBuilder.md#adaXT.leaf_builder.leaf_builder.LeafBuilderPartialLinear)

Gradient trees are a non-standard type of trees that allow estimation of
derivates (in the first coordinate) of the conditional expectation function. The
provided implementation is a slight modification of the procedure used in the
[Xtrapolation](https://github.com/NiklasPfister/ExtrapolationAware-Inference)
method. Instead of using the raw response values, we generally recommend to
first use your favorite regression procedure (e.g., a regular random forest or a
neural network) and use the resulting fitted values instead of the raw response.
This has the advantage of first removing noise and then using the gradient tree
as an additional step to estimate the gradients.

Gradient trees are similar to regression trees but instead of fitting a constant
in each leaf they fit a quadratic function in the first predictor variable
$X[:, 0]$. This allows them to fit quadratic functions in the first coordinate
without splitting, as illustrated in the following example:

```python
import numpy as np
import matplotlib.pyplot as plt
from adaXT.decision_tree import DecisionTree

n = 200
X = np.random.normal(0, 1, (n, 1))
Y = X[:, 0] ** 2 + np.random.normal(0, 0.5, n)

tree_reg = DecisionTree("Regression", min_samples_leaf=50)
tree_reg.fit(X, Y)
Yhat_reg = tree_reg.predict(X)

tree_grad = DecisionTree("Gradient", min_samples_leaf=50)
tree_grad.fit(X, Y)
Yhat_grad = tree_grad.predict(X,  order=[0, 1])

plt.scatter(X, Y, label='raw data')
plt.scatter(X, Yhat_reg, label='Regression tree')
plt.scatter(X, Yhat_grad[:, 0], label='Gradient tree (order 0)')
plt.scatter(X, Yhat_grad[:, 1], label='Gradient tree (order 1)')
plt.legend()
plt.show()
```

### Custom tree types

It is also possible to manually specify the tree type. This is particularly
useful when you have custom components for the tree and do not want to use any
of the default classes. To do this simply set `tree_type` to None and provide
the `criteria`, `predictor` and `leaf_builder` classes when initializing the tree.

## Further functionality

adaXT provides various additional functionality, each of which is discussed in
other sections of the user guide.

- [Tree-based weights](tree_based_weights.md): A fitted
  decision tree provides a similarity notion on the predictor space that has
  some useful properties. Check out this section to see how this can be used.
- [Visualizations and analysis tools](vis_and_analysis.md): There are
  several function available that can help with analyzing a fitted decision
  tree.
