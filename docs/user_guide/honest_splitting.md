# Honest splitting

Decision trees generally use the training data twice during fitting: Once for
deciding where to create splits resulting in the tree structure and once to
create a prediction for each leaf node.

While such a double use of the data may lead to overfitting, the effect is often
negligible, in particular when otherwise regularizing (e.g., fitting a forest or
constraining the maximum depth). Nevertheless it can be beneficial to adapt the
splitting to improve generalization performance. One way of achieving this is
via _honest splitting_
([Athey and Imbens, 2016](https://doi.org/10.1073/pnas.1510489113)). The
approach was originally developed in the context of causal effect estimation,
where the bias is shown to be reduced by honest splitting allowing for inference
on the causal effects.

Below we provide a short overview of honest splitting and explain how to use it
in adaXT.

## Honest splitting in adaXT

In its most basic form honest splitting consists of dividing the training data
into two disjoint subsets; a fitting set (called `fitting_indices` in the code
base) used to create a tree and a prediction set (called `prediction_indices` in
the code base) used to populate the leaf nodes that are later used when
predicting.

This basic version of honest splitting can be achieved manually in adaXT by
using the `refit_leaf_nodes` function as follows:

```python
from adaXT.decision_tree import DecisionTree
import numpy as np

# Create toy training data
n = 100
X = np.random.normal(0, 1, (n, 2))
Y = 2.0 * (X[:, 0] > 0) + np.random.normal(0, 0.25, n)

# Split training data
fitting_indices = np.arange(0, int(n/2))
prediction_indices = np.arange(int(n/2)+1, n)

# Fit a regression decision tree using only fitting_indices
tree = DecisionTree("Regression", max_depth=5)
tree.fit(X, Y, sample_indices=fitting_indices)

# Predict on two test points
print(tree.predict(np.array([[-1, 0], [1, 0]])))

# Refit tree using only prediction_indices
tree.refit_leaf_nodes(X, Y, sample_indices=prediction_indices)

# Predict on the same two test points
print(tree.predict(np.array([[-1, 0], [1, 0]])))
```

Using the refitting function manually is tedious when fitting random forests.
Therefore the RandomForest class has an optional parameter to perform honest
splitting. The precise behaviour is controlled by the parameters `sampling` and
`sampling_parameter`. Currently there are two version of honest splitting
available:

- **honest_tree**: In this case, for each tree in the forest, a new split into
  `fitting_indices` and `prediction_indices` is randomly drawn and used to fit
  and refit the tree, respectively. Unlike in classical random forest that draw
  a bootstrap sample, this approaches uses all training data for each tree,
  however it never uses the same sample in the same tree to both create splits
  and populate the leaf nodes. The relative sizes of the two splits are
  controlled by `sampling_parameter`.
- **honest_forest**: In this case, the data is split only once into
  `fitting_indices` and `prediction_indices`. Again the relative sizes are
  determined by the first value of the `sampling_parameter`. Then, for each tree
  two bootstrap samples from `fitting_indices` and `prediction_indices` are
  drawn and used to fit and refit the tree, respectively. The sizes of the
  bootstrap samples are controlled by the second value of the
  `sampling_parameter`. This approach ensures a total separation between the
  data used to create the splits and the data used to populate the leafs.
  Importantly, this guarantees (for independent training samples) that the
  resulting random forest weights (extracted using `predict_forest_weights`) are
  independent of the samples corresponding to the `prediction_indices`.
