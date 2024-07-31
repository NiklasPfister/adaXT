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

Below we provide a short overview of honest splitting and explain how to use
it in adaXT.

## Honest splitting in adaXT
In its most basic form honest splitting consists of dividing the training data into two disjoint subsets; a fitting set (called ```fitting_indices``` in the code base) used to create a tree and a prediction set (called ```prediction_indices``` in the code base) used to populate the leaf nodes that are later used when predicting.

This basic honest split can be achieved manually in adaXT as follows:
```python
from adaXT.decision_tree import DecisionTree

# Create toy training data
n = 100
X = np.random.normal(0, 1, (n, 2))
Y = 2.0 * (X[:, 0] > 0) + np.random.normal(0, 1, n)

# Split training data
fitting_indices = np.arange(0, int(n/2))
prediction_indices = np.arange(int(n/2)+1, n)

# Fit a regression decision tree using only fitting_indices
tree = DecisionTree("Regression", max_depth=5)
tree.fit()


```
