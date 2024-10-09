# RandomForest Class

This is the class used to construct a random forest. Random forests consist of
multiple individual decision trees that are trained on subsets of the data and
then combined via averaging. This can greatly improve the generalization
performance by avoiding the tendency of decision trees to over fit to the
training data. Since random forest learn individual trees many of the
parameters and functionality in this class overlaps with the
[DecisionTree](DecisionTree.md) class.

The RandomForest can be imported as follows:

```python
from adaXT.random_forest import RandomForest
```

::: adaXT.random_forest.random_forest
    options:
      members:
        - RandomForest
