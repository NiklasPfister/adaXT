# The DecisionTree Class

The decision tree lies at the heart of adaXT. It uses the following
four individual components to construct a specific type of decision
tree that can then be applied to data.

- [Criterion](../criteria/criteria.md)
- [Leafbuilder]()
- [Prediction]()

Instead of the user specifying these components individually. For
several commonly used decision trees, it is also possible to only
specify the ``tree_type``, which then internally selects the
corresponding default components.

For more advanced modifications, it might be necessary to change the
splitting itself. This can be done by passing a custom [Splitter
class](../splitter/splitter.md).

The decision tree is implemented in the DecisionTree class and can be
loaded as follows:

```python
from adaXT.decision_tree import DecisionTree
```
::: adaXT.decision_tree.DecisionTree