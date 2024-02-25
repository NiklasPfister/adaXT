# DecisionTree Class

The decision tree lies at the core of adaXT. It uses the following
four individual components to construct specific types of decision
trees that can then be applied to data.

- [Criteria](Criteria.md)
- [LeafBuilder](LeafBuilder.md)
- [Prediction](Prediction.md)

Instead of the user specifying all three components individually, it
is also possible to only specify the ``tree_type``, which then
internally selects the corresponding default components for several
established tree-algorithms.

For more advanced modifications, it might be necessary to change the
splitting itself. This can be done by passing a custom [Splitter
class](Splitter.md).

The decision tree is implemented in the DecisionTree class and can be
loaded as follows:

```python
from adaXT.decision_tree import DecisionTree
```
::: adaXT.decision_tree.DecisionTree
