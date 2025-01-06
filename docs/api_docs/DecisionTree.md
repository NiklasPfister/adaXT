# DecisionTree Class

This is the class used to construct a decision tree. It uses the following four
individual components to construct specific types of decision trees that can
then be applied to data.

- [Criteria](Criteria.md)
- [LeafBuilder](LeafBuilder.md)
- [Predictor](Predictor.md)

Instead of the user specifying all three components individually, it is also
possible to only specify the `tree_type`, which then internally selects the
corresponding default components for several established tree-algorithms, see
[user guide](../user_guide/decision_tree.md).

For more advanced modifications, it might be necessary to change how the
splitting is performed. This can be done by passing a custom
[Splitter](Splitter.md) class.

The DecisionTree class and can be imported as follows:

```python
from adaXT.decision_tree import DecisionTree
```

::: adaXT.decision_tree.DecisionTree
