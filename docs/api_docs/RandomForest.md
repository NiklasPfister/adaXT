# RandomForest Class
The Random Forest, an integral component of the AdaXT package, plays a crucial role in enhancing predictions from [DecisionTrees](/docs/api_docs/DecisionTree.md).
By generating multiple instances of the [DecisionTree](/docs/api_docs/DecisionTree.md), it combines their predictive outputs to mitigate the inherent risk of overfitting commonly associated
with [DecisionTrees](/docs/api_docs/DecisionTree.md) during training, thereby optimizing overall accuracy and efficiency.

The RandomForest can be imported into your code as follows:

```python
from adaXT.random_forest import RandomForest
```

::: adaXT.random_forest.RandomForest
