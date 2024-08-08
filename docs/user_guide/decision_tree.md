# Decision Trees

A [decision tree](https://en.wikipedia.org/wiki/Decision_tree) is a machine
learning model, which is fitted to data and then used to perform
prediction or data analysis. Decisions trees have a tree structure, where each
internal node splits the dataset based on some threshold value for a given
feature index.

The [DecisionTree](../api_docs/DecisionTree.md) class is used to create decision
trees in adaXT. On an abstract level a decision tree defines two operations:

1. **Fit**: Given training data create a list of nodes arranged in a tree
   structure, with three types of nodes: (1) A root node, (2) decision nodes and
   (3) leaf nodes. In adaXT this operation is determined by the `criteria`,
   `leaf_builder` and `splitter` parameters, as well as several other
   hyperparameters that are common across all decision trees.
1. **Predict**: Given test data create predictions for all test samples by
   propagating them through the tree structure and using the leaf node they land
   in to create a prediction. In adaXT this operation is determined by the
   `prediction` parameter.

For a given application, one needs to fully specify these two operations. In
adaXT, this can either be done by specifying an existing default `tree_type` or
by directly specifying all components manually.

## Tree types

There are several default tree types implemented in adaXT. Currently:
- ```Classification```: Prediction tasks in which the response is categorical.
- ```Regression```: Prediction tasks in which the response is continuous.
- ```Quantile```: Uncertainty quantification tasks in which the response is
continuous and the goal is to estimate one or more quantiles of the conditional
distribution of the response given the predictors.
- ```LinearRegression```: Prediction tasks in which the response is continuous and in contrast to ```Regression``` it fits a linear function in the first predictor component. This tree type can be used for derivative estimation and is used in the [Xtrapolation](https://github.com/NiklasPfister/ExtrapolationAware-Inference) method.

Moreover, if you want to create a custom tree type, this can be done by setting
the tree type to None and providing all components manually. Each of the
options is discussed in the following sections.

### Classification trees

When using the ```Classification``` tree type, the following default components are used:
- Critera class: [Entropy](../api_docs/Criteria.md#adaXT.criteria.criteria.Entropy)
- Predict class: [PredictClassification](../api_docs/#adaXT.predict.predict.PredictClassification)
- LeafBuilder class: [LeafBuilderClassification](../api_docs/#adaXT.leaf_builder.leaf_builder.LeafBuilderClassification)

Below is a short example that illustrates how to use a classification tree.

```py
from adaXT.decision_tree import DecisionTree
from adaXT.criteria import Gini_index
X = [[0, 0], [1, 0]]
Y = [0, 1]
tree = DecisionTree("Classification", criteria=Gini_index)
tree.fit(X, Y)
```

In the example above we are creating and fitting a classification tree, but
overwrite the Criteria with the
[Gini Index](../api_docs/Criteria.md#adaXT.criteria.criteria.Gini_index); it is always possible to overwrite any of the default components of a specific tree type.

### Regression trees

When using the ```Regression``` tree type, the following default components are used:
- Critera class: [Squared_error](../api_docs/Criteria.md#adaXT.criteria.criteria.Squared_error)
- Predict class: [PredictRegression](../api_docs/#adaXT.predict.predict.PredictRegression)
- LeafBuilder class: [LeafBuilderRegression](../api_docs/#adaXT.leaf_builder.leaf_builder.LeafBuilderRegression)

Regression trees work similar to classification trees as 
demonstrated by the following example:

```py
from adaXT.decision_tree import DecisionTree
X = [[0, 0], [1, 0]]
Y = [0, 1]
tree = DecisionTree("Regression")
tree.fit(X, Y)
```

### Quantile trees

When using the ```Quantile``` tree type, the following default components are used:
- Critera class: [Squared_error](../api_docs/Criteria.md#adaXT.criteria.criteria.Squared_error)
- Predict class: [PredictQuantile](../api_docs/#adaXT.predict.predict.PredictQuantile)
- LeafBuilder class: [LeafBuilderRegression](../api_docs/#adaXT.leaf_builder.leaf_builder.LeafBuilderRegression)

```py
from adaXT.decision_tree import DecisionTree
X = [[0, 0], [1, 0]]
Y = [0, 1]
tree = DecisionTree("Quantile")
tree.fit(X, Y)
```



### LinearRegression trees


### Custom tree types

## Using the fitted tree

### Prediction

After the fitting we can predict new values using the
[predict](../api_docs/DecisionTree.md#adaXT.decision_tree.DecisionTree.DecisionTree.predict)
method of the tree.

```py
result = tree.predict(np.array([5, 0]))
print(result) # prints [1.]
```

This works for both regression and classification trees, but the output is
different. Both tree types first locate the nodes in which the new samples fall
based on their fitted structure. The classification tree then calculates the
proportion of each class within those nodes, chooses the class with the highest
proportion and returns that value as the prediction. In contrast the regression
tree only calculates the mean value of all samples within each of those node and
returns that as the prediction.

### Prediction probability

For classification it can useful to get a notion of classification probability
for each of the possible classes instead of the predicted class label. With
classification trees a commonly used option is to output the frequency each
class has in the leaf nodes. In adaXT this can be done using the
[predict_proba](../api_docs/DecisionTree.md#adaXT.decision_tree.DecisionTree.DecisionTree.predict_proba)
function.

It works in much the same fashion as the prediction function, but importantly
only applies to classification trees. Given the above fitted classification
tree, we have the following:

```py
result = tree.predict_proba([5, 0])
print(result) # prints  array([[0., 1.]]
```

Here the function returns a
[numpy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
with the probability of an element being either of the classes. To get the list
of classes (in the correct order) use .classes on the
[DecisionTree](../api_docs/DecisionTree.md).
