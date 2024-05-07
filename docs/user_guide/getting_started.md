# Getting started with adaXT

A [decision tree](https://en.wikipedia.org/wiki/Decision_tree) is a
machine learning model, which can trained or fitted to data in order
to perform prediction or data analysis. Decisions trees have a tree
structure, where each internal node splits the dataset based on some
threshold value for a given feature index.

The [DecisionTree](../api_docs/DecisionTree.md) class is used when
creating both a regression and classification tree.

## Classification trees
```py
from adaXT.decision_tree import DecisionTree
from adaXT.criteria import Gini_index
X = [[0, 0], [1, 0]]
Y = [0, 1]
tree = DecisionTree("Classification", criteria=Gini_index)
tree.fit(X, Y)
```
In the example above we are creating and fitting a classification tree
with the [Gini
Index](../api_docs/Criteria.md#adaXT.criteria.criteria.Gini_index)
criteria function using the X and Y data specified as training data.

## Regression trees

Regression trees work similar to classification trees, with one small
difference demonstrated by the following example:
```py
from adaXT.decision_tree import DecisionTree
from adaXT.criteria import Squared_error
X = [[0, 0], [1, 0]]
Y = [0, 1]
tree = DecisionTree("Regression", criteria=Squared_error)
tree.fit(X, Y)
```
You have to specify that you are now using a regression tree instead
of a classification tree, and adaXT takes care of the rest. The reason
for this specification is that regression and classification trees
need to have some small differences in saved objects. This is for
example relevant when making predictions as shown next.

## Using the fitted tree

### Prediction

After the fitting we can predict new values using the
[predict](../api_docs/DecisionTree.md#adaXT.decision_tree.DecisionTree.DecisionTree.predict)
method of the tree.

```py
result = tree.predict(np.array([5, 0]))
print(result) # prints [1.]
```

This works for both regression and classification trees, but the
output is different. Both tree types first locate the nodes in which
the new samples fall based on their fitted structure. The
classification tree then calculates the proportion of each class
within those nodes, chooses the class with the highest proportion and
returns that value as the prediction. In contrast the regression tree
only calculates the mean value of all samples within each of those
node and returns that as the prediction.

### Prediction probability

For classification it can useful to get a notion of classification
probability for each of the possible classes instead of the predicted
class label. With classification trees a commonly used option is to
output the frequency each class has in the leaf nodes. In adaXT this
can be done using the
[predict_proba](../api_docs/DecisionTree.md#adaXT.decision_tree.DecisionTree.DecisionTree.predict_proba)
function.

It works in much the same fashion as the prediction function, but
importantly only applies to classification trees. Given the above
fitted classification tree, we have the following:
```py
result = tree.predict_proba([5, 0])
print(result) # prints  array([[0., 1.]]
```
Here the function returns a [numpy
arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
with the probability of an element being either of the classes. To get
the list of classes (in the correct order) use .classes on the
[DecisionTree](../api_docs/DecisionTree.md).


## RandomForest
The RandomForest algorithm operates similarly to the DecisionTree,
yet it has currently been designed exclusively for use with both Classification and Regression DecisionTrees.
As such, you can create a [RandomForest](/docs/api_docs/RandomForest.md) as shown:
```python
from adaXT.random_forest import RandomForest
from adaXT.criteria import Gini_index
X = [[0, 0], [1, 0]]
Y = [0, 1]
forest = RandomForest("Classification", criteria=Gini_index)
forest.fit(X, Y)
```

Once the RandomForest has been fitted, it can be used to predict in the same manner as the [DecisionTree](/docs/api_docs/DecisionTree.md).

```python
forest.predict(X)  # results in [0, 1]
```


### How to chose the n_jobs parameter of RandomForest
When constructing a random forest model, you can adjust the 'n_jobs' parameter. As indicated in the documentation, this setting determines the quantity of parallel processes employed during both training and prediction stages of the random forest algorithm. To allow users to define their own criteria functions without requiring the Global Interpreter Lock (GIL) to be released, we chose multiprocessing over multithreading. We implemented this using Python's built-in [multiprocessing library](https://docs.python.org/3/library/multiprocessing.html).

Bear in mind that initializing each new process comes with a substantial overhead cost. Consequently, there is an inherent trade-off between the setup time for additional processes and the workload allocated to each individual process. For smaller datasets or models with fewer trees, this often leads to diminishing returns as more processors are utilized. Given that 'n_jobs' governs the number of processors employed, it's recommended to explore the performance gains when training and predicting using various values for the parameter 'n_jobs'. Although a general rule of thumb is to never set n_jobs larger
than the number of cores available on your CPU.
