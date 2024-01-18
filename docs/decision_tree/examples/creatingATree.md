# Decision trees
A [decision tree](https://en.wikipedia.org/wiki/Decision_tree) is a predictive regression and classification model, which trained on previous data, can be used to predict future responses. Decisions trees have a tree structure, where each internal node splits the dataset based on some threshold value for a given feature index.

The [DecisionTree](../tree/DecisionTree.md) class is used when creating both a regression and classification tree.

## Classification trees
```py
from adaXT.decision_tree import DecisionTree
from adaXT.decision_tree.criteria import Gini_index
X = [[0, 0], [1, 0]]
Y = [0, 1]
tree = DecisionTree("Classification", criteria=Gini_index)
tree.fit(X, Y)
```
In the example above we are creating and fitting a classification
tree with the [Gini Index](../criteria/criteria.md#adaXT.decision_tree.criteria.Gini_index) criteria function using the X and Y data specified as training data.

## Regression trees
Regression trees work similar to classification trees, with one small difference demonstrated by the following example:
```py
from adaXT.decision_tree import DecisionTree
from adaXT.decision_tree.criteria import Squared_error
X = [[0, 0], [1, 0]]
Y = [0, 1]
tree = DecisionTree("Regression", criteria=Squared_error)
tree.fit(X, Y)
```
You have to specify that you are now using a regression tree instead of a classification tree, and adaXT takes care of the rest. The reason for this specification is that regression and classification trees need to have some small differences in saved objects. This is for example relevant when making predictions as shown next.

## Using the fitted tree

### Prediction
After the fitting we can predict new values using the [predict](../tree/DecisionTree.md#adaXT.decision_tree.DecisionTree.DecisionTree.predict) method of the tree.
```py
result = tree.predict(np.array([5, 0]))
print(result) # prints [1.]
```
This works for both regression and classification trees, but the output is different. Both tree types first locate the nodes in which the new samples fall based on their fitted structure. The classification tree then calculates the proportion of each class within those nodes, chooses the class with the highest proportion and returns that value as the prediction. In contrast the regression tree only calculates the mean value of all samples within each of those node and returns that as the prediction.

### Prediction Probability
For classification it can useful to get a notion of classification probability for each of the possible classes instead of the predicted class label. With classification trees a commonly used option is to output the frequency each class has in the leaf nodes. In adaXT this can be done using the [predict_proba](../tree/DecisionTree.md#adaXT.decision_tree.DecisionTree.DecisionTree.predict_proba) function.

It works in much the same fashion as the prediction function, but importantly only applies to classification trees. Given the above fitted classification tree, we have the following:
```py
result = tree.predict_proba([5, 0])
print(result) # prints (array([0., 1.]), array([[0., 1.]]))
```
Here it the function returns a tuple of two [numpy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html), where the first element is the classes within the training data, and the second is the class probabilities for the new X values.
