# Decision Trees
A [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree) is a predictive model, which trained on previous data, can be used to predict future outcomes. It takes on a treelike structure, where each internal node splits the dataset based upon some threshold value on a given feature index.

The [DecisionTree](../tree/DecisionTree.md) class is used when creating both a Regression and Classification tree.

## Classification Trees
```py
from adaXT.decision_tree import DecisionTree
from adaXT.decision_tree.criteria import Gini_index
X = [[0, 0], [1, 0]]
Y = [0, 1]
tree = DecisionTree("Classification", criteria=Gini_index)
tree.fit(X, Y)
```
In the example above we are creating and fitting a Classification
tree with the [Gini Index](../criteria/criteria.md#adaXT.decision_tree.criteria.Gini_index) criteria function using the X and Y data specified.

## Regression Trees
Regression trees work in much the same way classification trees works, with some key differences:
```py
from adaXT.decision_tree import DecisionTree
from adaXT.decision_tree.criteria import Squared_error
X = [[0, 0], [1, 0]]
Y = [0, 1]
tree = DecisionTree("Regression", criteria=Squared_error)
tree.fit(X, Y)
```
And it's that simple. You only have to specify that you are now using a regression tree instead of a classification tree, and adaXT takes care of the rest. The reason for this specification becomes clear in the following section.

## Using the fitted tree

### Prediction
After the fitting we can predict new values using the tree with the [predict](../tree/DecisionTree.md#adaXT.decision_tree.DecisionTree.DecisionTree.predict) function.
```py
result = tree.predict(np.array([5, 0]))
print(result) # prints [1.]
```
Now you might be tempted to say that the Regression Tree and the Classification Tree always produce the same result when predicting, but that is not the case. The Classification tree takes all the nodes within the leaf and calculates the proportion of each class within the node. It then chooses the class with the highest proportion and returns that value as the prediction.
On the other hand the Regression tree just takes the mean value of all the samples within the leaf node and returns that as the prediction.
An important distinction is, the classification tree will never produce a prediction that is not within the training data, however the regression tree can produce a prediction outside it's training data.

### Prediction Probability
In the case of the Classification tree it could be of interest to get the probability of a sample to be any of the possible classes. For this you could make use of [predict_proba](../tree/DecisionTree.md#adaXT.decision_tree.DecisionTree.DecisionTree.predict_proba) function.

It works in much the same fashion as the prediction, but with a key distinction. It only works for the **classification tree** as it returns the proportion of each class within the leaf node. So given the previously mentioned classification tree, we have the following:
```py
result = tree.predict_proba([5, 0])
print(result) # prints (array([0., 1.]), array([[0., 1.]]))
```
Here it instead returns a tuple of two [numpy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html), where the first element is the classes within the training data, and the second is the probability of input being anyone of the classes.