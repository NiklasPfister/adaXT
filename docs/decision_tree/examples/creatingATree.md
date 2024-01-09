# Decision Trees
[Decision Trees](https://en.wikipedia.org/wiki/Decision_tree) is a predictive model, which trained on previous data can be used to predict future outcomes. It takes on a tree like structure, where each internal node splits the data set based upon some threshold value on a given feature index.

## Classification Trees
The [DecisionTree](../tree/DecisionTree.md) class is used when creating both a Regression and Classification tree.
```py
from adaXT.decision_tree import DecisionTree
from adaXT.decision_tree.criteria import Gini_index
X = [[0, 0], [1, 0]]
Y = [0, 1]
tree = DecisionTree("Classification", criteria=Gini_index)
tree.fit(X, Y)
```
In the example above we are creating and fitting a Classification
tree with the [Gini Index](../criteria/criteria.md) criteria function, and then fitting the tree with the X and Y data specified.

## Regression Trees
Regression trees work in much the same way the classification tree works, with some key differences:
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

You can also make use of a wide array of other methods, which can be seen in [DecisionTree](../tree/DecisionTree.md) class documentation.

### Prediction Probability
In the case of the Classification tree it could be of interest to get the probability of a sample to be any of the possible classes. For this you could make use of [predict_proba](../tree/DecisionTree.md#adaXT.decision_tree.DecisionTree.DecisionTree.predict_proba) function.