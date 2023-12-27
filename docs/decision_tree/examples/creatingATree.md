# How to create a Decision Tree
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

After the fitting we can predict new values using the tree such as:
```py
result = tree.predict(np.array([5, 0]))
print(result) # prints [1.]
```