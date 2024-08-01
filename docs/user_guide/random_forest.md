# Random Forests

[Random forests](https://en.wikipedia.org/wiki/Random_forest) are ensembles of
decision trees that aggregate the predictions of the individual trees.

The [RandomForest](../api_docs/RandomForest.md) class is used in adaXT to create
random forests.

```python
from adaXT.random_forest import RandomForest
from adaXT.criteria import Gini_index
X = [[0, 0], [1, 0]]
Y = [0, 1]
forest = RandomForest("Classification", criteria=Gini_index)
forest.fit(X, Y)
```

Once the random forest has been fitted, it can be used to predict in the same
manner as the [DecisionTree](../api_docs/DecisionTree.md).

```python
forest.predict(X)  # results in [0, 1]
```

### How to chose the n_jobs parameter

When constructing a random forest model, you can adjust the 'n_jobs' parameter.
As indicated in the documentation, this setting determines the quantity of
parallel processes employed during both training and prediction stages of the
random forest algorithm. To allow users to define their own criteria functions
without requiring the Global Interpreter Lock (GIL) to be released, we chose
multiprocessing over multithreading. We implemented this using Python's built-in
[multiprocessing](https://docs.python.org/3/library/multiprocessing.html) library.

Bear in mind that initializing each new process comes with a substantial
overhead cost. Consequently, there is an inherent trade-off between the setup
time for additional processes and the workload allocated to each individual
process. For smaller datasets or models with fewer trees, this often leads to
diminishing returns as more processors are utilized. Given that 'n_jobs' governs
the number of processors employed, it's recommended to explore the performance
gains when training and predicting using various values for the parameter
'n_jobs'. Although a general rule of thumb is to never set n_jobs larger than
the number of cores available on your CPU.
