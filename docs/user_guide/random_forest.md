# Random Forests

[Random forests](https://en.wikipedia.org/wiki/Random_forest) are ensembles of
decision trees that aggregate the predictions of all individual trees. By
combining multiple decision trees, each trained on slightly different training
data, random forests naturally avoid overfitting and tend to generalize much
better than decision trees alone.

The [RandomForest](../api_docs/RandomForest.md) class is used in adaXT to create
random forests. It takes mostly the same parameters as the
[DecisionTree](../api_docs/DecisionTree.md) class, as illustrated in the
example below.

```python
import numpy as np
import matplotlib.pyplot as plt
from adaXT.random_forest import RandomForest
from adaXT.criteria import Partial_linear
from adaXT.leaf_builder import LeafBuilderPartialLinear
from adaXT.predict import PredictLocalPolynomial

# Training and test data
n = 200
Xtrain = np.random.uniform(-1, 1, (n, 1))
Ytrain = np.sin(Xtrain[:, 0]*np.pi) + np.random.normal(0, 0.2, n)
Xtest = np.linspace(-1, 1, 50).reshape(-1, 1)

# Fit a regular regression forest and a regression forest with linear splits
rf = RandomForest("Regression", min_samples_leaf=30)
rf_lin = RandomForest("Regression",
                      criteria=Partial_linear,
                      leaf_builder=LeafBuilderPartialLinear,
                      predict=PredictLocalPolynomial,
                      min_samples_leaf=30)
rf.fit(Xtrain, Ytrain)
rf_lin.fit(Xtrain, Ytrain)

# Plot the resulting fits on Xtest
plt.scatter(Xtrain, Ytrain, alpha=0.5)
plt.plot(Xtest, rf.predict(Xtest), label="regular RF")
plt.plot(Xtest, rf_lin.predict(Xtest, order=0), label="linear RF")
plt.legend()
plt.show()
```

In this example, we fit a regular regression forest (which uses the
[Squared_error](../api_docs/Criteria.md)) and a regression forest that uses the
[Partial_linear](../api_docs/Criteria.md) splitting criteria and predicts a
linear function in each leaf. As can be seen when running this example, the
forest with the linear splits is able to produce a better fit when both forests
are grown similarly deep.

### Implementation details

The RandomForest class has essentially two functionalities. Firstly, it
implements several sampling schemes that split the data and secondly it handles
initializing and fitting of all of the DecisionTrees based on these splits.

#### Sampling schemes

In the basic random forest as proposed by
[Breiman, 2001](https://doi.org/10.1023/A:1010933404324), one draws for a each
decision tree a new random bootstrap sample from the full the training data
(that is, with replacement) and uses only the boostrap data to fit the decision
tree. The idea behind this subsampling is that while each individual tree may
overfit to its training sample, the overfitting averages out as each tree
overfits to a different data set.

While the original bootstrap sampling scheme is the most common approach, other
sampling schemes have also been proposed and the choice can have a significant
impact on the generalization performance of the random forest. adaXT currently
implements the following sampling schemes that are selected via the `sampling`
parameter with more refined settings available via the `sampling_parameters`
parameter:

- `bootstrap`: Subsamples are drawn with replacement from the full training
  sample. The size of the subsamples can be controlled with the
  `sampling_parameters`.
- `honest_tree`: This is a weak form of [honest splitting](honest_splitting.md)
  which splits the training data for each tree into two and uses one split to
  fit the tree structure and the other split to populate the leafs.
- `honest_forest`: This is a strong form of
  [honest splitting](honest_splitting.md) which splits the training data once
  and then uses one split with bootstrapping to fit the tree structures and the
  other part to populate the leafs of all trees.
- `None`: The entire training sample is used to fit each tree. This is generally
  not recommended as long as the decision trees are deterministic.

#### Parallelization via multiprocessing

Since each decision tree can be fitted separately, it is possible to parallelize
the fitting of a random forest. In adaXT we chose to use multiprocessing, using
Python's built-in
[multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
library, instead of multithreading. The advantage of this approach is that it
allows users to define their own criteria functions without requiring the Global
Interpreter Lock (GIL) to be released. However, it adds an additional overhead
cost related to managing the individual processes. Consequently, there is an
inherent trade-off between the setup time for additional processes and the
workload allocated to each individual process. For smaller datasets or models
with fewer trees, this often leads to diminishing returns as more processors are
utilized. Users are therefore encouraged to select the `n_jobs` parameter with
this trade-off in mind.

In order to avoid making the random forest code too complex, we have separated
the multiprocessing logic into a separate class called
[ParallelModel](../api_docs/Parallel.md#adaXT.parallel.ParallelModel). The
[ParallelModel](../api_docs/Parallel.md#adaXT.parallel.ParallelModel) offers a variety of
methods capable of computing functions in parallel. With this it aims to reduce
the complexity of working with multiprocessing.

When working with the [ParallelModel](../api_docs/Parallel.md#adaXT.parallel.ParallelModel)
we generally advise on creating the parallel functions on the module level
instead of being class methods. Class method parallelization often leads to
AttributeErrors when attempting to access instance dependent attributes through
self due to the nature of multiprocessings use of
[pickle](https://docs.python.org/3/library/pickle.html). Instead working with
functions defined on the module level allows for seamless use of the
multiprocessing as it is safe for serialization. As an example, take a look at 
the functions defined in the [RandomForest source
code](https://github.com/NiklasPfister/adaXT/blob/main/src/adaXT/random_forest/random_forest.py).
