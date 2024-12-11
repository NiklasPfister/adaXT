# Model Selection
To allow for model selection, adaXT's DecisionTree and RandomForest are both
compatible with scikit-learn's [model
selection](https://scikit-learn.org/1.5/modules/grid_search.html#exhaustive-grid-search).
This allows for use with classes such as
[GridSearchCV](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.GridSearchCV.html)
and
[Pipeline](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html).


## Using GridSearchCV with adaXT
Here we introduce the difference when using scikit-learn's own
DecisionTreeClassifier and adaXT's DecisionTree with the GridSearchCV. First,
there is the initial setup:

```python
from adaXT.decision_tree import DecisionTree
from adaXT.criteria import Gini_index, Entropy
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import time

n = 20000
m = 5

X = np.random.uniform(0, 100, (n, m))
Y = np.random.randint(1, 3, n)

param_grid = {
    "max_depth": [3, 5, 10, 20, 100],
    "min_samples_split": [2, 5, 10],
}

param_grid_ada = param_grid | {"criteria": [Gini_index, Entropy]}
param_grid_sk = param_grid | {"criterion": ["gini", "entropy"]}
```
First we import the necessary components and setup the parameter grids of the
two decision trees. Here we note our first difference. In adaXT we generally
stick to the naming theme of calling it a criteria rather than a criterion, when
passing input to the tree.
Second, when passing in the possible parameters instead of it being a string as
in DecisionTreeClassifier we instead pass in the two classes Gini_index or
Entropy (or perhaps your own [implementation](creatingCriteria.md)). Next, we
can define and fit the GridSearchCV instance.

```python
grid_search_ada = GridSearchCV(
    estimator=DecisionTree(tree_type="Classification"),
    param_grid=param_grid_ada,
    cv=5,
    scoring="accuracy",
)

grid_search_sk = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=param_grid_sk,
    cv=5,
    scoring="accuracy",
)

grid_search_ada.fit(X, Y)
grid_search_sk.fit(X, Y)

print("Best Hyperparameters ada: ", grid_search_ada.best_params_)
print("Best Hyperparameters sklearn: ", grid_search_sk.best_params_)
print("Best accuracy ada: ", grid_search_ada.best_score_)
print("Best accuracy sklearn: ", grid_search_sk.best_score_)

```
And that is it. The workflow resembles what you are used to with only a few
minor tweaks. And the same can be said when using the Pipeline.


## Using Pipeline

AdaXT makes it easy to use any preprocessing from sklearn when fitting as adaXT
is compatible with sklearn's
[Pipeline](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html).
An example of the use case can be seen here:
```python
from adaXT.decision_tree import DecisionTree
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = Pipeline(
    [("scaler", StandardScaler()), ("tree", DecisionTree("Classification"))]
)

print(pipe.fit(X_train, y_train).score(X_test, y_test))
print(pipe.set_params(tree__max_depth=5).fit(X_train, y_train).score(X_test, y_test))
```

Again, there are only minor changes between how the DecisionTree and the
DecisionTreeClassifier would be used. The only difference is, that we have to
specify, that the DecisionTree is for classification. However, one could also pass
in a custom criteria, leaf_builder, and predictor and the DecisionTree would still work
fine with the Pipeline.






