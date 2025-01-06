# Creating a custom Predictor

## General overview of the Predictor

Like other elements in adaXT, it is possible to create a custom
[Predictor](../api_docs/Predictor.md). You can start by creating a new
.pyx file using the following template:

```cython
from adaXT.predictor cimport Predictor

cdef class MyPredictorClass(Predictor):

  cdef:
    # attribute_type attribute_name

  def __init__(
    self,
    double[:, ::1] X,
    double[:, ::1] Y,
    object root,
    **kwargs):
  super().__init__(X, Y, root, **kwargs)
  # Any custom initialization you would need for your predictor class
  # If you don't have any, you don't need to define the __init__ function.
  

  def predict(self, cnp.ndarray X, **kwargs) -> np.ndarray: 
    # Define your own custom predict function

  @staticmethod
  def forest_predict(cnp.ndarray X_old, cnp.ndarray Y_old, cnp.ndarray X_new,
                      trees: list[DecisionTree], parallel: ParallelModel,
                      **kwargs) -> np.ndarray:
    # Define special handling for the RandomForest predict.
    # If it is not defined, then the RandomForest will take the mean of all the
    # predict for its estimators.
  

```

The template includes three main components:

1. \_\_init\_\_ function: This function is used to initialize the class. Because Cython
   removes a lot of the boilerplate with default Python classes Cython, you cannot
   add attributes to a cdef class without explicitly defining them. The \_\_init\_\_
   function allows you to initialize these attributes after you have defined them above.
   If you do not need additional attributes, you can skip this step.
3. predict method: This method is used to compute predictions for the given input X
   values. It is a standard Python method and can be used like any other. Within this
   method, you have access to the general attributes of the
   [Predictor](../api_docs/Predictor.md) class, including the number of features and
   the root node object, which can be used to traverse the tree.
4. forest_predict method: This static method aggregates predictions across multiple
   trees for forest predictions. It enables parallel processing across trees. If your
   custom Predictor simply averages tree predictions, you can inherit this method
   from the base Predictor class.

## Example of creating a Predictor

To illustrate each component, we go over the PredictorQuantile class, which is used
for quantile regression trees and forests. It does not add any additional attributes
so the \_\_init\_\_ function is not needed in this case.

### The predict method

In quantile regression we want to predict the quantiles of the conditional distribution
instead of just the conditional mean as in regular regression. For a single tree this can
be done with the following predict method:

```cython
cdef class PredictorQuantile(Predictor):
    def predict(self, cnp.ndarray X, **kwargs) -> np.ndarray:
        cdef:
            int i, cur_split_idx, n_obs
            double cur_threshold
            object cur_node
            cnp.ndarray prediction
        if "quantile" not in kwargs.keys():
            raise ValueError(
                        "quantile called without quantile passed as argument"
                    )
        quantile = kwargs['quantile']
        # Make sure that x fits the dimensions.
        n_obs = X.shape[0]
        # Check if quantile is an array
        if isinstance(quantile, Sequence):
            prediction = np.empty((n_obs, len(quantile)), dtype=DOUBLE)
        else:
            prediction = np.empty(n_obs, dtype=DOUBLE)

        for i in range(n_obs):
            cur_node = self.root
            while isinstance(cur_node, DecisionNode):
                cur_split_idx = cur_node.split_idx
                cur_threshold = cur_node.threshold
                if X[i, cur_split_idx] < cur_threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child

            prediction[i] = np.quantile(self.Y.base[cur_node.indices, 0], quantile)
        return prediction

```

Here, we first define the types of the variables used. This allows Cython to
optimize the code, which leads to a faster prediction runtime.

Next, we check the kwargs for the key `quantile`. Any keyword arguments passed
to the DecisionTree.predict is passed directly to the Predictor.predict, meaning
that we can access the desired quantile from the predict signature without having
to change anything else. As we want to allow for multiple quantiles to be
predicted at the same time, we have to initalize the `prediction` variable differently
depending on whether `quantile` is a Sequence or just a single element.

Finally, we iterate over the tree: For every observation, we go to the root node
and loop as long as we are in a DecisionNode. In each step, we check if we split
to the left or the right, and traverse down the tree. Once `cur_node` is no longer
an instance of the DecisionNode, we know that we have reached a LeafNode.
We can access all Y values via `self.Y.base` ('.base' has to be added,
as we are indexing with a list of elements) and the indices of the elements
within the LeafNode via `cur_node.indices`. As we only have a single Y output
value, we simply want the first column of Y. This is then repeated for the rest
of the provided X values.

### The forest_predict method

The forest_predict method looks a lot more intimidating, but is just as
straightforward as the predict method. Here is the code:

```cython
def predict_quantile(
    tree: DecisionTree, X: np.ndarray, n_obs: int
) -> list:
    # Check if quantile is an array
    indices = []

    for i in range(n_obs):
        cur_node = tree.root
        while isinstance(cur_node, DecisionNode):
            cur_split_idx = cur_node.split_idx
            cur_threshold = cur_node.threshold
            if X[i, cur_split_idx] < cur_threshold:
                cur_node = cur_node.left_child
            else:
                cur_node = cur_node.right_child

        indices.append(cur_node.indices)
    return indices

cdef class PredictorQuantile(Predictor):
  @staticmethod
  def forest_predict(cnp.ndarray X_old, cnp.ndarray Y_old, cnp.ndarray X_new,
                      trees: list[DecisionTree], parallel: ParallelModel,
                      **kwargs) -> np.ndarray:
      cdef:
          int i, j, n_obs, n_trees
          list prediction_indices, pred_indices_combined, indices_combined
      if "quantile" not in kwargs.keys():
          raise ValueError(
              "quantile called without quantile passed as argument"
          )
      quantile = kwargs['quantile']
      n_obs = X_new.shape[0]
      prediction_indices = parallel.async_map(predict_quantile,
                                              map_input=trees, X=X_new,
                                              n_obs=n_obs)
      # In case the leaf nodes have multiple elements and not just one, we
      # have to combine them together
      n_trees = len(prediction_indices)
      pred_indices_combined = []
      for i in range(n_obs):
          indices_combined = []
          for j in range(n_trees):
              indices_combined.extend(prediction_indices[j][i])
          pred_indices_combined.append(indices_combined)
      ret = np.quantile(Y_old[pred_indices_combined], quantile)
      return np.array(ret, dtype=DOUBLE)
```

The forest_predict method is a staticmethod, meaning that it is tied to the
Predictor class itself and not a specific instance of the class. The reason for
this is that it allows us to fully control the parallization over trees. For
the PredictorQuantile, for example, we want to be able to control this ourselves.

As before we define the variables used and check the input for the kwarg
`quantile`. However, this time we needed to define a globally available function
`predict_quantile` at the top level of the file. It has to be a globally available
for the multiprocessing to work probably. This function traverses a given tree,
and finds the LeafNode each element of X would end up in and adds the indices
of the elements already in the LeafNode. We then call `predict_quantile`
using the parallel.async_map, which is adaXTs way of making
parallelization more manageable. It makes use of the
[Parallel](../api_docs/Parallel.md) class. The async_map calls
`predict_quantile` with all the trees in parallel, and returns the result. This
means, that `prediction_indices` will contain a list with the length equal
to the number of trees in the forest. Each element of the list will be a single
trees prediction for the input array X. We then create a list
`pred_indices_combined` where we combine all the predictions for X.
To get the final result, we then just call numpy's quantile implementation.
