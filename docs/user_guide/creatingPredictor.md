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
    # predict for it's estimators.
  

```
The template has three components: (1) The \_\_init\_\_ function, which is used to
initialize the class. Because Cython removes a lot of the boiler plate with
default Python classes, we cannot just add attributes to our cdef class without
defining the attributes specifically on the class via the \_\_init\_\_ function.
You can overwrite the \_\_init\_\_ function if desired to initialize variables on 
the Predictor class, which can then be used in the predict method.
If you do not use the \_\_init\_\_ functionality, you generally do not need to
add any extra attributes. (2) The predict method, wich is used to compute the
predictions at the provided X values. It is a simple def function 
and can be used like any other regular Python method. Within this function you 
in particular have access to the general attributes found on the
[Predictor](../api_docs/Predictor.md) class. This includes the number of
features and the root node object, which you can use to traverse the tree (see 
example below).
(3) The forest_predict method, which is used to aggregate predictions across trees
for forest predictions. It is a static method, which allows
us to parallelize across trees. If your custom Predictor just averages the
tree predictions, you can just inherit the forest_predict method from the base
Predictor class.

## Example of creating a Predictor

### The predict method

As an example we have the PredictorQuantile, which is able to predict the
quantiles of regression data instead of just the mean squared error as the
regular regression predict. First, let us just focus on the predict method:

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

The PredictorQuantile needs no initialization beyond what is done by the regular
Predictor, and so it does not implement any special attributes or the
\_\_init\_\_ function. Next, inside the predict method, we define the types of
the variables used. This allows cython for greater optimisation, which leads to
a faster prediction time. Then, we check the kwargs for the key "quantile". Any
keyword arguments passed to the DecisionTree.predict is passed directly to the
Predictor.predict, meaning that we can access the desired quantile from the
predict signature without having to changed anything else. As we allow for
multiple quantiles, we have to setup the prediction variable depending on if the
quantile is a Squence or just a single element. Then we can proceed by going
through the tree.

For every observation, we go to the root node and loop as long as we are in a
DecisionNode. We can check if we are split to the left or the right, and
traverse down the tree.

Once the cur_node is no longer an instance of the DecisionNode, then we have
reached a LeafNode. We can access all Y values via self.Y(.base has to be added,
as we are indexing with a list of elements) and the indices of the elements
within the LeafNode via cur_node.indices. As we only have a single Y output
value, we simply want the first column of Y. This is then repeated for the rest
of the given X values.

### The forest_predict method

The forest_predict method looks a lot more intimidating, but is just as
straight forward as working with the predict method. Here is the overview:

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
general Predictor class and not a specific instance of the class. The reason for
this is, that we in the predictor can control specifically have the
parallelisation happens, when we are predicting for the tree. For the quantile
for example, we want to be able to control this ourselves.

As before we define the variables used and check the input for quantile.
However, this time we have defined a function at the top level of the file,
which is not some method and is globally available. This has to be a globally
available function for the multiprocessing to work probably. This function
simply traverse a given tree, and finds the LeafNode each element of X would end
up in and adds the indices of elements already in the LeafNode. This
predict_quantile function is called using the parallel.async_map, which is
adaXTs way of making parallelisation more manageable. It makes use of the
[Parallel](../api_docs/Parallel.md) class. The async_map calls the
predict_quantile with all the trees in parallel, and returns the result. This
means, that prediction_indices will contain a list with the length of the number
of estimators of the random forest. Each element of the list will be a single
trees prediction for the input array X. As we want a list, where we have
combined all the predictions for X, we create pred_indices_combined for the
purpose. This just leaves us with calling numpy's quantile implementation and
returning the result!
