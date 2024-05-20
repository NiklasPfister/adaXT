# Creating a custom criteriadecision_tree

In the following we explain how to create a custom criteria function
by walking through the required steps. We start by creating .pyx file.

## Creating a .pyx file

In your working directory create a .pyx file in which you first import
the Criteria "super" class and then define the new custom Criteria
class which inherits from the imported Criteria class. An example of
the skeleton of the the Linear_regression Criteria file is given
below.

```cython
from adaXT.criteria cimport Criteria

cdef class Linear_regression(Criteria):
    # Your implementation here
```

The class type of Criteria is a cdef class, which works in much of the
same fashion as a regular Python class, but with faster
performance. To read more on cdef classes checkout [Cython extension
types](https://cython.readthedocs.io/en/latest/src/tutorial/cdef_classes.html).

Now for the criteria to work with adaXT you have to implement the
impurity method inherited from the Criteria class. The impurity method
has to follow the type specified within
[criteria.pxd](https://github.com/NiklasPfister/adaXT/blob/main/src/adaXT/criteria/criteria.pxd),
which is as follows.

```cython
    cpdef double impurity(self, int[:] indices):
```

Here, the indices refer to the sample indices for which the impurity
value should be computed. To access the feature and response you can
make use of ```self.x``` and ```self.y```, respectively. More
specifically, ```self.x[indices] ``` and ```self.y[indices]``` are the
feature and response samples for which the impurity needs to be
computed. With this in place you should be able to build the your
custom Cython file. This can be done either by following [building
cython
code](https://cython.readthedocs.io/en/latest/src/quickstart/build.html)
or by following the steps below.

Now that you have implemented your custom criteria function, we
explain how to use it in a simple setting.

## Using a custom criteria

Basically you can now create your **.py** file in which you fit a
decision tree using the new custom criteria within the same folder
where you created the **.pyx**. For this to work there are a few small
things left to do. In order to not manually recompile the Cython file
every time you make changes, you can automate the compilation within
the Python file by telling Cython to comile the Cython source file,
whenever you run the file:

```python
from adaXT.decision_tree import DecisionTree
import pyximport; pyximport.install()
import <your .pyx filename>
```

Now you can access the new criteria within the Python file like you
would do with any other criteria function:

```python
import <your .pyx filename>
import numpy as np

n = 100
m = 4

X = np.random.uniform(0, 100, (n, m))
Y = np.random.uniform(0, 10, n)
tree = DecisionTree("Regression", criteria=<your .pyx filename>.<your cdef class>, max_depth=3)
tree.fit(X, Y)
```

We now go through a detailed example in which we construct the linear
regression criteria.

## A detailed example: Linear regression

The general idea of the linear regression criteria is to fit a linear
function on the first feature with the y value as the response, that
is,
$$
L = \sum_{i \in I} (Y[i] - \theta_0 - \theta_1 X[i, 0])^2 \\
$$
$$
\theta_1 = \frac{\sum_{i \in I} (X[i, 0] - \mu_X)(Y[i] - \mu_Y)}{\sum_{i \in I} (X[i, 0] - \mu_X)^2} \\
$$
$$
\theta_0 = \mu_Y - \theta_1 \mu_X,
$$

where $I$ denotes the indices of the samples within a given node, $X$
is the feature matrix, $Y$ is the response vector, $\mu_X$ is mean
vector of the columns of $X$ and $\mu_Y$ is the mean of $Y$.


When creating a new criteria function we import and define the class
as described above. This leads to

```python
from adaXT.criteria cimport Criteria

cdef class Linear_regression(Criteria):
```

### Calculating the mean

Now although we have provided a mean function within the
adaXT.decision_tree.crit_helpers, in this specific example we need to
calculate multiple means, and there is no reason to do two passes over
the same indices $I$, so we create a custom mean method in Cython.

```cython
# Custom mean function, such that we don't have to loop through twice.
cdef (double, double) custom_mean(self, int[:] indices):
    cdef:
        double sumX, sumY
        int i
        int length = indices.shape[0]
    sumX = 0.0
    sumY = 0.0
    for i in range(length):
        sumX += self.x[indices[i], 0]
        sumY += self.y[indices[i]]

    return ((sumX / (<double> length)), (sumY / (<double> length)))
```

You might notice that the syntax is a little different than for
default Python. In general you can write default Python within the
cdef class, however the runtime can be significantly decreased by
adding type definitions and other Cython magic. If you wish to learn
more about the Cython language be sure to check out the [Cython
documentation](https://cython.readthedocs.io/en/latest/).

Furthermore, note that you can freely create any new method within the
custom Criteria class criteria even if it is not defined in the
standard parent Criteria class. Therefore you are not limited by the
parent class and can freely extend it as long as you do not overwrite
the `evaluate_split` (unless that is your intention).

### Calculating theta

Now that we have a method for getting the mean, we can create a method
for calculating the theta values, such that our code is more
manageable.

```cython linenums="1"
cdef (double, double) theta(self, int[:] indices):
    """
    Calculate theta0 and theta1 used for a Linear Regression
    on X[:, 0] and Y
    ----------

    Parameters
    ----------
    indices : memoryview of NDArray
        The indices to calculate

    Returns
    -------
    (double, double)
        where the first element is theta0 and second element is theta1
    """
    cdef:
        double muX, muY, theta0, theta1
        int length, i
        double numerator, denominator
        double X_diff

    length = indices.shape[0]
    denominator = 0.0
    numerator = 0.0
    muX, muY = self.custom_mean(indices)
    for i in range(length):
        X_diff = self.x[indices[i], 0] - muX
        numerator += (X_diff)*(self.y[indices[i]]-muY)
        denominator += (X_diff)*X_diff
    if denominator == 0.0:
        theta1 = 0.0
    else:
        theta1 = numerator / denominator
    theta0 = muY - theta1*muX
    return (theta0, theta1)
```

Again the majority of the Cython is not mandatory but speeds up the
code.  On line 26 we access our previously defined custom mean
function, which returns the mean of the $X$ indices and the mean of
the $Y$ indices as described above. Then at line 27 we loop over all
the indices a second time and calculate $\sum_{i \in I} (X[i, 0] -
\mu_X) (Y[i] - \mu_Y)$ and $\sum_{i \in I} (X[i, 0] - \mu_X)^2$ which
are the numerator and denominator, respectively. These are the two
values used to calculate $\theta_1$. Further on line 31 we check to
make sure that the denominator is not 0.0, this ensures that we can
consider the underidentified case separately and do not divide by zero
by accident. If the denominator is zero, we simply set $\theta_1$ to
0.0 as this will give an L value of 0 in the end. We finish off by
returning the two values $\theta_0,\theta_1$.


### The impurity function

Finally, we reach the most important step, which is to create the
impurity function.

```cython linenums="1"
cpdef double impurity(self, int[:] indices):
    cdef:
        double step_calc, theta0, theta1, cur_sum
        int i, length

    length = indices.shape[0]
    theta0, theta1 = self.theta(indices)
    cur_sum = 0.0
    for i in range(length):
        step_calc = self.y[indices[i]] - theta0 - theta1 * self.x[indices[i], 0]
        cur_sum += step_calc*step_calc
    return cur_sum
```

Making sure the impurity method has the required signature, we simply
calculate L as described and return its value. One important thing to
note is that `cur_sum` is defined on line 3 to have type double,
because the impurity function is defined to have a double as return
value. When creating the impurity method you must ensure this return
type.

### Finishing up

And that is it. You have now created your first Criteria function, and
can freely use it within your own Python code. If you use the method
described above to compile the Cython file every time the Python file
is run leads to code that looks like this:

```python
from adaXT.decision_tree import DecisionTree
from adaXT.decision_tree.tree_utils import plot_tree
import matplotlib.pyplot as plt
import pyximport; pyximport.install()
import testCrit
import numpy as np

n = 100
m = 4


X = np.random.uniform(0, 100, (n, m))
Y = np.random.uniform(0, 10, n)
tree = DecisionTree("Regression", testCrit.Linear_regression, max_depth=3)
tree.fit(X, Y)

plot_tree(tree)
plt.show()
```

This just creates a regression tree with the newly created custom
Linear_regression Criteria function, specifies the `max_depth` to be 3
and then plots the tree using both our
[plot_tree](../api_docs/tree_utils.md#adaXT.decision_tree.tree_utils.plot_tree) and the
[matplotlib](https://matplotlib.org/). The full source code used
within this article can be found
[here](https://github.com/NiklasPfister/adaXT/tree/Documentation/docs/assets/examples/linear_regression).
