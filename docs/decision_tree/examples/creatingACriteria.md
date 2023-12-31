# How to create your own Criteria Function

### Installing cython and adaXT
First make sure you are setup in a virtual environment where cython is installed. You could for example make use of [virtualenv](https://virtualenv.pypa.io/en/latest/). Then run ```pip install adaXT``` and ```pip install cython```.

### Creating a pyx file
Then in your working folder create a .pyx file. First you have to import the Criteria class and create your new Criteria class, that is going to inherit from the "super" Criteria class. As an example here is the definition of the Linear_regression.
```cython
from adaXT.decision_tree.criteria cimport Criteria

cdef class Linear_regression(Criteria):
```
The class type of Criteria is a cdef class, which works in much of the same fashion as your regular Python class, however it gives a faster performance, as we can create cdef methods within. To read more on cdef classes checkout [cython Extension types](https://cython.readthedocs.io/en/latest/src/tutorial/cdef_classes.html).

Now for the criteria to work with the library you have to create the impurity function. The impurity method has to follow the type specified within [criteria.pxd](https://github.com/NiklasPfister/adaXT/blob/main/src/adaXT/decision_tree/criteria.pxd). Which is the following:
```cython
    cpdef double impurity(self, int[:] indices):
```
Here the indices stand for the indices of the values for which you have to calculate the impurity. To access the specific feature values you can make use of the self.x, which stores all the feature values for the entire tree. Such that ```cython self.x[indices] ``` and ```cython self.y[indices]``` would be the specific feature values and sample values for the specific node you have to calculate impurity for. From here you should now be able to build the cython file, by following [building cython code](https://cython.readthedocs.io/en/latest/src/quickstart/build.html). Or you can follow the next section.

### Using the Criteria
Within the same folder as you create the **.pxd**, you can now create your **.py** file. But you have to do a little more work. If you wish to not manually recompile the cython file every time you make changes to it, you can within the python file tell cython to compile the cython source file whenver you run the file:
```python
from adaXT.decision_tree import DecisionTree
import pyximport; pyximport.install()
import <your .pyx filename>
```
Now you can access the new criteria within the python file like you would any other criteria function and such as:

```python
import <your .pyx filename>
import numpy as np

n = 100
m = 4

X = np.random.uniform(0, 100, (n, m))
Y = np.random.uniform(0, 10, n)
tree = DecisionTree("Regression", <your .pyx filename>.<your cdef class>, max_depth=3)
tree.fit(X, Y)
```

## A detailed example: Linear Regression
The general idea of the Linear Regression criteria is to fit a linear function on the first feature with the y value as the outcome, such that:
$$
L = \sum_{i \in I} (Y[i] - \theta_0 - \theta_1 X[i, 0])^2 \\
$$
$$
\theta_1 = \frac{\sum_{i \in I} (X[i, 0] - \mu_X) * (Y[i] - \mu_Y)}{\sum_{i \in I} (X[i, 0] - \mu_X)^2 } \\
$$
$$
\theta_0 = \mu_Y - \theta_1 \mu_X
$$
Where $I$ denotes the indices of the samples within a given node, X is the feature values, Y is the outcomes values and $\mu$ denotes the mean.


When creating a new criteria function we import and define the class as described above, such that we have:
```python
from adaXT.decision_tree.criteria cimport Criteria

cdef class Linear_regression(Criteria):
```

### Calculating the mean

Now although we have provided a mean function within the adaXT.decision_tree.crit_helpers, in this specific example we need to calculate multiple means, and there is no reason to do 2 passes over the same indices I, so we create a custom mean method:

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
You might notice that the syntax is a little different than the default python. Much of it could be left like default python, and it is mainly just for speeding up the runtime, such that we don't have to use the python interpreter while running the code. However, if you wish to learn more about the cython language be sure to check out the [cython documentation](https://cython.readthedocs.io/en/latest/).

Another important note is, we are freely able to create any new methods on our criteria function, even if they are not defined as the standard within the parent Criteria class. You are not limited by the parent, and as such can extend it with however many methods you want, as long as you don't overwrite the evaluate split (unless that is your intention).

### Calculating theta
Now that we have a method for getting the mean, then we might aswell create a method for calculating the theta values.
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
Again the majority of the cython is not needed and are mainly just for speedup.
on line 26 we access our previously defined custom mean function, which returns the mean of the X indices and the mean of the Y indices as described previously. Then at line 27 we loop over all the indices a second time where we calculate the $\sum_{i \in I} (X[i, 0] - \mu_X) * (Y[i] - \mu_Y)$ and $\sum_{i \in I} (X[i, 0] - \mu_X)^2$ which is the numerator and denominator respectively. These are the two values used to calculate $\theta_1$. Now note that on line 31 we have a check to make sure that the denominator is not 0.0. This would be the case if all the X values of the first feature in a node is the same. If that is the case, we simply set $theta_1$ to 0.0 as this will give an L value of 0 in the end. We finish off by returning the two $\theta$ values.

[comment]: # (TODO: check if the description made is correct with Niklas.)

### The impurity function
Atlast we reach the crux of the problem. Creating the impurity function.
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
Making sure it follows the same signature as described previously we simply calculate L as described and return the value. One important thing to note is, that cur_sum is defined on line 3 to have the type double. That is because the impurity function is defined to have a double as the return value. As such when creating the impurity function you must uphold this.

### Finishing up
And that is it. You have now created your first Criteria function, and can freely use it within your own python code. Using the described method previously of compiling the cython file every time we run the python file, then we could end up with a file that looks like this:
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
tree = DecisionTree("Regression", testCrit.Linear, max_depth=3)
tree.fit(X, Y)

plot_tree(tree)
plt.show()
```
This just creates a regression tree with our new Linear Criteria function. Specifies the max_depth to be 3 and then plots the tree using both our [plot_tree](../utils/utils.md) and the [matplotlib](https://matplotlib.org/). To see the full source code used within this article check it out [here](https://github.com/NiklasPfister/adaXT/tree/Documentation/docs/assests/examples/LinearRegressionExample).