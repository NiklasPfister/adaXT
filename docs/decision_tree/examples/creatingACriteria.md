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

## A detailed example