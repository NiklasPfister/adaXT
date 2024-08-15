import numpy as np
import matplotlib.pyplot as plt
from adaXT.decision_tree import DecisionTree
from adaXT.decision_tree.tree_utils import plot_tree

import pyximport
pyximport.install()
import testCrit

# Generate training data
n = 100
m = 4
X = np.random.uniform(0, 100, (n, m))
Y = np.random.uniform(0, 10, n)

# Initialize and fit tree
tree = DecisionTree("Regression",
                    criteria=testCrit.Partial_linear,
                    max_depth=3)
tree.fit(X, Y)

# Plot the tree
plot_tree(tree)
plt.show()
