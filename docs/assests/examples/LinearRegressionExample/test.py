import numpy as np
import testCrit
from adaXT.decision_tree import DecisionTree
from adaXT.decision_tree.tree_utils import plot_tree
import matplotlib.pyplot as plt
import pyximport
pyximport.install()

n = 100
m = 4


X = np.random.uniform(0, 100, (n, m))
Y = np.random.uniform(0, 10, n)
tree = DecisionTree("Regression", testCrit.Linear, max_depth=3)
tree.fit(X, Y)

plot_tree(tree)
plt.show()
