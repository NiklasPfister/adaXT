import pandas as pd
import matplotlib.pyplot as plt
import adaXT
import numpy as np
from adaXT.decision_tree.criteria import gini_index
from adaXT.decision_tree.tree import DepthTreeBuilder, Tree
from importlib import reload
from adaXT.decision_tree.tree_utils import plot_tree
from sklearn import tree
reload(adaXT)
X = np.array([[1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2]])
Y= np.array([1, -1, 1, -1, 1, -1, 1, -1])
our_tree = Tree("Classification")
our_tree.fit(X, Y, gini_index)
plot_tree(our_tree)
plt.show()

weight_matrix = our_tree.weight_matrix()
print(weight_matrix)