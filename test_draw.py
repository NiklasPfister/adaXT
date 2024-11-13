from adaXT.decision_tree import DecisionTree, plot_tree
import numpy as np
import matplotlib.pyplot as plt

N = 1000
M = 5
X = np.random.uniform(0, 100, (N, M))
Y = np.random.randint(0, 4, N)
tree = DecisionTree("Classification", max_depth=5)
tree.fit(X, Y)
plot_tree(tree)
plt.show()
