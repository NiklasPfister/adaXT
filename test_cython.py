from adaXT.decision_tree.criteria import gini_index
import numpy as np
X = np.array([[1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2]])
Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])


print(gini_index(X, Y_cla))
