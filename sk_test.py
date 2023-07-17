from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
import numpy as np
X = np.array([[1, -1],
              [-0.5, -2],
              [-1, -1],
              [-0.5, -0.5],
              [1, 0],
              [-1, 1],
              [1, 1],
              [-0.5, 2]])

Y_reg = np.array([2.2, -0.5, 0.5, -0.5, 2, -3, 2.2, -3])

tr_cla = DecisionTreeRegressor()

tr_cla.fit(X, Y_reg)

plot_tree(tr_cla)
plt.show()