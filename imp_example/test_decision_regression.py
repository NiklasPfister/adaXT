import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import adaXT.decision_tree
from adaXT.decision_tree.criteria import variance
from adaXT.decision_tree.tree import DepthTreeBuilder, Tree, Node
from importlib import reload
from sklearn import tree

import pytest

    
data = pd.read_csv(r'C:\Users\Simon\Programming\adaXT\decision_tree\data\data_banknote_authentication.csv')

data = data.to_numpy()
X = data[:, :-1]
Y = data[:, -1]
builder = DepthTreeBuilder(X, Y, variance)
our_tree = Tree("Regression", max_depth=3)
our_tree = builder.build_tree(our_tree)

clf = tree.DecisionTreeRegressor()
clf.fit(X, Y)

tree.plot_tree(clf, fontsize=10)
plt.show()