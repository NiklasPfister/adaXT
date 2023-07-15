import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("decision_tree")
from adaXT.decision_tree.criteria import variance
from adaXT.decision_tree.tree import DepthTreeBuilder, Tree, Node
from adaXT.decision_tree.tree_utils import plot_tree, print_tree
from importlib import reload
from sklearn import tree


    
data = pd.read_csv(r'C:\Users\wheus\OneDrive\adaXT\tests\real_estate.csv')

data = data.to_numpy()
X = data[:, :-1]
Y = data[:, -1]
builder = DepthTreeBuilder(X, Y, variance)
our_tree = Tree("Regression", max_depth=3)
our_tree = builder.build_tree(our_tree)

clf = tree.DecisionTreeRegressor(max_depth=3)
clf.fit(X, Y)

#print_tree(our_tree)
plot_tree(our_tree)
plt.figure()
tree.plot_tree(clf, fontsize=10)
plt.show()
x = 10