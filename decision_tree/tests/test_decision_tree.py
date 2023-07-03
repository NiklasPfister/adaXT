import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import decision_tree
from decision_tree.criteria import gini_index
from decision_tree.tree import DepthTreeBuilder, Tree
from importlib import reload
reload(decision_tree)

data = pd.read_csv(r'C:\Users\Simon\Programming\adaXT\decision_tree\data\data_banknote_authentication.csv')

data = data.to_numpy()
X = data[:, :-1]
Y = data[:, -1]
builder = DepthTreeBuilder(X, Y, 3, gini_index)
our_tree = Tree()
our_tree = builder.build_tree(our_tree)

clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(data[:, :-1], data[:, -1])

our_tree.print_tree()
tree.plot_tree(clf, fontsize=10)
plt.show()