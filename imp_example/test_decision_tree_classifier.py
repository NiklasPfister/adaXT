import pandas as pd
import matplotlib.pyplot as plt
import adaXT
from adaXT.decision_tree.criteria import gini_index
from adaXT.decision_tree.tree import DepthTreeBuilder, Tree
from importlib import reload
from sklearn import tree
reload(adaXT)

data = pd.read_csv(r'C:\Users\Simon\Programming\adaXT\decision_tree\data\data_banknote_authentication.csv')

data = data.to_numpy()
X = data[:, :-1]
Y = data[:, -1]
builder = DepthTreeBuilder(X, Y, gini_index)
our_tree = Tree("Classification")
our_tree = builder.build_tree(our_tree)

clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, Y)

tree.plot_tree(clf, fontsize=10)
plt.show()