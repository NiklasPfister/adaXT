import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import decision_tree
from importlib import reload
reload(decision_tree)

data = pd.read_csv(r'C:\Users\Simon\Programming\adaXT\decision_tree\data\data_banknote_authentication.csv')

data = data.to_numpy()
builder = decision_tree.tree.DepthTreeBuilder(data, 2)
our_tree = decision_tree.tree.Tree()
our_tree =builder.build_tree(our_tree)

clf = tree.DecisionTreeClassifier(max_depth=2)
clf.fit(data[:, :-1], data[:, -1])

our_tree.print_tree()
tree.plot_tree(clf)
plt.show()