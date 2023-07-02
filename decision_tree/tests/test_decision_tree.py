import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from decision_tree import DepthTreeBuilder, Tree
from decision_tree import gini_index

data = pd.read_csv(r'C:\Users\Simon\Programming\adaXT\decision_tree\tests\data_banknote_authentication.csv')

data = data.to_numpy()
builder = DepthTreeBuilder(data, 2, gini_index)
our_tree = Tree()
our_tree =builder.build_tree(our_tree)

clf = tree.DecisionTreeClassifier(max_depth=2)
clf.fit(data[:, :-1], data[:, -1])

our_tree.print_tree()
tree.plot_tree(clf)
plt.show()