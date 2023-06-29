import pandas as pd
import sys
from sklearn import tree
import matplotlib.pyplot as plt

from decision_tree.splitter import Splitter

data = pd.read_csv('./tests/data_banknote_authentication.csv')
print(data.head(657).describe())
data = data.to_numpy()
splitter = Splitter(data, 1000, 1000)

split, best_threshold, best_index, best_score = splitter.get_split(0, len(data))
clf = tree.DecisionTreeClassifier(max_depth=2)
clf.fit(data[:, :-1], data[:, -1])

print(split)
split, best_threshold, best_index, best_score = splitter.get_split(0, split)

print(split)
split, best_threshold, best_index, best_score = splitter.get_split(split, len(data))

tree.plot_tree(clf)
plt.show()