import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from decision_tree.criteria import *
from decision_tree.splitter import Splitter


data = pd.read_csv(r'C:\Users\Simon\Programming\adaXT\decision_tree\data\data_banknote_authentication.csv')
#print(data.head(657).describe())
data = data.to_numpy()
print(data[data[:, 0] <= 0.32])
splitter = Splitter(data, gini_index)

all_idx = [*range(len(data))]
split, best_threshold, best_index, best_score, best_imp = splitter.get_split(all_idx)
clf = tree.DecisionTreeClassifier(max_depth=2)
clf.fit(data[:, :-1], data[:, -1])

print(len(split[0]), len(split[1]))
split, best_threshold, best_index, best_score = splitter.get_split(split[0])

print(len(split[0]), len(split[1]))
split, best_threshold, best_index, best_score = splitter.get_split(split[1])

tree.plot_tree(clf)
plt.show()