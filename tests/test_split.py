import pandas as pd
import sys
from sklearn import tree
import matplotlib.pyplot as plt

# Needed on my installation to run, comment out for running on Simon
sys.path.append("decision_tree")
from splitter import Splitter

# Uncomment on Simon
#from decision_tree.splitter import Splitter


data = pd.read_csv('./tests/diabetes_prediction_dataset_short.csv')  


#print(data.head(657).describe())
data = data.to_numpy()
#print(data[data[:, 0] <= 0.32])
splitter = Splitter(data)

split, best_threshold, best_index, best_score = splitter.get_split(range(len(data)))
clf = tree.DecisionTreeClassifier(max_depth=2)
clf.fit(data[:, :-1], data[:, -1])

print(len(split[0]), len(split[1]))
split, best_threshold, best_index, best_score = splitter.get_split(split[0])

print(len(split[0]), len(split[1]))
split, best_threshold, best_index, best_score = splitter.get_split(split[1])

tree.plot_tree(clf)
plt.show()