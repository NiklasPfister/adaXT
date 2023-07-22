import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from adaXT.decision_tree.criteria import *
from adaXT.decision_tree.splitter_new import Splitter_new
from adaXT.decision_tree.splitter import Splitter
import time


data = pd.read_csv(r'C:\Users\wheus\OneDrive\adaXT\tests\diabetes_prediction_dataset_short.csv')
#print(data.head(657).describe())
data = data.to_numpy()
X = data[:, :-1]
Y = data[:, -1]

start_time_new = time.perf_counter()

splitter = Splitter_new(X, Y, gini_index)

all_idx = [*range(len(Y))]
split, best_threshold, best_feature, best_score, best_imp = splitter.get_split(all_idx)
print(len(split[0]), len(split[1]))
print("Threshold", best_threshold)
print("best_feature", best_feature)
print("best_score", best_score)
print("best_imp", best_imp, "\n\n")

split, best_threshold, best_feature, best_score, best_imp  = splitter.get_split(split[0])
print(len(split[0]), len(split[1]))
print("Threshold", best_threshold)
print("best_feature", best_feature)
print("best_score", best_score)
print("best_imp", best_imp, "\n\n")

end_time_new = time.perf_counter()


print("New splitter \n\n\n\n")
start_time_old = time.perf_counter()
splitter = Splitter(X, Y, gini_index)

all_idx = [*range(len(Y))]
split, best_threshold, best_feature, best_score, best_imp = splitter.get_split(all_idx)
print(len(split[0]), len(split[1]))
print("Threshold", best_threshold)
print("best_feature", best_feature)
print("best_score", best_score)
print("best_imp", best_imp, "\n\n")

split, best_threshold, best_feature, best_score, best_imp  = splitter.get_split(split[0])
print(len(split[0]), len(split[1]))
print("Threshold", best_threshold)
print("best_feature", best_feature)
print("best_score", best_score)
print("best_imp", best_imp, "\n\n")

end_time_old = time.perf_counter()

start_time_sklearn = time.perf_counter()

clf = tree.DecisionTreeClassifier(max_depth=2)
clf.fit(data[:, :-1], data[:, -1])
end_time_sklearn = time.perf_counter()


tree.plot_tree(clf)
plt.show()

print("New time", (end_time_new - start_time_new) * 1000)
print("Old time", (end_time_old - start_time_old) * 1000)
print("sklearn time", (end_time_sklearn - start_time_sklearn) * 1000)