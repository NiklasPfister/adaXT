import pstats, cProfile
import time
import numpy as np
from adaXT.decision_tree._criteria import gini_index_wrapped, variance_wrapped
from adaXT.decision_tree._tree import Tree
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import DecisionTreeRegressor as DTR

n = 1000
X = np.random.uniform(0, 250, (n, 6))
Y_cla = np.random.randint(0, 10, n)
Y_reg = np.random.uniform(0, 1000, n)
sk_tree = DTC(criterion="gini")
tree = Tree("Classification")
gini = gini_index_wrapped()
# profiler = cProfile.Profile()
# profiler.enable()
# tree.fit(X, Y_cla, gini)
# profiler.disable()
# profiler.print_stats(sort="tottime")
print("Classification")
st = time.time()
tree.fit(X,Y_cla, gini)
et = time.time()
print("runtime: ", et-st)
st = time.time()
sk_tree.fit(X, Y_cla)
et = time.time()
print("sklearn runtime: ", et-st)

print("Regression")
tree_reg = Tree('Regression')
var = variance_wrapped()
st = time.time()
tree_reg.fit(X, Y_reg, var)
et = time.time()
print("runtime: ", et-st)
sk_reg = DTR()
st = time.time()
sk_reg.fit(X, Y_reg)
et = time.time()
print("runtime: ", et-st)


# print("datapoints: ", len(Y))