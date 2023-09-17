import pstats, cProfile
import time
import numpy as np
from adaXT.decision_tree._criteria import gini_index_wrapped
from adaXT.decision_tree._tree import Tree
from sklearn.tree import DecisionTreeClassifier as DTC

X = np.random.randint(0, 100, (1000, 6))
Y = np.random.randint(0, 40, 1000)

sk_tree = DTC(criterion="gini")
tree = Tree("Classification")
gini = gini_index_wrapped()
profiler = cProfile.Profile()
profiler.enable()
tree.fit(X, Y, gini)
profiler.disable()
profiler.print_stats(sort="tottime")
st = time.time()
tree.fit(X,Y, gini)
et = time.time()
print("Actual runtime: ", et-st)
st = time.time()
sk_tree.fit(X, Y)
et = time.time()
print("sklearn runtime: ", et-st)
print("datapoints: ", len(Y))