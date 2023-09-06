import pstats, cProfile
import numpy as np
from adaXT.decision_tree._criteria import gini_index_wrapped
from adaXT.decision_tree._tree import Tree
X = np.array([[1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2]])
Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])
tree = Tree("Classification")
cProfile.runctx("tree.fit(X, Y_cla, gini_index_wrapped)", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()