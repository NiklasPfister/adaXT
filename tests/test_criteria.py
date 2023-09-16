from adaXT.decision_tree.criteria import gini_index
from adaXT.decision_tree._criteria import gini_index_wrapped, call_criteria_wrapped
import numpy as np
def test_simple():
    c_gini = gini_index_wrapped()
    gini = gini_index
    X = np.array([[1, -1],
                [-0.5, -2],
                [-1, -1],
                [-0.5, -0.5],
                [1, 0],
                [-1, 1],
                [1, 1],
                [-0.5, 2]]).astype(np.double)
    Y = np.array([1, -1, 1, -1, 1, -1, 1, -1]).astype(np.double)
    c_res = call_criteria_wrapped(c_gini, X, Y)
    res = gini(X, Y)
    print(f"C_gini: {c_res}, gini: {res}")
    assert(c_res == res)

test_simple()