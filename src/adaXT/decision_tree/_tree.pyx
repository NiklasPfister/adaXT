from ._splitter cimport Splitter
import numpy as np

def testing():
    X = np.array([[1, -1],
                [-0.5, -2],
                [-1, -1],
                [-0.5, -0.5],
                [1, 0],
                [-1, 1],
                [1, 1],
                [-0.5, 2]])
    Y_cla = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    split = Splitter.gen_cla_splitter(X, Y_cla)
    sorted = split.sort_feature(list(range(X.shape[0])), X[0])
    return sorted