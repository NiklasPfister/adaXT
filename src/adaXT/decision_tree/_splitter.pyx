import numpy as np
cimport numpy as cnp
from ._func_wrapper import FuncWrapper
cnp.import_array()


cdef class Splitter:
    """
    Splitter class used to create splits of the data
    """
    def __init__(self, cnp.ndarray[npFloat] X, cnp.ndarray[npFloat, ndim = 1] Y, cnp.ndarray[npInt, ndim=2] presort, criterion: FuncWrapper):
        self.features = X 
        self.outcomes = Y 

        self.n_features = X.shape[0]
        self.criteria = criterion
        self.pre_sort = presort
        # self.constant_features = np.empty(len(self.features)) #TODO: not yet implemented

    cdef cnp.ndarray sort_feature(self, cnp.ndarray[npInt, ndim=1] indices, cnp.ndarray[npFloat, ndim=1] feature):
        cdef npFloat[:] sort_list
        sort_list = feature[indices]
        sort_list = np.argsort(sort_list)

        return indices[sort_list]

    cdef int test_split(self, test_obj test, int[:] left_indices, int[:] right_indices, int feature):
        cdef double[2] imp
        cdef int[:] indices 
        cdef double[:] curr
        cdef double crit, mean_thresh
        cdef int n_total, n_curr
        cdef double[:] x, y

        features = np.array(self.features)
        outcomes = np.array(self.outcomes)
        func_wrap = self.criteria
        if func_wrap.func:
            criteria = func_wrap.func
        else:
            raise MemoryError
        
        indices = self.indices
        idx_split = [left_indices, right_indices]

        n_total = len(indices)
        for i in range(2): # Left and right side
            curr = idx_split[i]
            n_curr = len(curr)
            if n_curr == 0:
                continue
            x = features[curr]
            y = outcomes[curr]
            imp[i] = criteria(x, y) # Calculate the impurity of current child
            crit += imp[i] * (n_curr / n_total) # Weight by the amount of datapoints in child

        mean_thresh = np.mean([features[left_indices[-1], feature], features[right_indices[0], feature]])
        
        test.crit = crit
        test.idx_split = idx_split
        test.imp = imp
        test.split_val = mean_thresh
        return 1