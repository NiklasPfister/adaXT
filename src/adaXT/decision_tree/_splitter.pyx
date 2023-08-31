import numpy as np
cimport numpy as cnp
from ._func_wrapper import FuncWrapper
cnp.import_array()


ctypedef cnp.float64_t npFloat
ctypedef cnp.int_t npInt
cdef class Splitter:
    """
    Splitter class used to create splits of the data
    """
    def __init__(self, cnp.ndarray[npFloat, ndim=2] X, cnp.ndarray[npFloat, ndim=1] Y, cnp.ndarray[npInt, ndim=2] presort, criterion: FuncWrapper):
        """
        Parameters
        ----------
        X : npt.NDArray
            The input features of the dataset

        Y : npt.NDArray
            The outcomes of the dataset

        criterion : Callable, optional
            Criteria function for calculating information gain,
            if None it uses the specified function in the start of splitter.py
        
        presort : ndarray 
            A sorted index list of the features 
        """
        self.features = X 
        self.outcomes = Y 

        self.n_features = X.shape[0]
        self.criteria = criterion
        self.pre_sort = presort
        # self.constant_features = np.empty(len(self.features)) #TODO: not yet implemented

    cdef cnp.ndarray sort_feature(self, cnp.ndarray[npInt, ndim=1] indices, cnp.ndarray[npFloat, ndim=1] feature):
        """
        Parameters
        ----------
        indices : List[int]
            A list of the indices which are to be sorted over
        
        feature: np.ndarray
            A 1 dimensional numpy array containing the feature values that are to be sorted over
            
        Returns 
        -----------
        np.ndarray
            A 1 dimensional numpy array containing the indices sorted given the feature values
        """
        cdef cnp.ndarray[npFloat, ndim=1] sort_list
        sort_list = feature[indices]
        sort_list = np.argsort(sort_list)

        return indices[sort_list]