import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef class Splitter:
    """
    Splitter class used to create splits of the data
    """

    def __cinit__(self):
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
        """
        self.features = None
        self.outcomes = None
        self.n_features = None


    @staticmethod
    cdef Splitter gen_cla_splitter(cnp.ndarray[cnp.double_t, ndim=2] features, cnp.ndarray[cnp.int64_t, ndim=1] outcomes):
        cdef Splitter out = Splitter()
        out.features = features
        out.outcomes = outcomes
        out.n_features = features.shape[0]
        return out

    @staticmethod
    cdef Splitter gen_reg_splitter(cnp.ndarray[cnp.double_t, ndim=2] features, cnp.ndarray[cnp.double_t, ndim=1] outcomes):
        cdef Splitter out = Splitter()
        out.features = features
        out.outcomes = outcomes
        out.n_features = features.shape[0]
        return out

    cdef cnp.ndarray sort_feature(self, list[int] indices, cnp.ndarray[cnp.cdouble_t, ndim=1] feature):
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
        cdef cnp.ndarray[cnp.double_t, ndim=1] sort_list
        sort_list = feature[indices]
        sort_list = np.argsort(sort_list)

        return indices[sort_list]