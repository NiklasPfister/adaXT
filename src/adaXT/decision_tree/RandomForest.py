from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory

from adaXT.decision_tree import DecisionTree
from adaXT.decision_tree.criteria import Squared_error
from adaXT.decision_tree.criteria import Criteria

import numpy as np
import time

class SharedNumpyArray:
    '''
    Wraps a numpy array so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing.
    '''
    def __init__(self, array):
        '''
        Creates the shared memory and copies the array therein
        '''
        # create the shared memory location of the same size of the array
        self._shared = SharedMemory(create=True, size=array.nbytes)
        
        # save data type and shape, necessary to read the data correctly
        self._dtype, self._shape = array.dtype, array.shape
        
        # create a new numpy array that uses the shared memory we created.
        # at first, it is filled with zeros
        res = np.ndarray(
            self._shape, dtype=self._dtype, buffer=self._shared.buf
        )
        
        # copy data from the array to the shared memory. numpy will
        # take care of copying everything in the correct format
        res[:] = array[:]

    def read(self):
        '''
        Reads the array from the shared memory without unnecessary copying.
        '''
        # simply create an array of the correct shape and type,
        # using the shared memory location we created earlier
        return np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)
        
    def unlink(self):
        '''
        Releases the allocated memory. Call when finished using the data,
        or when the data was copied somewhere else.
        '''
        self._shared.close()
        self._shared.unlink()


class RandomForrest:
    '''
    The Random Forrest 
    '''
    def __init__(
            self,
            forrest_type: str, 
            n_estimators: int = 100,
            criterion: Criteria = Squared_error,
            bootstrap: bool = True, 
            n_jobs: int = 1, 
            max_samples: int = None,
            per_tree_features = "None"):
        """
        Parameters
        ----------
        forrest_type : str
            Classification or Regression
        n_estimators : int, default=100
            The number of trees in the forest.
        criterion : Criteria, default=Squared_errror
            The criteria function used to evaluate a split
        bootstrap : bool, default=True
            Whether bootstrap is used when building trees
        n_jobs : int, default=1
            The number of processes created to train the forrest, -1 means using all proccesors 
        max_samples : int, default=None
            The number of samples drawn from the feature values
        per_tree_features:
            features passed directly on to the tree, SHOULD BE IMPLEMENTED LATER
        """
        self.forrest_type = forrest_type
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.bootstrap = bootstrap 
        self.n_jobs = n_jobs 
        self.max_samples = max_samples 
        self.per_tree_features = per_tree_features        

    def _fit_tree(self, tree:DecisionTree):
        tree.fit(self.features.read(), self.outcomes.read())
        return tree
        
    def __build_trees(self):
        if(self.n_jobs == 1):
            for tree in self.trees:
                self._fit_tree(tree)
        else:
            pool = Pool(self.n_jobs)
            self.trees = pool.map(self._fit_tree, self.trees)
            pool.close()
            pool.join()
    
    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.features = SharedNumpyArray(X)
        self.outcomes = SharedNumpyArray(Y)
        #Should check that bootstrap is false, but sample_indices is set
        
        num_rows, _ = X.shape
        
        # Bootstrap
        if self.bootstrap:
            self.trees = [DecisionTree(tree_type=self.forrest_type, criteria=self.criterion, sample_indices=self.__get_sample_indices(num_rows)) for _ in range(self.n_estimators)]
        else:
            self.trees = [DecisionTree(tree_type=self.forrest_type, criteria=self.criterion) for _ in range(self.n_estimators)]

        # Fit trees
        self.__build_trees()
    
        return self
    
    def __get_sample_indices(self, n_obs):
        if self.max_samples is None:
            self.max_samples = n_obs
        return np.random.randint(low=0, high=n_obs, size=self.max_samples)

if __name__ == '__main__':
    # Setup Forrest parameters
    num_trees = 1000

    # Get Dataset
    n = 100
    m = 25
    np.random.seed(2024)
    X = np.array(np.random.uniform(0, 100, (n, m)), dtype=np.double)
    Y = np.array(np.random.uniform(0, 5, n), dtype=np.double)
    
    for i in range(1, 7):
        # Create forrest
        adaForrest = RandomForrest("Regression", n_estimators=num_trees, criterion=Squared_error, bootstrap=True, n_jobs=i, max_samples=500, per_tree_features="None")

        # Fit forrest
        start_time_fit = time.perf_counter()
        adaForrest.fit(X, Y)
        end_time_fit = time.perf_counter()
        elapsed_fit = (end_time_fit - start_time_fit)
        print("Fitted in", elapsed_fit, "seconds, with", i, "processes")

    # Use forrest - not implemented
    #adaForrest.predict(X)
    