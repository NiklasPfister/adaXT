import numpy.typing as npt
import numpy as np

def gini_index(x: npt.NDArray, y: npt.NDArray) -> float:
    """
    Calculates the gini coefficient given outcomes, y.

    Parameters
    ----------
    x : npt.NDArray
        features, not used in this implementation
    y : npt.NDArray
        1-dimensional outcomes

    Returns
    -------
    float  
        The gini coefficient
    """
    # Finds unique y values
    class_labels = list(set(y))
    
    sum = 0
    y_len = len(y)
    for cls in class_labels:
        n_in_class = len([val for val in y if val == cls])
        proportion_cls = n_in_class / y_len
        sum += proportion_cls**2
    return 1 - sum    

def variance(x: npt.NDArray, y:npt.NDArray) -> float:
    """
    Calculates the variance 

    Parameters
    ----------
    x : npt.NDArray
        features, not used in this implementation
    y : npt.NDArray
        1-dimensional outcomes

    Returns
    -------
    float
        variance of the y data
    """
    assert(y.ndim == 1), f'Number of dimensions is not correct it is {y.ndim}'
    cur_sum = 0
    mu = np.mean(y)
    for val in y:
        cur_sum += (val - mu)**2
    return cur_sum/len(y)
