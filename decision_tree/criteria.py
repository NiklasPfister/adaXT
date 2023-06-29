# Could proberbly be quicker in Cython
def gini_index(y):
    """
    Calculates the gini coefficient with the formula gini = 1 - sum(P_i**2) where P_i is the probability of class i E.G. what is 
    the chance that an observation is in the class i out of all the classes we have in y

    Args:
        data: A list of the outcomes(y-values)

    Returns:
        returns the gini index of the data (double) 
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