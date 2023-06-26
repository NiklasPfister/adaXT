cdef class Node():
    """
    Two types of Nodes. EIther a leaf node or a decision node: 
    """
    def __cinit__(self, *Node leftchild=None, *Node rightchild=None, double threshhold=None, value=None) -> None:
        # Decision node values
        self.leftchild = leftchild
        self.rightchild = rightchild
        self.threshold = threshhold
        self.impurity = self.get_impurity()
        
        # Value of leaf node
        self.value = value
    
    cpdef get_impurity(self):


 