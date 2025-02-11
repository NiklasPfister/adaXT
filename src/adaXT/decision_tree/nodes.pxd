cdef class Node:
    cdef public: 
        int[:] indices
        int depth 
        double impurity 
        object parent
        bint visited 
        bint is_leaf 


cdef class DecisionNode(Node):
    cdef public: 
        double threshold
        int split_idx
        object left_child
        object right_child


