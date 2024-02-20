cimport numpy as cnp

cdef class Node:
    cdef public:
        int[::1] indices
        int depth
        double impurity
        int n_samples


cdef class DecisionNode(Node):
    cdef public:
        double threshold
        int split_idx
        object left_child
        object right_child
        object parent


cdef class LeafNode(Node):
    cdef public:
        cnp.ndarray value
        int id
        object parent
