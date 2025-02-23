cimport numpy as cnp
cdef class Node:
    cdef public:
        cnp.ndarray indices
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

cdef class LeafNode(Node):
    cdef public:
        double weighted_samples
        int id
        cnp.ndarray value

cdef class LocalPolynomialLeafNode(LeafNode):
    cdef public:
        double theta0, theta1, theta2
