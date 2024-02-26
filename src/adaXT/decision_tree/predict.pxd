
cdef class Predict():
    cdef:
        double[:, ::1] X
        double[::1] Y
        int n_features
        object root


    cdef double[:, ::1] __check_dimensions(self, object X)

    cpdef double[:, ::1] predict_leaf_matrix(self, object X, bint scale=*)

    cpdef list predict_proba(self, object X)


cdef class PredictClassification(Predict):
    cdef:
        double[::1] classes

    cdef int __find_max_index(self, double[::1] lst)

    cpdef list predict_proba(self, object X)


cdef class PredictRegression(Predict):
    pass
