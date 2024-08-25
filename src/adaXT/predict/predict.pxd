cimport numpy as cnp

cdef class Predict():
    cdef:
        double[:, ::1] X
        double[::1] Y
        int n_features
        object root

    cdef double[:, ::1] __check_dimensions(self, object X)

    cpdef dict predict_leaf(self, object X)

    cpdef list predict_proba(self, object X)


cdef class PredictClassification(Predict):
    cdef:
        double[::1] classes

    cdef int __find_max_index(self, double[::1] lst)

    cdef cnp.ndarray __predict_proba(self, object X)

    cdef cnp.ndarray __predict(self, object X)


cdef class PredictRegression(Predict):
    pass


cdef class PredictLinearRegression(PredictRegression):
    pass


cdef class PredictQuantile(Predict):
    pass
