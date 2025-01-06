cimport numpy as cnp

cdef class Predictor():
    cdef:
        double[:, ::1] X
        double[:, ::1] Y
        int n_features
        object root

    cpdef dict predict_leaf(self, cnp.ndarray X)


cdef class PredictorClassification(Predictor):
    cdef:
        readonly double[::1] classes

    cdef int __find_max_index(self, double[::1] lst)

    cdef cnp.ndarray __predict_proba(self, cnp.ndarray X)

    cdef cnp.ndarray __predict(self, cnp.ndarray X)


cdef class PredictorRegression(Predictor):
    pass


cdef class PredictorLocalPolynomial(PredictorRegression):
    pass


cdef class PredictorQuantile(Predictor):
    pass
