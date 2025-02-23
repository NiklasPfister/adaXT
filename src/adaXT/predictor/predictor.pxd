cimport numpy as cnp

cdef class Predictor():
    cdef:
        const double[:, ::1] X
        const double[:, ::1] Y
        int n_features
        object root

    cpdef dict predict_leaf(self, double[:, ::1] X)


cdef class PredictorClassification(Predictor):
    cdef:
        readonly double[::1] classes

    cdef int __find_max_index(self, double[::1] lst)

    cdef cnp.ndarray __predict_proba(self, double[:, ::1] X)

    cdef cnp.ndarray __predict(self, double[:, ::1] X)


cdef class PredictorRegression(Predictor):
    pass


cdef class PredictorLocalPolynomial(PredictorRegression):
    pass


cdef class PredictorQuantile(Predictor):
    pass
