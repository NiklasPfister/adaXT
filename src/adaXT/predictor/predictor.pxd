cimport numpy as cnp

cdef class Predictor():
    cdef:
        # Must be ndarray such that it and all children can be pickled
        cnp.ndarray X
        cnp.ndarray Y
        int n_features
        object root

    cpdef dict predict_leaf(self, double[:, ::1] X)


cdef class PredictorClassification(Predictor):
    cdef:
        readonly cnp.ndarray classes

    cdef int __find_max_index(self, float[::1] lst)

    cdef cnp.ndarray __predict_proba(self, double[:, ::1] X)

    cdef cnp.ndarray __predict(self, double[:, ::1] X)


cdef class PredictorRegression(Predictor):
    pass


cdef class PredictorLocalPolynomial(PredictorRegression):
    pass


cdef class PredictorQuantile(Predictor):
    pass
