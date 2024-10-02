cimport numpy as cnp

cdef class Predict():
    cdef:
        double[:, ::1] X
        double[:, ::1] Y
        int n_features
        object root

    cpdef dict predict_leaf(self, cnp.ndarray X)


cdef class PredictClassification(Predict):
    cdef:
        double[::1] classes

    cdef int __find_max_index(self, double[::1] lst)

    cdef cnp.ndarray __predict_proba(self, cnp.ndarray X)

    cdef cnp.ndarray __predict(self, cnp.ndarray X)


cdef class PredictRegression(Predict):
    pass


cdef class PredictLocalPolynomial(PredictRegression):
    pass


cdef class PredictQuantile(Predict):
    pass
