cimport cython

@cython.profile(False)
@cython.binding(False)
@cython.linetrace(False)
cdef inline double dsum(double[::1] arr) noexcept:
    cdef size_t i, I
    cdef double res
    I = arr.shape[0]
    res = 0.0
    for i in range(I):
        res += arr[i]

    return res

