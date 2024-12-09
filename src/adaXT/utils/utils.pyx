
cdef inline double dsum(double[::1] arr, int[::1] indices):
    cdef size_t i
    cdef double res
    res = 0.0
    for i in indices:
        res += arr[i]

    return res

