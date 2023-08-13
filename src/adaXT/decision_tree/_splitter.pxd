cimport numpy as cnp
cdef class Splitter:
    
    @staticmethod 
    cdef Splitter gen_cla_splitter(cnp.ndarray[cnp.double_t, ndim=2] features, cnp.ndarray[cnp.int64_t, ndim=1] outcomes)

    @staticmethod 
    cdef Splitter gen_reg_splitter(cnp.ndarray[cnp.double_t, ndim=2] features, cnp.ndarray[cnp.double_t, ndim=1] outcomes)

    cdef cnp.ndarray sort_feature(self, list[int] indices, cnp.ndarray[cnp.cdouble_t, ndim=1] feature)