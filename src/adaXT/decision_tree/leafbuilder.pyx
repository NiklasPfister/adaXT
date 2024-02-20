# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from .nodes cimport LeafNode
import numpy as np
cimport numpy as cnp

cdef class LeafBuilder:
    def __cinit__(self, double[:, ::1] x, double[::1] y, int[::1] all_idx):
       self.x = x
       self.y = y

    cpdef LeafNode build_leaf(self,
                             int leaf_id,
                             int[::1] indices,
                             int depth,
                             double impurity,
                             int n_samples,
                             object parent):
        raise NotImplementedError("Build leaf not implemented for this LeafBuilder")


cdef class LeafBuilderClassification(LeafBuilder):
    def __cinit__(self, double[:, ::1] x, double[::1] y, int[::1] all_idx):
        self.classes = np.array(np.unique(y.base[all_idx]), dtype=np.double)
        self.n_classes = self.classes.shape[0]

    cdef double[::1] __get_mean(self, int[::1] indices, int n_samples):
        cdef:
            cnp.ndarray[double, ndim=1] ret
            int i, idx

        ret = np.zeros(self.n_classes)
        for idx in range(n_samples):
            for i in range(self.n_classes):
                if self.y[indices[idx]] == self.classes[i]:
                    ret[i] += 1  # add 1, if the value is the same as class value
                    break

        ret = ret / n_samples
        return ret

    cpdef LeafNode build_leaf(self,
                             int leaf_id,
                             int[::1] indices,
                             int depth,
                             double impurity,
                             int n_samples,
                             object parent):
        cdef double[::1] mean = self.__get_mean(indices, n_samples)
        return LeafNode(leaf_id, indices, depth, impurity, n_samples, mean, parent)

cdef class LeafBuilderRegression(LeafBuilder):

    cdef double __get_mean(self, int[::1] indices):
        cdef:
            int i
            double sum = 0.0
            int count = len(indices)

        for i in indices:
           sum += self.y[i]

        return sum / count

    cpdef LeafNode build_leaf(self,
                             int leaf_id,
                             int[::1] indices,
                             int depth,
                             double impurity,
                             int n_samples,
                             object parent):

        cdef double[::1] mean = np.array(self.__get_mean(indices), dtype=np.double, ndmin=1)
        return LeafNode(leaf_id, indices, depth, impurity, n_samples, mean, parent)
