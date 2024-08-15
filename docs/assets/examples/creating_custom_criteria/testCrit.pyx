from adaXT.criteria cimport Criteria

cdef class Partial_linear(Criteria):

    # Custom mean function, such that we don't have to loop through twice.
    cdef (double, double) custom_mean(self, int[::1] indices):
        cdef:
            double sumX, sumY
            int i
            int length = indices.shape[0]
        sumX = 0.0
        sumY = 0.0
        for i in range(length):
            sumX += self.x[indices[i], 0]
            sumY += self.y[indices[i]]

        return ((sumX / (<double> length)), (sumY/ (<double> length)))

    cdef (double, double) theta(self, int[::1] indices):
        """
        Estimates regression parameters for a linear regression of the response on
        the first coordinate, i.e., Y is approximated by theta0 + theta1 * X[:, 0].
        ----------

        Parameters
        ----------
        indices : memoryview of NDArray
            The indices to calculate

        Returns
        -------
        (double, double)
            where first element is theta0 and second is theta1
        """
        cdef:
            double muX, muY, theta0, theta1
            int length, i
            double numerator, denominator
            double X_diff

        length = indices.shape[0]
        denominator = 0.0
        numerator = 0.0
        muX, muY = self.custom_mean(indices)
        for i in range(length):
            X_diff = self.x[indices[i], 0] - muX
            numerator += (X_diff)*(self.y[indices[i]]-muY)
            denominator += (X_diff)*X_diff
        if denominator == 0.0:
            theta1 = 0.0
        else:
            theta1 = numerator / denominator
        theta0 = muY - theta1*muX
        return (theta0, theta1)

    cpdef double impurity(self, int[::1] indices):
        cdef:
            double step_calc, theta0, theta1, cur_sum
            int i, length

        length = indices.shape[0]
        theta0, theta1 = self.theta(indices)
        cur_sum = 0.0
        for i in range(length):
            step_calc = self.y[indices[i]] - theta0 - theta1 * self.x[indices[i], 0]
            cur_sum += step_calc*step_calc
        return cur_sum
