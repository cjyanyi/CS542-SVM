# -*- coding:UTF-8 -*-

import numpy as np
import numpy.linalg as la


class Kernel:
    """Implements linear, gaussian,
    """
    @staticmethod
    def linear():
        def f(x, y):
            return np.inner(x,y)
        return f

    @staticmethod
    def rbf(sigma):
        def f(x, y):
            if y.ndim>1:
                exponent = -la.norm(x-y,axis=1) ** 2 / (2 * sigma ** 2)
            else:
                exponent = -la.norm(x - y) ** 2 / (2 * sigma ** 2)
            return np.exp(exponent)
        return f

    @staticmethod
    def hyperbolic_tangent(kappa, c):
        def f(x, y):
            return np.tanh(kappa * np.inner(x, y) + c)
        return f

    @staticmethod
    def polynomial(dimension):
        def f(x, y):
            return np.inner(x, y) ** dimension
        return f