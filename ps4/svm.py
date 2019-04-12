# -*- coding:UTF-8 -*-
import numpy as np
from sklearn.base import BaseEstimator
import cvxopt.solvers

class SVM:
    def __init__(self, kernel, lagrange_multipliers, sv, sv_y, bias):
        self.kernel = kernel
        self.alphas = lagrange_multipliers
        self.sv_x = sv
        self.sv_y = sv_y
        self.b = bias
        assert len(self.alphas)==len(self.sv_x) and len(self.sv_x)==len(self.sv_y)


    def predict(self, x):
        """
        f(x) = Sum a_i*y_i*K(x,x_i) + b
        :param x:
        :return:
        """
        # print('b: ',self.b)
        # print('x shape: ', x.shape)
        preds = self.b

        for a_i, x_i, y_i in zip(self.alphas, self.sv_x, self.sv_y):
            preds+= a_i*y_i*self.kernel(x_i,x)
        return preds.reshape(-1,1) if x.ndim>1 else preds[0]

        # w = ((self.alphas*self.sv_y.A1).T @ self.sv_x).reshape(-1,1)
        # return np.sign(x@w+self.b)

class SVC(BaseEstimator):
    def __init__(self, kernel,C):
        self.kernel = kernel
        self.C = C
        self.svm = None
        self.ZERO_INCVXOPT = 1e-5

    def fit(self,X,y):
        print('X shape: ', X.shape)
        print('y shape: ', y.shape)
        self._solve_svm(X,y)
        return self

    def predict(self,x):
        return self.svm.predict(x) if self.svm else None


    def _solve_svm(self,X, y):
        lagrange_multipliers = self._solve_la_multipliers(X,y)
        sv_indices = lagrange_multipliers > self.ZERO_INCVXOPT

        sv_la = lagrange_multipliers[sv_indices]
        sv_x = X[sv_indices]
        sv_y = y[sv_indices]

        # print('sv_y shape: ', sv_y.shape)
        # print('svm',SVM(self.kernel,sv_la,sv_x,sv_y,0.0).predict(sv_x).shape)
        bias = np.mean(sv_y-SVM(self.kernel,sv_la,sv_x,sv_y,0.0).predict(sv_x))
        self.svm = SVM(self.kernel,sv_la,sv_x,sv_y,bias)

    def _solve_la_multipliers(self, X, y):
        n_samples, n_features = X.shape
        K_matrix = self._kenerel_matrix(X)

        P = cvxopt.matrix(np.outer(y, y) * K_matrix)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))

        G_soft = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_soft = cvxopt.matrix(np.ones(n_samples) * self.C)

        G = cvxopt.matrix(np.vstack((G, G_soft)))
        h = cvxopt.matrix(np.vstack((h, h_soft)))

        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)

        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(sol['x'])

    def _kenerel_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self.kernel(x_i, x_j)
        return K

    def decision_function(self,x):
        return self.predict(x)








