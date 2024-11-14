# Tools for covariance
import numpy as np
from celer import Lasso as celer_Lasso
from skglm.solvers import AndersonCD
from skglm.penalties import L1
from skglm.datafits import QuadraticHessian
from skglm import GeneralizedLinearEstimator


class L1PenalizedQP():
    # solves min_x (1/2) x^T A x + <b, x> + alpha |x|_1
    def __init__(self, alpha, verbose=0, max_iter=50, max_epochs=50000, p0=10, tol=0.0001):
        self.alpha = alpha
        self.pen = L1(alpha)
        self.solv = AndersonCD(warm_start=False, verbose=verbose, fit_intercept=False,
                               max_iter=max_iter, max_epochs=max_epochs, p0=p0, tol=tol)
        self.qpl1 = GeneralizedLinearEstimator(
            QuadraticHessian(), self.pen, self.solv)

    def fit(self, A, b):
        self.qpl1.fit(A, b)
        return self


class CovSolver():
    # Solve min_S > 0 f(S) + lamda |S^-1|_1
    def __init__(self, f, df, reg=1.0, max_iter=1000, max_iter_qp=100, thresh=1e-6):
        self.max_iter = max_iter
        self.max_iter_qp = max_iter_qp
        self.reg = reg
        self.f = f
        self.df = df
        self.thresh

    def get_slices(self, col, indices):
        indices_minus_col = np.concatenate(
            [indices[:col], indices[col + 1:]])
        _11 = indices_minus_col[:, None], indices_minus_col[None]
        _12 = indices_minus_col, col
        _21 = col, indices_minus_col
        _22 = col, col

        return _11, _12, _21, _22

    def min_gamma(self, w_12, Theta_11, S_11, s_12, s_22):
        tmp = Theta_11 @ w_12
        a = (tmp * (S_11 @ tmp)).sum() - 2 * (s_12 * tmp).sum() + s_22
        return (1.0 / (2*self.reg))*(-1 + np.sqrt(1+4*a*self.reg))

    def fit(self, S):

        p = S.shape[0]
        W = np.diag(np.diag(S + self.reg))
        Theta = np.diag(1/np.diag(S + self.reg))

        qp_solver = L1PenalizedQP(
            alpha=2*self.reg, max_iter=self.max_iter_qp)  # careful of the 2

        indices = np.arange(p)
        loop = True
        while loop:
            for col in range(p):
                _11, _12, _21, _22 = self.get_slices(col, indices)

                Theta_11 = Theta[_11]
                theta_22 = Theta[_22]
                theta_12 = Theta[_12]

                W_11 = W[_11]
                w_22 = W[_22]
                w_12 = W[_12]

                S_11 = S[_11]
                s_12 = S[_12]
                s_22 = S[_22]

                w_22_new = self.min_gamma(w_12, Theta_11, S_11, s_12, s_22)
                # this is the bad part: it is d^3, should do a dedicated solver
                A = (1.0 / w_22_new) * Theta_11 @ S_11 @ Theta_11 + \
                    self.reg * Theta_11
                b = - (2.0 / w_22_new) * Theta_11 @ s_12

                # schur = w_22 - np.einsum('i, ij, i', w_12, Theta_11, w_12)
                w_12_new = qp_solver.fit(A, b).coef_
                theta_12_new = - (1.0 / w_22_new) * (Theta_11 @ w_12_new)

                # update covariance and its inverse
                W[_22] = w_22_new
                W[_12] = w_12_new

                Theta[_11] = Theta_11 + w_22_new * \
                    theta_12_new @ theta_12_new.T
                Theta[_22] = 1.0 / w_22_new
                Theta[_12] = theta_12_new
                Theta[_21] = theta_12_new

        return W
