
import numpy as np
import time
from sklearn.covariance import graphical_lasso
from sketchtools import SketchingOperator


class GlassoSDPError(Exception):
    pass


class NotSketchingOperatorError(Exception):
    pass


def decoder_glasso(y, 
                   sketching_op,
                   S0,
                   max_iter=100,
                   max_iter_glasso=100,
                   reg=1.0,
                   step_size=1e-1,
                   verbose=False,
                   tol=1e-5,
                   tol_glasso=1e-4,
                   n_verbose=100,
                   tolog=False,
                   ):
    """Decoder proposed in the paper.
    It recovers a d*d precision matrix from an empirical sketch of the empirical covariance of size M.

    Parameters
    ----------
    y : ndarray of shape (M,)
        The empirical sketch of the data, equivalently the sketch of the empirical covariance.
    sketching_op : SketchingOperator
        The SketchingOperator used for sketching the empirical covariance.
    S0 : ndarray of shape (d, d)
        Initialization of the estimated covariance matrix.
    max_iter : int, optional
        Number of iterations for the algorithm, by default 100.
    max_iter_glasso : int, optional
        Number of maximal iterations for the inner graphical lasso, by default 100.
    reg : float, optional
        Regularization parameter for the ell_1 penalty, by default 1.0.
    step_size : float, optional
        Step-size for the algorithm, by default 1e-1.
    verbose : bool, optional
        Verbose parameter, by default False.
    tol : float, optional
        Tolerance parameter for the convergence test, by default 1e-5.
    tol_glasso : float, optional
        Tolerance parameter for the graphical lasso, by default 1e-4.
    n_verbose : int, optional
        Number of steps before printing, by default 100.
    tolog : bool, optional
        Whether to store the intermediate results in a dictionary, by default False.

    Returns
    -------
    precision: ndarray of shape (d, d)
        The estimated precision matrix
    log: dictionary, when tolog = True
        Contains intermediate results.

    """

    if not isinstance(sketching_op, SketchingOperator):
        raise NotSketchingOperatorError(
            'sketching_op  must be a SketchingOperator.')
    go = True
    nit = 0
    S_old = S0
    if tolog:
        res = {}
        res['all_cov'] = []
        res['all_cov'].append(S_old)
    while go:
        st = time.time()
        s = sketching_op.sketch_covariance(S_old)  # sketch A(Sigma)
        # Gradient step on Sigma -> (1/2) \|A Sigma - y \|_2^2
        grad = sketching_op.conjugate(s - y)  # Gradient
        cov = S_old - step_size * grad  # Step

        st_glasso = time.time()
        try:
            S_new, precision = graphical_lasso(cov,
                                               alpha=reg * step_size,
                                               max_iter=max_iter_glasso,
                                               eps=1e-7,
                                               mode='cd',
                                               verbose=False,
                                               tol=tol_glasso
                                               )
        except FloatingPointError as exception:
            print(exception)
            raise GlassoSDPError('Glasso SDP Error')
        except ValueError as exception:
            print(exception)
            raise GlassoSDPError('Glasso Value Error')
        ed_glasso = time.time()

        if nit >= max_iter:
            print('--- max_iter attained ---')
            go = False

        norm_iter = np.linalg.norm(S_old - S_new)
        if norm_iter <= tol:
            print('Break: convergence')
            go = False

        if verbose:
            if not (nit) % n_verbose:
                print('Iteration {0} done in {1:.4f} secs (maxiter = {2})'.format(
                    nit, ed_glasso - st, max_iter))
                print('Norm iterates = {0:.5f}'.format(norm_iter))
        if tolog:
            res['all_cov'].append(S_new)
        nit += 1
        S_old = S_new

    if tolog:
        res['cov'] = S_new
        res['precision'] = np.linalg.pinv(S_new, hermitian=True)
        return precision, res
    else:
        return precision
