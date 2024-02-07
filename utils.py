import numpy as np
from sklearn.datasets import make_sparse_spd_matrix
import random
from sklearn.metrics import f1_score, precision_score, recall_score
import networkx as nx

def criteria(true_precision, esti_precision, type_='RE', thresh=1e-5):
    if type_ == 'RE':
        return np.linalg.norm(true_precision - esti_precision) / \
            np.linalg.norm(true_precision)
    else:
        A = true_precision.copy()
        A[np.abs(A) >= thresh] = 1
        A[np.abs(A) < thresh] = 0
        B = esti_precision.copy()
        B[np.abs(B) >= thresh] = 1
        B[np.abs(B) < thresh] = 0

        y_true = np.array(A, dtype=bool).flatten()
        y_pred = np.array(B, dtype=bool).flatten()

        if type_ == 'F':
            return f1_score(y_true, y_pred)
        elif type_ == 'recall':
            return recall_score(y_true, y_pred)
        elif type_ == 'precision':
            return precision_score(y_true, y_pred)


def generate_precision_matrix(
        p=100, M=10, style='powerlaw', gamma=2.8, prob=0.2, scale=False, seed=None):
    """
    Generates a sparse precision matrix with associated covariance matrix from a random network.


    Parameters
    ----------
    p : int, optional
        size of the matrix. The default is 100.
    M : int, optional
        number of subblocks. p/M must result in an integer. The default is 10.
    style : str, optional
        Type of the random network. Available network types:
            * 'powerlaw': a powerlaw network.
            * 'erdos': a Erdos-Renyi network.

        The default is 'powerlaw'.
    gamma : float, optional
        parameter for powerlaw network. The default is 2.8.
    prob : float, optional
        probability of edge creation for Erdos-Renyi network. The default is 0.1.
    scale : boolean, optional
        whether Sigma (cov. matrix) is scaled by diagonal entries (as described by Danaher et al.). If set to True, then the generated precision matrix is not
        the inverse of Sigma anymore.
    seed : int, optional
        Seed for network creation and matrix entries. The default is None.
    Returns
    -------
    Sigma : array of shape (p,p)
        Covariance matrix.

    Theta: array of shape (p,p)
        Precisiion matrix, inverse of Sigma. If ``scale=True`` we return ``None``.
    """

    L = int(p / M)
    assert M * L == p

    A = np.zeros((p, p))
    Sigma = np.zeros((p, p))

    if seed is not None:
        nxseed = seed
    else:
        nxseed = None

    for m in np.arange(M):

        if nxseed is not None:
            nxseed = int(nxseed + m)

        if style == 'powerlaw':
            G_m = nx.generators.random_graphs.random_powerlaw_tree(
                n=L, gamma=gamma, tries=max(5 * p, 1000), seed=nxseed)
        elif style == 'erdos':
            G_m = nx.generators.random_graphs.erdos_renyi_graph(
                n=L, p=prob, seed=nxseed, directed=False)
        else:
            raise ValueError(
                f"{style} is not a valid choice for the network generation.")
        A_m = nx.to_numpy_array(G_m)

        # generate random numbers for the nonzero entries
        if seed is not None:
            np.random.seed(seed)

        B1 = np.random.uniform(low=.1, high=.4, size=(L, L))
        B2 = np.random.choice(a=[-1, 1], p=[.5, .5], size=(L, L))

        A_m = A_m * (B1 * B2)

        # only use upper triangle and symmetrize
        #A_m = np.triu(A_m)
        #A_m = .5 * (A_m + A_m.T)

        A[m * L:(m + 1) * L, m * L:(m + 1) * L] = A_m

    row_sum_od = 1.5 * abs(A).sum(axis=1) + 1e-10
    # broadcasting in order to divide ROW-wise
    A = A / row_sum_od[:, np.newaxis]

    A = .5 * (A + A.T)

    # A has 0 on diagonal, fill with 1s
    A = A + np.eye(p)
    assert all(np.diag(A) == 1), "Expected 1s on diagonal"

    # make sure A is pos def
    D = np.linalg.eigvalsh(A)
    if D.min() < 1e-8:
        A += (0.1 + abs(D.min())) * np.eye(p)

    #D = np.linalg.eigvalsh(A)
    #assert D.min() > 0, f"generated matrix A is not positive definite, min EV is {D.min()}"

    Ainv = np.linalg.pinv(A, hermitian=True)

    # scale by inverse of diagonal and 0.6*1/sqrt(d_ii*d_jj) on off-diag
    if scale:
        d = np.diag(Ainv)
        scale_mat = np.tile(np.sqrt(d), (Ainv.shape[0], 1))
        scale_mat = (1 / 0.6) * (scale_mat.T * scale_mat)
        np.fill_diagonal(scale_mat, d)

        Sigma = Ainv / scale_mat
        Theta = None

    else:
        Sigma = Ainv.copy()
        Theta = A.copy()

    assert abs(Sigma.T - Sigma).max() <= 1e-8
    D = np.linalg.eigvalsh(Sigma)
    assert D.min() > 0, "generated matrix Sigma is not positive definite"

    return Sigma, Theta


def generate_sparse_pd_precision(
        dim, k=None, alpha=0.97, type_='eigen', lmin=1e-2, lmax=1., prob=0.2, M=10):
    if k is None:
        k = dim
    if type_ == 'frompaper':
        epsi = 1e-4
        U = np.zeros((dim * dim,))
        idx = random.sample(range(dim * dim), k)
        U[idx] = np.random.choice([-1., 1.], k)
        U = U.reshape(dim, dim)
        A = U.T.dot(U)
        D = np.diag(np.diag(A))
        A2 = np.maximum(np.minimum(A - D, 1), -1)
        A3 = A2 + np.diag(np.diag(D) + 1)
        lmin = np.min(np.linalg.eigvalsh(A3))
        Theta = A3 + max(-1.2 * lmin, epsi) * np.eye(dim, dim)
    elif type_ == 'sklearn':
        Theta = make_sparse_spd_matrix(dim=dim, alpha=alpha)
    elif type_ == 'powerlaw':
        Theta, _ = generate_precision_matrix(p=dim, style='powerlaw', M=M)
    elif type_ == 'erdos':
        Theta, _ = generate_precision_matrix(
            p=dim, prob=prob, style='erdos', M=M)
    return Theta
