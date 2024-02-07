import numpy as np
from scipy.linalg import hadamard


class DimError(Exception):
    pass


def simulate_R(d):
    R = np.diag(2 * np.random.binomial(1, 0.5, d) - np.ones(d))
    return R


def simulate_G(d):
    G = np.diag(np.random.normal(0, 1, d))
    return G


def simulate_fastfood(delta):
    assert isinstance(delta, int)
    d = np.power(2, delta)
    H = hadamard(d)
    H_tilde = (1 / np.sqrt(d)) * hadamard(d)
    B = simulate_R(d)
    G = simulate_G(d)
    permutation = np.random.permutation(d)
    I = np.eye(d)
    Pi = I[permutation, :]

    return H @ G @ Pi @ H_tilde @ B


def triple_rademacher(delta):
    assert isinstance(delta, int)
    d = np.power(2, delta)
    H = hadamard(d)
    D1 = simulate_R(d)
    D2 = simulate_R(d)
    D3 = simulate_R(d)
    return (1. / d**(1.5)) * H @ D3 @ H @ D2 @ H @ D1


class Block_rademacher():
    def __init__(self, M, d, method='rad'):
        """Class used for creating structured random matrices.
        Computes a list of M/d blocks of triple-Rademacher or fast-food like matrices [1].
        Only implemented when the dimension is a power of two.

        Parameters
        ----------
        m : int
            Should be a divisible by d. Corresponds to the sketch dimension.
        d : int
            The dimension. Should be a power of two.
        method : {'rad', 'food'}, optional
            The type of structured, 'rad' for triple-Rademacher matrices and 'food' for  fast-food like matrices, by default 'rad'.

        References
        ----------
        [1] Quoc Viet Le, Tamas Sarlos, Alexander Johannes Smola. Fastfood: Approximate Kernel Expansions in Loglinear Time

        """
        if not np.log2(d).is_integer():
            raise DimError('The dimension is not power of 2, d = {}'.format(d))
        if not M >= d:
            raise DimError('M is lower than d, m = {}, d = {}'.format(M, d))
        if not M % d == 0:
            raise DimError(
                'M should be a multiple of d, M = {}, d = {}'.format(M, d))
        assert method in ['rad', 'food']
        self.m = M
        self.d = d
        self.b = int(M / d)  # number of blocks
        self.p = int(np.log2(d))
        if method == 'rad':
            self.B = [triple_rademacher(self.p).T for i in range(
                self.b)]
        elif method == 'food':
            self.B = [simulate_fastfood(self.p).T for i in range(self.b)]


class SketchingOperator():
    def __init__(self, d, M=50, sigma=1., A=None, method='dg'):
        """Main class for covariance sketching with rank-one projections.
        Two possibilites are implemented: with structured sketching or dense rank-one Gaussian vectors.

        Parameters
        ----------
        d : int
            The dimension of the problem, corresponds to the #rows of the Theta matrix. Should be a power of two.
        M : int, optional
            Sketch size, by default 50.
        sigma : float, optional
            Gaussian scale, by default 1.
        A : ndarray of shape (d, M), optional
            The matrix that stores all the vectors for rank-one projections (a_1, ..., a_M), by default None.
        method : {'dg', 'structured'}, optional
            The choice of randomness for the vectors: 'dg' for dense Gaussian, 'structured' for structured vectors (not i.i.d.), by default 'dg'.
        """
        self.M = M  # the sketch dimension
        self.d = d  # d is the dimension of the Theta matrix
        assert method in ['dg', 'structured']
        self.method = method
        self.sigma = sigma

        if self.method == 'structured':
            self.rad = Block_rademacher(M, d, method='rad')

        # The projection matrix: each column is a d vector
        if A is None:
            if self.method == 'structured':
                Z = []
                for i in range(self.d):  # for each dim
                    for b in range(self.rad.b):  # for each block
                        Z.append(self.rad.B[b][i, :])
                A = self.sigma * np.array(Z).T
            elif self.method == 'dg':  # normalize in the case of dense Gaussian random vectors
                A = (1 / np.sqrt(self.M)) * np.random.randn(self.d, self.M)
            self.A = A
        else:
            assert A.shape == (self.d, self.M)
            self.A = A

    def sketch_covariance(self, X):
        """Computes the sketch A(X) of a d*d covariance matrix X

        Parameters
        ----------
        X : ndarray of shape (d, d)
            The input covariance matrix.

        Returns
        -------
        y: ndarray of shape (M,)
            The sketch.
        """
        if not X.shape == (self.d, self.d) or not np.all(
                np.abs(X - X.T) < 1e-8):
            raise DimError(
                'Input covariance should be a symmetric d*d array')
        # X must be a d*d ndarray symmetric matrix
        # y must be a (M,) tensor
        return np.sum((self.A.T @ X) * self.A.T, 1)  # output a (M,) vector

    def conjugate(self, y):
        """Computes the conjugate transpose A*(y) of the sketching operator.

        Parameters
        ----------
        y: ndarray of shape (M,)
            A sketch vector.

        Returns
        -------
        X : ndarray of shape (d, d)
            A d*d ndarray.
        """
        # dual operator of A
        # y should be a (M,) vector
        return (y[None, :] * self.A) @ self.A.T
