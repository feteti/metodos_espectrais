import numpy as np
from functools import reduce
from typing import List, Any
import numpy.typing as npt


class Polynomial:
    def __init__(self, coefficients: np.array):
        self.coefficients = coefficients
        self.degree = len(coefficients) - 1

    def __add__(self, other):
        gt, lt = self.coefficients, other.coefficients
        gt, lt = (gt, lt) if self.degree >= other.degree else (lt, gt)
        return Polynomial(gt + np.concatenate((np.zeros(gt.size - lt.size), lt)))

    def __mul__(self, other):
        return Polynomial(np.convolve(self.coefficients, other.coefficients))

    def scalar_mul(self, scalar: float):
        return Polynomial(scalar * self.coefficients)

    def eval(self, x: float):
        if hasattr(x, "__iter__"):
            return np.array(list(map(lambda y: self.eval(y), x)))

        aux = np.array([x ** (i) for i in range(self.degree, -1, -1)])
        return np.sum(self.coefficients * aux)

    @classmethod
    def lagrange_interpolator(cls, xv: np.array, yv: np.array):
        n = yv.size
        return reduce(
            lambda x, y: x + y,
            [cls.lagrange_polynomial(xv, j).scalar_mul(yv[j]) for j in range(n)],
        )

    @classmethod
    def lagrange_polynomial(cls, xv: np.array, j: int):
        numerator = reduce(
            lambda x, y: x * y,
            [
                cls(np.array([1, -1 * xv[k]])) if k != j else cls(np.array([1]))
                for k in range(xv.size)
            ],
        )

        denominator = reduce(
            lambda x, y: x * y,
            [xv[j] - xv[k] if k != j else 1 for k in range(xv.size)],
        )
        return numerator.scalar_mul(1 / denominator)


class BarycentricLagrangeInterpolator:
    def __init__(self, xv: np.array, yv: np.array):
        self.xv = xv
        self.yv = yv

    @classmethod
    def _get_weights(cls, xv: np.array):
        return np.array([cls._get_weight(xv, j) for j in range(xv.size)])

    @classmethod
    def _get_weight(cls, xv: np.array, j: int):
        aux = xv - xv[j]
        return 1 / np.product(aux[aux != 0])

    def eval(self, x: float):
        if hasattr(x, "__iter__"):
            return np.array(list(map(lambda y: self.eval(y), x)))
        weights = self._get_weights(self.xv)
        aux = x - self.xv
        if np.any(aux == 0):
            return self.yv[aux == 0][0]
        aux = np.reciprocal(aux)
        return np.sum((weights * aux * self.yv)) / np.sum(weights * aux)


def generalized_diff_matrix(xv: np.array) -> np.ndarray:
    X = np.repeat(xv.reshape(-1, 1), xv.size, axis=1)
    dX = (X - X.T) + np.eye(xv.size)
    aj = np.prod(dX, axis=1).reshape(-1, 1)
    D = aj @ (np.reciprocal(aj).T) * np.reciprocal(dX)
    return D - np.diag(np.sum(D.T, axis=0))


def chebyshev_points(n: int):
    theta = np.linspace(0, np.pi, n + 1)
    return np.cos(theta)


def chebyshev_base_matrix(n: int, x=None):
    if x is None:
        theta = np.linspace(0, np.pi, n + 1).reshape(-1, 1)
        aux = np.repeat(np.array(range(0, n + 1)).reshape(1, -1), n + 1, axis=0)
        B = np.repeat(theta, n + 1, axis=1)
        return np.cos(B * aux)
    N = n + 1
    P = np.zeros((len(x), N))
    P[:, 0] = 1
    P[:, 1] = x
    for k in range(1, n):
        P[:, k + 1] = (2 * P[:, k]) * x - P[:, k - 1]
    return P


def inverse_chebyshev_base_matrix(n: int):
    cb = np.transpose(chebyshev_base_matrix(n))
    cb[[0, 0, -1, -1], [0, -1, 0, -1]] = 1 / 2 * cb[[0, 0, -1, -1], [0, -1, 0, -1]]
    cb[1:-1, 1:-1] = 2 * cb[1:-1, 1:-1]
    return 1 / n * cb


def chebyshev_coefficients(y):
    n = y.size - 1
    V = np.append(y, np.flip(y)[1:-1])
    Y = np.fft.fft(V)
    Y = Y / n
    Y[[0, n]] = Y[[0, n]] / 2
    return np.real(Y[0 : n + 1])


def chebyshev_polynomial(n: int) -> Polynomial:
    T_0 = Polynomial(np.array([1.0]))
    T_1 = Polynomial(np.array([1.0, 0.0]))

    if n == 0:
        return T_0
    elif n == 1:
        return T_1

    return (T_1 * chebyshev_polynomial(n - 1)).scalar_mul(2) + chebyshev_polynomial(
        n - 2
    ).scalar_mul(-1)


def q_matrix_cheb(n: int) -> np.ndarray:
    """
    Implements the Chebyshev Quadrature Matrix.
    """
    Q = np.zeros((n + 2, n + 1))
    for i in range(1, n):
        Q[i + 2, i + 1] = 1 / (2 * (i + 1) + 2)
        Q[i, i + 1] = -1 / (2 * (i + 1) - 2)
        Q[0, i + 1] = -1 * (-1) ** (i + 1) / ((i + 1 - 1) * (i + 1 + 1))

    Q[[0, 1], 0] = 1
    Q[[0, 2], 1] = [-0.25, 0.25]
    Q[[0, 1], 0] = 1
    Q[0, 1] = -0.25
    Q[1, 2] = -0.5
    return Q[: n + 1, :n]


def q_matrix_leg(n: int) -> np.ndarray:
    M = np.zeros((n + 2, n + 2))
    M[0, 0] = 1
    M[0, 1] = -1 / 3

    for k in range(1, n + 1):
        M[k, k - 1] = 1 / (2 * (k + 1) - 3)
        M[k, k + 1] = -1 / (2 * (k + 1) + 1)

    M[n + 1, n] = 1 / (2 * n + 1)

    return M[:, : n + 1]


def cheby_to_monomial(n: int) -> np.ndarray:
    if n <= 2:
        return np.eye(n)

    M = np.zeros((n, n))
    M[0, 0] = 1
    M[1, 1] = 1

    for i in range(2, n):
        M[i, :] = 2 * np.hstack([[0], M[i - 1, 0 : n - 1]]) - M[i - 2, :]

    return M.T


def leg_to_monomial(n: int) -> np.ndarray:
    if n <= 2:
        return np.eye(n)

    M = np.zeros((n, n))
    M[0, 0] = 1
    M[1, 1] = 1

    for i in range(2, n):
        M[i, :] = (
            (
                (2 * i - 1) * np.hstack([[0], M[i - 1, 0 : n - 1]])
                - (i - 1) * M[i - 2, :]
            )
            * 1
            / i
        )

    return M.T


def cheby_to_leg(n: int, v: np.ndarray) -> np.ndarray:
    A = cheby_to_monomial(n)
    return np.linalg.solve(leg_to_monomial(n), A @ v)


def legendre_diff_matrix(n: int) -> np.ndarray:
    if n == 0:
        return 0
    if n == 1:
        return np.ndarray([[0, 0], [0, 1]])

    Z = np.zeros((n + 1, n + 1))
    Z[1, 0] = 1
    for k in range(1, n):
        Z[k + 1, k] = 2 * (k + 1) - 1
        Z[k + 1, :] = Z[k + 1, :] + Z[k - 1, :]

    return Z.T


def legendre_basis(n, x):
    N = n + 1
    P = np.zeros((len(x), N))
    P[:, 0] = 1
    P[:, 1] = x
    for k in range(1, n):
        P[:, k + 1] = (((2 * k + 1) * P[:, k]) * x - k * P[:, k - 1]) / (k + 1)
    return P


def legendre_lobbato_points(n: int) -> np.ndarray:
    x, _ = np.polynomial.legendre.leggauss(n - 2)

    return np.hstack([-1, x, 1])


def GLLobatto(n):
    N1 = n + 1
    x = chebyshev_points(n)[::-1]
    P = np.zeros((N1, N1))

    while True:
        xold = x

        P[:, 0] = 1
        P[:, 1] = x

        for k in range(1, n):
            P[:, k + 1] = (((2 * k + 1) * P[:, k]) * x - k * P[:, k - 1]) / (k + 1)

        x = xold - (x * P[:, N1 - 1] - P[:, n - 1]) / (N1 * P[:, N1 - 1])

        if np.max(np.abs(x - xold)) <= 1e-13:
            w = 2.0 / (n * N1 * P[:, N1 - 1] ** 2)
            invP = (
                np.diag((2 * np.array(list(range(n + 1))) + 1) / 2) @ (P.T) @ np.diag(w)
            )
            invP[n, :] = invP[n, :] * n / (2 * n + 1)
            return (x, w, P, invP)


def b_fourrier(x, a, b, m):
    L = (b - a) / 2

    BF = np.ones((len(x), 2 * m + 1))

    for k in range(1, m + 1):
        BF[:, 2 * k - 1] = np.cos(k * np.pi * (x) / L)
        BF[:, 2 * k] = np.sin(k * np.pi * (x) / L)

    return BF


def base_fourier(a, b, m):
    L = (b - a) / 2
    ts = np.linspace(a, b, 2 * m + 1, endpoint=False)

    BF = np.ones((2 * m + 1, 2 * m + 1))
    for k in range(1, m + 1):
        BF[:, 2 * k - 1] = np.cos(k * np.pi * (ts) / L)
        BF[:, 2 * k] = np.sin(k * np.pi * (ts) / L)

    BF_I = BF.T.copy()
    BF_I[0, :] = BF_I[0, :] / 2
    BF_I = 2 * BF_I / (2 * m + 1)

    DFS = np.zeros((2 * m + 1, 2 * m + 1))
    
    for k in range(1, m + 1):
        DFS[2 * k, 2 * k - 1] = -k
        DFS[2 * k - 1, 2 * k] = k

    DFS = np.pi * DFS / L
    DFP = BF @ DFS @ BF_I

    JFS = np.zeros((2 * m + 1, 2 * m + 1))
    JFS[0,0] = 1
    for k in range(1, m + 1):
        JFS[0, 2 * k] = (-1)**k *1 / k
        JFS[2 * k - 1, 2 * k] = -1 / k
        JFS[2 * k, 2 * k - 1] = 1 / k

    JFP = BF @ JFS @ BF_I
    return BF, BF_I, JFP, JFS, DFP, DFS, ts


def standard_fft(y_fft):
    r = np.zeros_like(y_fft)
    m = (len(r) - 1) // 2
    r[0] = y_fft[0]
    for i in range(1, m + 1):
        r[2 * i - 1] = y_fft[i] + y_fft[-i]
        r[2 * i] = (y_fft[i] - y_fft[-i]) * 1j
    return np.real(r) * 1 / len(r)


if __name__ == "__main__":
    BF, BF_I, JFP, JFS, DFP, DFS, ts = Base_Fourier_A(-1, 1, 2)
    pass
