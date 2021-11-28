import numpy as np

from typing import Callable

# TODO Add module description + optional license


def get_diagonalizable_square_matrix_from_eigen(eigenvalues: list, eigenvectors: list[list], eps=1e-13) -> np.ndarray:
    """Given set of N eigenvalues with corresponding N eigenvectors, return NxN diagonalizable square matrix.
    Uses decomposition of a diagonalizable matrix (https://en.wikipedia.org/wiki/Diagonalizable_matrix).

    :param eigenvalues: List of N eigenvalues
    :type eigenvalues: list
    :param eigenvectors: List of N eigenvectors where eigenvectors are vectors in N
    :type eigenvectors: list[list]
    :param eps: Replace real and imag values with absolute value smaller than this to 0, defaults to 1e-13
    :type eps: [type], optional
    :return: NxN square matrix with matching eigenvalues and eigenvectors
    :rtype: np.ndarray
    """
    D = np.diag(eigenvalues)
    P = np.array(eigenvectors).T
    A = P @ D @ np.linalg.inv(P)

    if (A.real).any():
        A.real[abs(A.real) < eps] = 0
    if (A.imag).any():
        A.imag[abs(A.imag) < eps] = 0
    return A


def calc_flows(A: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Given a 2D matrix A, calculate flow fields A*x.

    :param A: 2D matrix A
    :type A: np.ndarray
    :return: Tuple of underlying meshgrid and resulting flow vectors
    :rtype: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    """
    x1 = np.arange(-1, 1.0, 0.1)
    x2 = np.arange(-1, 1.0, 0.1)
    X1, X2 = np.meshgrid(x1, x2)
    mesh_tuple = (X1, X2)

    U = A[0, 0] * X1 + A[0, 1] * X2
    V = A[1, 0] * X1 + A[1, 1] * X2
    flow_tuple = (U, V)

    return mesh_tuple, flow_tuple


# TODO description of class
class XDotPoly:
    def __init__(self, polynomial: np.ndarray):
        """Initialize a 1D DGL polynomial in the state x.

        :param polynomial: Rank-1 array of polynomial coefficients.
            If length of the p is is n+1 then the polynomial is described as
            p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
        :type polynomial: np.ndarray
        """
        self.polynomial = polynomial
        self.degree = len(polynomial) - 1

    def get_steady_states(self) -> list:
        """Return the roots of the polynomial DGL which is equal to the states of zero change, i.e. the steady states.

        :return: List of states where polynomial is equal to zero
        :rtype: list
        """
        return list(np.roots(self.polynomial))

    def evaluate(self, x: float) -> float:
        """Evaluate the polynomial at a given state. Equals the change in state.

        :param x: State to evaluate at
        :type x: float
        :return: Change in state at given state
        :rtype: float
        """
        out = 0
        for i in range(self.degree + 1):
            out += self.polynomial[i] * x ** (self.degree - i)
        return out

    def check_stability(self, x: float) -> str:
        """Check the stability at a given steady point.

        List of possibilities:
        - positive change on the left and positive change on the right -> "instable"
        - negative change on the left and negative change on the right -> "instable"
        - negative change on the left and positive change on the right -> "instable (repulsive)"
        - positive change on the left and negative change on the right -> "stable (attractive)"

        :param x: State value at which to check the stability
        :type x: float
        :return: Categorical string indicating the stability
        :rtype: str
        """
        left_x = x - 0.0001
        right_x = x + 0.0001
        left_val = self.evaluate(left_x)
        right_val = self.evaluate(right_x)

        if left_val * right_val > 0:
            return "instable"
        if left_val < 0 and right_val > 0:
            return "instable (repulsive)"
        if left_val > 0 and right_val < 0:
            return "stable (attractive)"
        return "error"

    @staticmethod
    def get_stability_for_parametric(
        f: Callable, param_min: float, param_max: float, num: int = 100
    ) -> tuple[list[float], list[float], list[str]]:
        """Investigates the stability of a polynomial DGL within a given range of parameters.
        Only real steady states are considered (zero imaginary part).

        :param f: Callable that takes in one parameter and returns a polynomial vector representation
        :type f: Callable
        :param param_min: Minimum parameter value to investigate stability at
        :type param_min: float
        :param param_max: Maximum parameter value to investigate stability at
        :type param_max: float
        :param num: Number of steps between param_min and param_max to investigate, defaults to 100
        :type num: int, optional
        :return: Tuple of:
            a) List of params that were checked
            b) List of steady state for this parameter configuration
            c) List strings indicating the stability at the tested configurations
            e.g
            a) [0, 1, 1, 2, 2]
            b) [0, -1, 1, -2, 2]
            c) ["instable", "instable", "stable", "instable", "stable"]
        :rtype: tuple[list[float], list[float], list[str]]
        """
        param_vec = np.linspace(param_min, param_max, num + 1)

        params = []
        steady_states = []
        stability = []

        for param in param_vec:
            p = XDotPoly(f(param))
            current_steady_states = p.get_steady_states()
            for steady_state in current_steady_states:
                if np.imag(steady_state) != 0:
                    continue
                params.append(param)
                steady_states.append(steady_state)
                stability.append(p.check_stability(steady_state))

        return (params, steady_states, stability)
