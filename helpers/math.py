import numpy as np

# TODO Add module description + optional license


def get_square_matrix_from_eigen(eigenvalues: list, eigenvectors: list[list], eps=1e-13) -> np.ndarray:
    """Given set of N eigenvalues with corresponding N eigenvectors, return NxN square matrix.
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
