import numpy as np
from scipy.integrate import odeint
from typing import Callable, List

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

def calc_logistic_map(r: float, x: float) -> float:
    """Calculate logistical growth for next time stamp

    :param r: growth/decay rate of the population
    :type r: float
    :param x: population at previous time step n
    :type x: float
    :return: population at previous time step n + 1 
    :rtype: float
    """
    return r*x*(1-x)


def calc_logistic_map_orbit(r: float, n_iterations: int) -> tuple[list, np.ndarray, float]:
    """Calculate the trajectory of the logistic map orbit  

    :param r: growth/decay rate of the population 
    :type r: float
    :param n_iterations: number of iterations for which the orbit states should be calculated 
    :type n_iterations: int
    :return: List/Numpy array of state and timepoints of the orbit trajectory and corresponding growth rate r
    :rtype: tuple[list, np.ndarray, float]
    """
    
    X=[]
    x = 0.5

    for _ in range(n_iterations):
        X.append(x)
        x = calc_logistic_map(r,x)

    T = np.linspace(0,100,len(X))
    return X, T, r


def calc_lorenz(state: np.ndarray, t: float, sigma: float, beta:float, rho:float) -> tuple[float, float, float]:
   """Evaluates the non-linear Lorentz equation ODEs

   :param state: current state of ode solver
   :type state: np.array
   :param t: current time stamp of ode solver
   :type t: float
   :param sigma: parameter of lorentz attractor (corresponds to Prandtl-number)
   :type sigma: float
   :param beta: parameter of lorentz attractor
   :type beta: float
   :param rho: parameter of lorentz attractor (corresponds to Rayleigh-number)
   :type rho: float
   :return: time derivatives of the current time stamp
   :rtype: tuple[float, float, float]
   """
   x, y, z  = state

   x_dot = sigma*(y - x)
   y_dot = rho*x - y - x*z
   z_dot = x*y - beta*z
   return x_dot, y_dot, z_dot


def calc_lorenz_attractor(sigma: float, beta:float, rho:float, x0: np.ndarray, T_end: int) -> np.ndarray:
    """Calculates a trajectory of the Lorentz attractor until a fixed end point for a initial value and a given parameter set  

    :param sigma: parameter of lorentz attractor (corresponds to Prandtl-number)
    :type sigma: float
    :param beta: parameter of lorentz attractor
    :type beta: float
    :param rho: parameter of lorentz attractor (corresponds to Rayleigh-number)
    :type rho: float
    :param x0: starting point of the trajectory 
    :type x0: np.array
    :param T_end: Timestamp until the trajectory should be calculated (!=number of time points in plot)
    :type T_end: int
    :return: matrix of all states from start till endpoint of trajectory
    :rtype: np.ndarray
    """

    t = np.arange(0.0, T_end, 0.01)
    states = odeint(calc_lorenz,x0, t, args=(sigma,beta,rho))
    return states 


def calc_trajectory_difference(x_t1: np.ndarray, x_t2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the euclidean distance between all trajectory points 

    :param x_t1: first trajectory with x,y,z states for t time points 
    :type x_t1: np.ndarray
    :param x_t2: second trajectory with x,y,z states for t time points 
    :type x_t2: np.ndarray
    :return: pointwise distance list where each entry corresponds to euclidean distance between the trajectories (t timepoints entries)
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    distance = np.zeros(x_t1.shape[0])
    distance = np.linalg.norm(x_t1-x_t2, ord=2, axis=1)

    T = np.linspace(0,1000,x_t1.shape[0])
    
    return distance, T


def calc_trajectory_passing_threshold(x_t_difference: np.ndarray, threshold: float, T_end: float) -> None:
    """Calculates the simulated time stamp where the difference between two trajectories passes a predefined threshold

    :param x_t_difference: list of distance values per time stamp
    :type x_t_difference: np.ndarray
    :param threshold: threshold which trajectory difference has to reach 
    :type threshold: float
    :param T_end: corresponds to simulated time ans is used to iteration counter into simulated seconds
    :type threshold: float
    """
    
    idx_geq_threshold = np.where(x_t_difference>=threshold)

    try:  
        first_idx = np.amin(idx_geq_threshold)
        t_first_geq_threshold = first_idx*T_end*(1/x_t_difference.shape[0])
        print(f'The threshold = {threshold} was reached after {t_first_geq_threshold} simulated seconds (or in iteration step n={first_idx})')
    
    #raised for zero length (=threshold never passed)
    except ValueError:
        print(f'The threshold = {threshold} was never reached in T={T_end}')  


# Class for generic polynomial evaluation function.
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
        - positive change on the left and positive change on the right -> "instable (undefined)"
        - negative change on the left and negative change on the right -> "instable (undefined)"
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
            return "instable (undefined)"
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


# Class for Andronov-Hopf bifurcation analysis of a 2D DGL.
class AndronovHopf:
    @staticmethod
    def calc_flows(y: tuple[float, float], alpha: float) -> tuple[float, float]:
        """Calculate the flow field for the Andronov-Hopf equation.

        :param y: State tuple (x0, x1)
        :type mesh_tuple: tuple[float, float]
        :param alpha: parameter alpha value
        :type alpha: float
        :return: Tuple of resulting partial derivatives (dx1, dx2)
        :rtype: tuple[float, float]
        """
        x1, x2 = y
        dx1 = alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2)
        dx2 = x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)
        return dx1, dx2

    @staticmethod
    def solve(t: np.ndarray, y0: tuple[float, float], alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Solve the DGL for the given initial conditions and parameter alpha.

        :param t: time vector for which to calculate the solution
        :type t: np.ndarray
        :param y0: initial condition (x0, x1)
        :type y0: tuple[float, float]
        :param alpha: parameter alpha value
        :type alpha: float
        :return: tuple of states (x1, x2) that are the solution of the DGL
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        deriv = lambda y, t, alpha: AndronovHopf.calc_flows(y, alpha)
        ret = odeint(deriv, y0, t, args=(alpha,))
        x1, x2 = ret.T
        return x1, x2
