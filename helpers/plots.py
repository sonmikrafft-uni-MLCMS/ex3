import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional

# TODO Add module description + optional license


def plot_eigenvalues(eigenvalues: np.ndarray, ax: Optional[Axes] = None) -> Axes:
    """Plot eigenvalues in 2D imaginary plane.

    :param eigenvalues: Numpy array of eigenvalues
    :type eigenvalues: np.ndarray
    :param ax: Already existing Axis object to plot on, defaults to None
    :type ax: Optional[Axes], optional
    :return: Axis object with additional scatter plot of eigenvalues
    :rtype: Axes
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.scatter(eigenvalues.real, eigenvalues.imag)
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.set_xlabel("Re()")
    ax.set_ylabel("Im()")

    return ax


def plot_phase_portrait(
    mesh_tuple: np.ndarray,
    flow_tuple: np.ndarray,
    density: Optional[float] = 1.0,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    """Plot phase portrait.

    :param mesh_tuple: Underlying 2D mesh
    :type mesh_tuple: np.ndarray
    :param flow_tuple: 2D flow tuple
    :type flow_tuple: np.ndarray
    :param density: Density of the points, defaults to 1.0
    :type density: Optional[float], optional
    :param ax: Already existing Axis object to plot on, defaults to None
    :type ax: Optional, optional
    :return: Axis object with additional phase portrait
    :rtype: Axes
    """
    (X1, X2) = mesh_tuple
    (U, V) = flow_tuple

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.streamplot(X1, X2, U, V, density=density, **kwargs)
    ax.set_aspect("equal")
    ax.set_xlim([X1[0, 0], X1[-1, -1]])
    ax.set_ylim([X2[0, 0], X2[-1, -1]])

    return ax


def plot_bifurcation(
    parameters: list[float], steady_states: list[float], stabilities: list[str], ax: Optional[Axes] = None
) -> Axes:
    """Plot bifurcation diagram as a scatter plot of the steady states over the parameters where the points are colored
    by their stability value.

    example:
    >>> parameters = [0, 1, 1, 2, 2]
    >>> steady_states = [0, -1, 1, -2, 2]
    >>> stabilities = ["instable", "instable", "stable", "instable", "stable"]
    >>> fig, ax = plt.subplots(1,1)
    >>> ax = plot_bifurcation(parameters, steady_states, stabilities, ax=ax)
    >>> ...
    >>> plt.plot()

    :param parameters: List of parameters of length N
    :type parameters: list[float]
    :param steady_states: List of steady states of length N
    :type steady_states: list[float]
    :param stabilities: List of categorical stabilities of length N
    :type stabilities: list[str]
    :param ax: Already existing Axis object to plot on, defaults to None, defaults to None
    :type ax: Optional[Axes], optional
    :return: Axis object with additional bifurcation diagram
    :rtype: Axes
    """
    df = pd.DataFrame({"parameter": parameters, "steady_state": steady_states, "stability": stabilities})

    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax = sns.scatterplot(data=df, x="parameter", y="steady_state", hue="stability", ax=ax)
    ax.set_xlim([min(parameters), max(parameters)])

    return ax


def plot_orbit(x1: list[float], x2: list[float], ax: Optional[Axes] = None) -> None:
    """Plot orbit of a system.

    :param x1: First coordinate of the orbit
    :type x1: list[float]
    :param x2: Second coordinate of the orbit
    :type x2: list[float]
    :param ax: Already existing Axis object to plot on, defaults to None
    :type ax: Optional[Axes], optional
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.plot(x1, x2, label="Orbit")
    ax.plot(x1[0], x2[0], "ro", label="Initial point")
    return ax


def plot_cusp(
    x_min_max: tuple[float, float], alpha2_min_max: tuple[float, float], elev: int, azim: int
) -> tuple[Figure, Axes]:
    """
    Visualizes the cusp bifurcation

    :param x_min_max: tuple of lower and upper bound for the x values
    :type x_min_max: tuple[float, float]
    :param alpha2_min_max: tuple of lower and upper bound for the alpha values
    :type alpha2_min_max: tuple[float, float]
    :param elev: for rotation of the visualization (elevation of the axes), ‘elev’ stores the elevation angle in the z
        plane
    :type elev: int
    :param azim: for rotation of the visualization (azimuth of the axes), ‘azim’ stores the azimuth angle in the x,y
        plane
    :type azim: int
    :return: Figure and Axes object
    :return: Figure, Axes
    """
    x = np.arange(*x_min_max, 0.01)
    alpha2 = np.arange(*alpha2_min_max, 0.01)
    X, A2 = np.meshgrid(x, alpha2)
    A1 = X ** 3 - A2 * X

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(A1, A2, X, cmap=cm.viridis)
    ax.set_title("Surface plot of the cusp bifurcation")
    ax.set_xlabel(r"$\alpha_1$")
    ax.set_ylabel(r"$\alpha_2$")
    ax.set_zlabel(r"$x$")
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()

    return fig, ax
