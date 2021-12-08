import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from helpers.math import calc_logistic_map
from matplotlib.axes import Axes
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


def plot_phase_portrait(mesh_tuple: np.ndarray, flow_tuple: np.ndarray, ax: Optional[Axes] = None) -> Axes:
    """Plot phase portrait.

    :param mesh_tuple: Underlying 2D mesh
    :type mesh_tuple: np.ndarray
    :param flow_tuple: 2D flow tuple
    :type flow_tuple: np.ndarray
    :param ax: Already existing Axis object to plot on, defaults to None
    :type ax: Optional, optional
    :return: Axis object with additional phase portrait
    :rtype: Axes
    """
    (X1, X2) = mesh_tuple
    (U, V) = flow_tuple

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.streamplot(X1, X2, U, V, density=0.8)
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


def plot_logistic_map_bifurcation(r_min: float, r_max: float, x_initial: float, n_iterations: int, n_discard: int) -> None:
    """Plots the bifurcation diagram of the logistic map function in dependence of the growth rate r

    :param r_min: set start point with lower limit of the growth rate 
    :type r_min: float
    :param r_max: set end point with upper limit of the growth rate 
    :type r_max: float
    :param x_initial: set initial population value
    :type x_initial: float
    :param n_iterations: number of iterations for which the population should be analyzed
    :type n_iterations: int
    :param n_discard: number of discarded point for visualization purposes 
    :type n_discard: int
    Adapted from: https://github.com/Alain1405/bifurcation_diagrams/blob/master/Bifurcation%20diagram.py
    """
    r_values = []
    x_values = []
    
    r_range = np.linspace(r_min, r_max, n_iterations)

    for r in r_range:
        x = x_initial
        
        for i in range(n_iterations+n_discard+1):
            if i >= n_discard:
                r_values.append(r)
                x_values.append(x)
                
            x = calc_logistic_map(r,x)

    #visualize
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes()
    ax.set_title(f"Bifurcation Diagram for $x_{{n + 1}} = rx_{{n}}(1 - x_{{n}})$ for $n_{{iterations}}=${n_iterations},", fontsize=16)
    ax.set_xlabel('r', fontsize=24)
    ax.set_ylabel('$x_{n}$', fontsize=24)
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(0, 1)
    ax.plot(r_values, x_values,',b', alpha=0.5)
    ax.plot(r_values, x_values,',b', alpha=0.5)
    ax.plot(1,0, 'ro', label='Transcritical Bifurcation')
    ax.plot(3,0.665, 'go', label='Period Doubling Bifurcation')
    ax.legend()
    

def plot_logistic_map_orbit(time_stamps: list, r_values: list, *states: np.array) -> None:
    """ Plot the trajectories of multiple logistic map orbits into one plot

    :param time_stamps: List of timestamps at which the orbits are visualized
    :type time_stamps: list
    :param r_values: List of r values used for labeling the corresponding plot
    :type r_values: list
    """
    
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes()

    ax.set_title(f"Orbit for $x_{{n + 1}} = rx_{{n}}(1 - x_{{n}})$ for  $n_{{iterations}}=${100}", fontsize=16)
    ax.set_xlabel('generation n', fontsize=24)
    ax.set_ylabel('$x_{n}$', fontsize=24)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)

    for i, state in enumerate(states):
        ax.plot(time_stamps, state, label=r_values[i])

    ax.legend()


def plot_logistic_map(r_values: list) -> None:
    """Plot logistic map parabola over different growth rate r values

    :param r_values: list of growth rate r values 
    :type r_values: list
    """
    print(r_values[0])
    x = np.linspace(0, 1)
    y = x

    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes()

    ax.set_title(f"Logistic Map over growth rate r", fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('$x_n$', fontsize=24)
    ax.set_ylabel('$x_{n+1}$', fontsize=24)
    
    for i in range(len(r_values)):
        ax.plot(x, calc_logistic_map(r_values[i],x), label=r_values[i])
    
    ax.plot(x, y, '-k', label="y=x")
    ax.legend()
    

def plot_lorenz_attractor(states: np.ndarray, sigma: float, beta: float, rho: float, x0: np.array, T_end: int, line_width: float) -> None:
    """Plots the trajectory of the Lorentz attractor in a 3D Plot 

    :param states: Matrix containing x,y,z states of the attractor for every timestamp
    :type states: np.ndarray
    :param sigma: [description]
    :param sigma: parameter of lorentz attractor (corresponds to Prandtl-number)
    :type sigma: float
    :param beta: parameter of lorentz attractor
    :type beta: float
    :param rho: parameter of lorentz attractor (corresponds to Rayleigh-number)
    :type rho: float
    :param x0: starting point of the trajectory 
    :type x0: np.array
    :param T_end: Timestamp until the trajectory should be plotted (!=number of time points in plot)
    :type T_end: int
    :param line_width: float to controll the line width of the trajectory
    :type line_width: float
    """

    beta = round(beta, 3)
    sigma = round(sigma, 3)
    rho = round(rho, 3)
    
    ax = plt.figure(figsize=(20, 15)).add_subplot(projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], linewidth=line_width, label="Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Lorenz Attractor \n $\sigma$={sigma}, " + r'$\beta$'+ f"={beta}," + r'$\rho$'+ f"={rho}\n $X_{{0}}$={x0} $T_{{end}}$={T_end}", fontsize=16)
    ax.plot(x0[0],x0[1],x0[2], 'ro', label='Initial Point')
    ax.plot(states[:, 0][-1], states[:, 1][-1], states[:, 2][-1], 'ko', label='End Point')
    legend = ax.legend()
    for line in legend.get_lines():
        line.set_linewidth(2.0)
    
    plt.show()


def plot_lorenz_orbit(states: np.ndarray, sigma: float, beta: float, rho: float, t_min, t_max) -> None:
    
    beta = round(beta, 3)
    sigma = round(sigma, 3)
    rho = round(rho, 3)

    T = np.linspace(0,1000,states.shape[0])

    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes()
    
    ax.plot(T,states[:, 0],'r',label='X')
    ax.plot(T,states[:, 1],'g',label='Y')
    ax.plot(T,states[:, 2],'b',label='Z')
    ax.set_xlim(t_min,t_max)
    ax.set_xlabel("time t")
    ax.set_title(f"Lorenz orbits over time \n $\sigma$={sigma}, " + r'$\beta$'+ f"={beta}," + r'$\rho$'+ f"={rho}", fontsize=16)
 
    legend = ax.legend()
    for line in legend.get_lines():
        line.set_linewidth(2.0)
    
    plt.show()


def plot_trajectory_difference(x_t_difference: list, T: np.array) -> None:
    """Plots the distance of two trajectories over time 

    :param x_t_difference: list of distance values per time stamp
    :type x_t_difference: list
    :param T: np.array of time stamps for visualization
    :type T: np.array
    """
    
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes()
    x = np.linalg

    ax.set_title(f"Trajectory Difference", fontsize=16)
    ax.set_xlabel('time t', fontsize=24)
    ax.set_ylabel('$||x(t)-\hat{x}(t)||_2$', fontsize=16)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 60)
    ax.axhline(y=1, color='r', linestyle='-', label="Threshold $||x(t)-\hat{x}(t)||_2= 1$")
    ax.plot(T, x_t_difference, label= "Trajectory Difference")
    legend = ax.legend()