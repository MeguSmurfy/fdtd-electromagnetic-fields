import numpy as np
import matplotlib.pyplot as plt
import time

def iterator(t, nx, ny, ex, ey, hz, dt, dx, dy, ep, mu):
    """
    Update one step in time for the fields Ex, Ey, and Hz according to Maxwell's Equations.
    Choices of constants includes ep and mu, as well as delta step sizes.
    These equations are completely described by these constants.

    Parameters
    ----------
    t:  int
        Current timestep
    nx: int
        Grid x-dimension
    ny: int
        Grid y-dimension
    ex: float[][]
        The x-component of the electric field
    ey: float[][]
        The y-component of the electric field
    hz: float[][]
        The z-component of the magnetic field
    dt: float
        Delta time step size
    dx: float
        Delta step size in x-dimension
    dy: float
        Delta step size in y-dimension
    ep: float
        Permittivity of environment
    mu: float
        Permeability of environment

    Returns
    -------
    ex: float[][]
    ey: float[][]
    hz: float[][]
    """

    ex[:, 1:-1] = ex[:, 1:-1] + (dt / (ep * dy)) * (hz[:, 1:] - hz[:, :-1])
    ey[1:-1, :] = ey[1:-1, :] - (dt / (ep * dx)) * (hz[1:, :] - hz[:-1, :])
    hz = hz - (dt / (mu * dx)) * (ey[1:, :] - ey[:-1, :]) + (dt / (mu * dy)) * (ex[:, 1:] - ex[:, :-1])

    hz[nx//2][ny//2] += 10e3 * np.sin(2 * 0.03 * np.pi * t) # Source Term
    return ex, ey, hz

def tester():
    """
    Main function for running the simulation.
    """

    nx = 200
    ny = 200
    nt = 1000
    ep = 8.854e-12
    mu = 1.256e-6
    c = np.sqrt(1/(ep*mu))

    dx = 0.001
    dy = 0.001
    dt = 0.5 * dx/(c) # Courant's Law

    ex = np.zeros((nx, ny+1))
    ey = np.zeros((nx+1, ny))
    hz = np.zeros((nx, ny))

    start = time.time()
    for t in range(nt):
        ex, ey, hz = iterator(t, nx, ny, ex, ey, hz, dt, dx, dy, ep, mu)
        plt.imshow(hz, interpolation="lanczos", cmap="gray", vmin=-2e3, vmax=2e3)
        plt.show()
    end = time.time()
    print(end - start)

tester()