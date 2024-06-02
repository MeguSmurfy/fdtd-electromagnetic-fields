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

    # Update Ex
    for i in range(nx):
        for j in range(1, ny-1):
            ex[i][j] = ex[i][j] + (dt / (ep * dy)) * (hz[i][j] - hz[i][j-1])
    # Update Ey
    for i in range(1, nx-1):
        for j in range(ny):
            ey[i][j] = ey[i][j] - (dt / (ep * dx)) * (hz[i][j] - hz[i-1][j])
    # Update Hz
    for i in range(nx):
        for j in range(ny):
            hz[i][j] = hz[i][j] + (dt / (mu * dy)) * (ex[i][j+1] - ex[i][j]) - (dt / (mu * dx)) * (ey[i+1][j] - ey[i][j])

    # Add source term
    hz[nx//2][ny//2] += 10e3 * np.sin(2 * 0.03 * np.pi * t)

    return ex, ey, hz

def tester():
    """
    Main function for running the simulation.
    """

    nx = 300
    ny = 300
    nt = 800
    ep = 8.854e-12
    mu = 1.256e-6
    c = np.sqrt(1/(ep*mu))

    dx = 0.001
    dy = 0.001
    dt = 0.5*dx/(c) # Courant's Law

    ex = np.zeros((nx, ny+1))
    ey = np.zeros((nx+1, ny))
    hz = np.zeros((nx, ny))

    start = time.time()
    for t in range(nt):
        ex, ey, hz = iterator(t, nx, ny, ex, ey, hz, dt, dx, dy, ep, mu)
        print(t)
        if t >= 150:
          plt.imshow(hz, interpolation="lanczos", cmap="gray", vmin=-2e3, vmax=2e3)
          plt.show()
    end = time.time()
    print(end - start)

tester()