import numpy as np
import matplotlib.pyplot as plt
import time

def iterator(t, nx, ny, ex, ey, hz, curr, dt, dx, dy, ep, mu, tau, D):
    """
    Update one step in time for the fields Ex, Ey, Hz, and current according to Maxwell's Equations.
    Obeys a vertical graphene sheet in the middle of the grid.
    Choices of constants includes ep, mu, delta step sizes, relaxation time, and the Drude constant.

    Parameters
    ----------
    t:    int
          Current timestep
    nx:   int
          Grid x-dimension
    ny:   int
          Grid y-dimension
    ex:   float[][]
          The x-component of the electric field
    ey:   float[][]
          The y-component of the electric field
    hz:   float[][]
          The z-component of the magnetic field
    curr: float[]
          The current along the graphene sheet
    dt:   float
          Delta time step size
    dx:   float
          Delta step size in x-dimension
    dy:   float
          Delta step size in y-dimension
    ep:   float
          Permittivity of environment
    mu:   float
          Permeability of environment
    tau:  float
          Relaxation time
    D:    float
          Drude constant

    Returns
    -------
    ex:   float[][]
    ey:   float[][]
    hz:   float[][]
    curr: float[]
    """

    ex[:, 1:-1] = ex[:, 1:-1] + (dt / (ep * dy)) * (hz[:, 1:] - hz[:, :-1])
    ey[1:-1, :] = ey[1:-1, :] - (dt / (ep * dx)) * (hz[1:, :] - hz[:-1, :])
    hz = hz - (dt / (mu * dx)) * (ey[1:, :] - ey[:-1, :]) + (dt / (mu * dy)) * (ex[:, 1:] - ex[:, :-1])

    # Graphene Sheet Implementation
    curr = (curr + D * dt * ex[:,ny//2 +1]) * tau / (tau + dt)
    hz[:,ny // 2 + 1] = hz[:,ny // 2 - 1] + curr
    ex[:,ny // 2 + 1] = ex[:,ny // 2 - 1]

    # Add source term shifted left
    hz[nx//2][ny//4] += 10e3 * np.sin(2 * 0.03 * np.pi * t)

    return ex, ey, hz, curr

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
    tau = 200
    D = 1e-6

    ex = np.zeros((nx, ny+1))
    ey = np.zeros((nx+1, ny))
    hz = np.zeros((nx, ny))
    curr = np.zeros(nx)

    start = time.time()
    for t in range(nt):
        ex, ey, hz, curr = iterator(t, nx, ny, ex, ey, hz, curr, dt, dx, dy, ep, mu, tau, D)
        plt.imshow(hz, interpolation="lanczos", cmap="gray", vmin=-2e3, vmax=2e3)
        plt.show()
    end = time.time()
    print(end - start)

tester()