import numpy as np
import matplotlib.pyplot as plt
import time

def iterator(t, nx, ny, ex, ey, hz, dt, dx, dy, ep, mu, lx, ly, sigma_star):
    """
    Update one step in time for the fields Ex, Ey, and Hz according to Maxwell's Equations.
    Choices of constants includes ep, mu, delta step sizes, and layer sizes.
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
    lx: int
        Boundary layer size in x-dimension
    ly: int
        Boundary layer size in y-dimension

    sigma_star: function
                Graded surface electric/magnetic conductivity

    Returns
    -------
    ex: float[][]
    ey: float[][]
    hz: float[][]
    """

    # Update Ex for left layer
    ss = sigma_star(np.arange(ly-1, 0, -1), ly, dy)
    ex[:, 1:ly] = ex[:, 1:ly] * ep * np.reciprocal(ep + ss*dt) + (hz[:, 1:ly] - hz[:, :ly-1]) * dt * np.reciprocal((ep + ss*dt) * dy)

    # Update Ex for middle layer
    ex[:, ly:ny-ly] = ex[:, ly:ny-ly] + (hz[:, ly:ny-ly] - hz[:, ly-1:ny-ly-1]) * (dt / (ep * dy))

    # Update Ex for right layer
    ss = sigma_star(np.arange(0, ly-1), ly, dy)
    ex[:, ny-ly:ny-1] = ex[:, ny-ly:ny-1] * ep * np.reciprocal(ep + ss*dt) + (hz[:, ny-ly:ny-1] - hz[:, ny-ly-1:ny-2]) * dt * np.reciprocal((ep + ss*dt) * dy)

    # Update Ey for top layer
    ss = sigma_star(np.arange(lx-1, 0, -1), lx, dx)
    ey[1:lx, :] = ey[1:lx, :] * ep * np.reciprocal(ep + ss*dt)[:, None] - (hz[1:lx, :] - hz[:lx-1, :]) * dt * np.reciprocal((ep + ss*dt) * dx)[:, None]

    # Update Ey for middle layer
    ey[lx:nx-lx, :] = ey[lx:nx-lx, :] - (hz[lx:nx-lx, :] - hz[lx-1:nx-lx-1, :]) * (dt / (ep * dx))

    # Update Ey for bottom layer
    ss = sigma_star(np.arange(0, lx-1), lx, dx)
    ey[nx-lx:nx-1, :] = ey[nx-lx:nx-1, :] * ep * np.reciprocal(ep + ss*dt)[:, None] - (hz[nx-lx:nx-1, :] - hz[nx-lx-1:nx-2, :]) * dt * np.reciprocal((ep + ss*dt) * dx)[:, None]

    # Create Hzx and Hzy
    hzx = np.multiply(hz, 0.5)
    hzy = np.multiply(hz, 0.5)

    # Update Hzx for top layer
    ss = sigma_star(np.arange(lx-1, -1, -1), lx, dx)
    hzx[:lx, :] = hzx[:lx, :] * mu * np.reciprocal(mu + ss*dt)[:, None] - (ey[1:lx+1, :] - ey[:lx, :]) * dt * np.reciprocal((mu + ss*dt) * dx)[:, None]

    # Update Hzx for bottom layer
    ss = sigma_star(np.arange(0, lx), lx, dx)
    hzx[nx-lx:nx, :] = hzx[nx-lx:nx, :] * mu * np.reciprocal(mu + ss*dt)[:, None] - (ey[nx-lx+1:nx+1, :] - ey[nx-lx:nx, :]) * dt * np.reciprocal((mu + ss*dt) * dx)[:, None]

    # Update Hzy for left layer
    ss = sigma_star(np.arange(ly-1, -1, -1), ly, dy)
    hzy[:, :ly] = hzy[:, :ly] * mu * np.reciprocal(mu + ss*dt) + (ex[:, 1:ly+1] - ex[:, :ly]) * dt * np.reciprocal((mu + ss*dt) * dy)

    # Update Hzy for right layer
    ss = sigma_star(np.arange(0, ly), ly, dy)
    hzy[:, ny-ly:ny] = hzy[:, ny-ly:ny] * mu * np.reciprocal(mu + ss*dt) + (ex[:, ny-ly+1:ny+1] - ex[:, ny-ly:ny]) * dt * np.reciprocal((mu + ss*dt) * dy)

    # Update all layers of Hz
    hz = hzx + hzy

    # Do the rest of the Hz grid
    hz[lx:nx-lx, ly:ny-ly] = hz[lx:nx-lx, ly:ny-ly] + (dt / (mu * dy)) * (ex[lx:nx-lx, ly+1:ny-ly+1] - ex[lx:nx-lx, ly:ny-ly]) - (dt / (mu * dx)) * (ey[lx+1:nx-lx+1, ly:ny-ly] - ey[lx:nx-lx, ly:ny-ly])

    # Add source term
    hz[nx//2][ny//2] += 10e3 * np.sin(2 * 0.03 * np.pi * t) # np.exp(-0.005 * (t * dt - 42)**2)

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

    lx = 10
    ly = 10
    R0 = 1e-6
    m = 3
    sigma_max = 1
    sigma_star = lambda x, layer_thickness, dx: (x / layer_thickness) ** 3 * sigma_max

    ex = np.zeros((nx, ny+1))
    ey = np.zeros((nx+1, ny))
    hz = np.zeros((nx, ny))

    start = time.time()
    for t in range(nt):
        ex, ey, hz = iterator(t, nx, ny, ex, ey, hz, dt, dx, dy, ep, mu, lx, ly, sigma_star)
        print(t)
        if t >= 200:
            plt.imshow(hz, interpolation="lanczos", cmap="gray", vmin=-2e3, vmax=2e3)
            plt.show()
    end = time.time()
    print(end - start)

tester()