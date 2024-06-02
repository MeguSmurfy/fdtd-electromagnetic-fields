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

    # Update Ex with PML left-to-right
    for i in range(nx):
        for j in range(ly-1, 0, -1):
            ss = sigma_star(ly-j, ly, dy)
            ex[i][j] = ex[i][j] * ep / (ep + ss*dt) + (hz[i][j] - hz[i][j-1]) * (dt / ((ep + ss*dt) * dy))

    for i in range(nx):
        for j in range(ly, ny - ly):
            ex[i][j] = ex[i][j] + (hz[i][j] - hz[i][j-1]) * (dt / (ep * dy))

    for i in range(nx):
        for j in range(ny - ly, ny - 1):
            ss = sigma_star(j-(ny-ly), ly, dy)
            ex[i][j] = ex[i][j] * ep / (ep + ss*dt) + (hz[i][j] - hz[i][j-1]) * (dt / ((ep + ss*dt) * dy))


    # Update Ey with PML top-to-bottom
    for i in range(lx-1, 0, -1):
        for j in range(ny):
            ss = sigma_star(lx-i, lx, dx)
            ey[i][j] = ey[i][j] * ep / (ep + ss*dt) - (hz[i][j] - hz[i-1][j]) * (dt / ((ep + ss*dt) * dx))
    for i in range(lx, nx - lx):
        for j in range(ny):
            ey[i][j] = ey[i][j] - (dt / (ep * dx)) * (hz[i][j] - hz[i-1][j])
    for i in range(nx - lx, nx - 1):
        for j in range(ny):
            ss = sigma_star(i-(nx-lx), lx, dx)
            ey[i][j] = ey[i][j] * ep / (ep + ss*dt) - (hz[i][j] - hz[i-1][j]) * (dt / ((ep + ss*dt) * dx))

    # Create Hzx and Hzy
    hzx = np.multiply(hz, 0.5)
    hzy = np.multiply(hz, 0.5)

    # Update Hzx for top layer
    for i in range(lx-1, -1, -1):
        for j in range(ny):
            ss = sigma_star(lx-i, lx, dx)
            hzx[i][j] = hzx[i][j] * mu / (mu + ss*dt) - (ey[i+1][j] - ey[i][j]) * (dt / ((mu + ss*dt) * dx))

    # Update Hzx for bottom layer
    for i in range(nx-lx, nx):
        for j in range(ny):
            ss = sigma_star(i-(nx-lx), lx, dx)
            hzx[i][j] = hzx[i][j] * mu / (mu + ss*dt) - (ey[i+1][j] - ey[i][j]) * (dt / ((mu + ss*dt) * dx))

    # Update Hzy for left layer
    for i in range(nx):
        for j in range(ly-1, -1, -1):
            ss = sigma_star(ly-j, ly, dy)
            hzy[i][j] = hzy[i][j] * mu / (mu + ss*dt) + (ex[i][j+1] - ex[i][j]) * (dt / ((mu + ss*dt) * dy))

    # Update Hzy for right layer
    for i in range(nx):
        for j in range(ny-ly, ny):
            ss = sigma_star(j-(ny-ly), ly, dy)
            hzy[i][j] = hzy[i][j] * mu / (mu + ss*dt) + (ex[i][j+1] - ex[i][j]) * (dt / ((mu + ss*dt) * dy))

    # Update all layers of Hz
    hz = hzx + hzy

    # Do the rest of the Hz grid
    for i in range(lx, nx - lx):
        for j in range(ly, ny - ly):
            hz[i][j] = hz[i][j] + (dt / (mu * dy)) * (ex[i][j+1] - ex[i][j]) - (dt / (mu * dx)) * (ey[i+1][j] - ey[i][j])

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
    sigma_star = lambda x, layer_thickness, dx: x * sigma_max / layer_thickness # x**3 * np.log(R0) * (m+1) / (2 * (layer_thickness**4) * dx)

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