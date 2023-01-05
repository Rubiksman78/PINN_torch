import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
from math import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def real_sol(x, y, t):
    # The following code sample describes solving the 2D wave equation.
    # This code will look at a 2D sine wave under initial conditions.
    # There are a few different steps for doing this.

    # STEP 2.  Set up the position and time grids (or axes).
    # Set up the position information.
    axis_size = 100  # Size of the 1D grid.
    side_length = 1  # Length of one side of one of the wave plot axes.
    dx, dy = side_length/axis_size, side_length/axis_size  # Space step
    axis_points = np.linspace(0, side_length, axis_size)  # Spatial grid points
    c = 1/np.sqrt(2)  # Constant chosen in the 2D wave equation.

    # Set up the time grid to calcuate the equation.
    T = 20  # Total time (s)
    # Time step size to ensure a stable discretization scheme.
    dt = 0.5*(1/c) * (1/np.sqrt(dx**(-2) + dy**(-2)))
    n = int(T/dt)  # Total number of time steps.

    # STEP 3.  Initialization condition function for the 2D wave equation.  2D sine wave pattern in this example.
    def initial_cond(x, y):
        return np.sin(2*np.pi*x + 2*np.pi*y)

    # Create a meshgrid for the 3D function of initial wave.
    X, Y = np.meshgrid(axis_points, axis_points)

    # Calculate the first initial condition using the initialization function.  This is the initial
    # wave state.
    U = initial_cond(X, Y)

    # Assign initial boundary conditions to their own variables.
    B1 = U[:, 0]
    B2 = U[:, -1]
    B3 = U[0, :]
    B4 = U[-1, :]

    # Set up matrix for the 2nd initial condition.
    U1 = np.zeros((axis_size, axis_size))

    # Calculate the 2nd initial condition needed for time iteration.
    U1[1:-1, 1:-1] = (U[1:-1, 1:-1] + (c**2/2)*(dt**2/dx**2)*(U[1:-1, 0:-2] - 2*U[1:-1, 1:-1] + U[1:-1, 2:]) +
                      (c**2/2)*(dt**2/dy**2)*(U[0:-2, 1:-1] - 2*U[1:-1, 1:-1] + U[2:, 1:-1]))

    # Reinforce the boundary conditions on the surface after the 2nd initial condition.
    U1[:, 0] = B1
    U1[:, -1] = B2
    U1[0, :] = B3
    U1[-1, :] = B4

    # Assign these initial boundary conditions to their own variables.
    B5 = U1[:, 0]
    B6 = U1[:, -1]
    B7 = U1[0, :]
    B8 = U1[-1, :]

    # STEP 4.  Solve the PDE for a result of all spatial positions after
    # time T has elapsed.
    # Create a leading array to update the wave at every time step.  Initialize it with zeros.
    U2 = np.zeros((axis_size, axis_size))

    # Create an initialized array to store all the wave amplitude map images for each time point.
    map_array = np.zeros((axis_size, axis_size, n))

    # Initialize the first two slices of the array with the two initial wave maps.
    map_array[:, :, 0] = U
    map_array[:, :, 1] = U1

    # Numerically solve the PDE by iteration over the specified total time.
    for i in range(2, n):

        U2[1:-1, 1:-1] = (2*U1[1:-1, 1:-1] - U[1:-1, 1:-1] + (c**2)*((dt/dx)**2)*(U1[1:-1, 0:-2] - 2*U1[1:-1, 1:-1] +
                                                                                  U1[1:-1, 2:]) + (c**2)*((dt/dy)**2)*(U1[0:-2, 1:-1] - 2*U1[1:-1, 1:-1] +
                                                                                                                       U1[2:, 1:-1]))

        # Direchlet boundary conditions for the wave.
        U2[:, 0] = B5
        U2[:, -1] = B6
        U2[0, :] = B7
        U2[-1, :] = B8

        U1[:, 0] = B5
        U1[:, -1] = B6
        U1[0, :] = B7
        U1[-1, :] = B8

        U[:, 0] = B1
        U[:, -1] = B2
        U[0, :] = B3
        U[-1, :] = B4

        # Update the wave array with the 2D wave data.
        map_array[:, :, i] = U2

        # Update the trailing wave maps with the leading ones to prepare them for the next time loop iteration.
        U = U1
        U1 = U2

        return map_array

    def plot_real_sol(lb, ub, N):
        x1space = np.linspace(lb[0], ub[0], N)
        tspace = np.linspace(lb[1], ub[1], N)
        T, X1 = np.meshgrid(tspace, x1space)

        U = real_sol(X1, T)

    plt.style.use('dark_background')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(T, X1, c=U, marker='X', vmin=-1, vmax=1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x1$')
    plt.savefig(f'results/real_sol.png')
    plt.close()


def plot_real_sol3D(lb, ub, N):
    x1space = np.linspace(lb[0], ub[0], N)
    tspace = np.linspace(lb[1], ub[1], N)
    T, X1 = np.meshgrid(tspace, x1space)
    T = torch.from_numpy(T).view(1, N*N, 1).to(device).float()
    X1 = torch.from_numpy(X1).view(1, N*N, 1).to(device).float()
    U = real_sol(X1, T)
    U = torch.squeeze(U).detach().cpu().numpy()
    T, X1 = T.view(N, N).detach().cpu().numpy(), X1.view(
        N, N).detach().cpu().numpy()
    plt.style.use('dark_background')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(T, X1, U, c=U, marker='X', vmin=-1, vmax=1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x1$')
    plt.savefig(f'results/real_sol3D.png')
    plt.close()


# lb = [0, 0]
# ub = [1, 1]
# N = 100
# plot_real_sol3D(lb, ub, N)
