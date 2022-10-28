import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch


def real_sol(x, t):
    u = np.sin(np.pi*x)*np.cos(2*np.pi*t) + 0.5 * \
        np.sin(4*np.pi*x)*np.cos(8*np.pi*t)
    return u


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

    U = real_sol(X1, T)

    plt.style.use('dark_background')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(T, X1, U, c=U, marker='X', vmin=-1, vmax=1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x1$')
    plt.savefig(f'results/real_sol3D.png')
    plt.close()


lb = [0, 0]
ub = [1, 1]
N = 100
plot_real_sol3D(lb, ub, N)
