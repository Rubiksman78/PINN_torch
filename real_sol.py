import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
from math import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def real_sol(x, t):
    u = torch.sin(np.pi*x)*torch.cos(2*np.pi*t) + 0.5 * \
        torch.sin(4*np.pi*x)*torch.cos(8*np.pi*t)
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


lb = [0, 0]
ub = [1, 1]
N = 100
plot_real_sol3D(lb, ub, N)
