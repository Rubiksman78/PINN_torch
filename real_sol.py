import matplotlib.pyplot as plt
import numpy as np
import torch


def sol(x, t):
    u = torch.sin(np.pi*x)*torch.cos(2*np.pi*t) + 0.5 * \
        torch.sin(4*np.pi*x)*torch.cos(8*np.pi*t)
    return u


def real_sol(lb, ub, N):
    x1space = np.linspace(lb[0], ub[0], N)
    tspace = np.linspace(lb[1], ub[1], N)
    T, X1 = np.meshgrid(tspace, x1space)

    U = sol(X1, T)

    plt.style.use('dark_background')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(T, X1, c=U, marker='X', vmin=-1, vmax=1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x1$')
    plt.savefig(f'results/real_sol.png')
    plt.close()
    return U
