import matplotlib.pyplot as plt
import numpy as np


def plot1dgrid_real(lb, ub, N):
    """Same for the real solution"""
    x1space = np.linspace(lb[0], ub[0], N)
    tspace = np.linspace(lb[1], ub[1], N)
    T, X1 = np.meshgrid(tspace, x1space)

    U = np.sin(np.pi*X1)*np.cos(2*np.pi*T) + 0.5 * \
        np.sin(4*np.pi*X1)*np.cos(8*np.pi*T)

    plt.style.use('dark_background')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(T, X1, c=U, marker='X', vmin=-1, vmax=1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x1$')
    plt.savefig(f'results/real_sol.png')
    plt.close()


lb = [0, 0]
ub = [1, 1]
N = 1000
plot1dgrid_real(lb, ub, N)
