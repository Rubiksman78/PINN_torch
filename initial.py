import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from equation import *

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)


def domain_boundaries(tmin, tmax, xmin, xmax):
    """
    Input: desired limits for the domain
    Output: domain boundaries in tensors
    """
    lb, ub = tf.constant([tmin, xmin], dtype=DTYPE), tf.constant(
        [tmax, xmax], dtype=DTYPE)  # Frontières basse et haute pour toute variable
    return lb, ub


def set_training_data(tmin, tmax, xmin, xmax, dimension, N_0, N_b, N_r, speed=True):
    """
    Input: -tmin,tmax,xmin,xmax = desired limits domain
           -dimension = spatial dimension domain
           -N_0,N_b,N_r = number of training points
           -speed = if initial speed condition is used
           """
    lb, ub = domain_boundaries(tmin, tmax, xmin, xmax)

    # Initial conditions
    t_0 = tf.ones((N_0, 1), dtype=DTYPE)*lb[0]  # lower time boundary
    # uniform distribution initial points
    x_0 = tf.random.uniform((N_0, dimension), lb[1], ub[1], dtype=DTYPE)
    X_0 = t_0
    for i in range(dimension):
        # getting X0 = (t0,x01,...,x0k)
        X_0 = tf.concat([X_0, tf.expand_dims(x_0[:, i], axis=-1)], axis=1)
    u_0 = u0(t_0, x_0)  # see equation.py for desired condition

    # Initial_speed
    v_0 = v0(t_0, x_0, dimension)  # see equation.py for desired condition

    # Boundary conditions
    # time uniformly sampled
    t_b = tf.random.uniform((N_b, 1), lb[0], ub[0], dtype=DTYPE)
    x_b = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((N_b,
                                                                       dimension), 0.5, dtype=DTYPE)  # x_b in either lower or upper bound
    X_b = t_b
    for i in range(dimension):
        X_b = tf.concat([X_b, tf.expand_dims(x_b[:, i], axis=-1)], axis=1)
    u_b = u_bound(t_b, x_b, dimension)  # see equation.py for desired condition

    # Residual of the equation
    # uniform t and x for residual
    t_r = tf.random.uniform((N_r, 1), lb[0], ub[0], dtype=DTYPE)
    x_r = tf.random.uniform((N_r, dimension), lb[1], ub[1], dtype=DTYPE)
    X_r = t_r
    for i in range(dimension):
        X_r = tf.concat([X_r, tf.expand_dims(x_r[:, i], axis=-1)], axis=1)

    # Training data
    if speed:
        X_data = [X_0, X_b, X_0]
        u_data = [u_0, u_b, v_0]
    else:
        X_data = [X_0, X_b]
        u_data = [u_0, u_b]
    time_x = [t_0, t_b, t_r, x_0, x_b, x_r, u_0, u_b]
    return X_data, u_data, time_x, X_r


def plot_training_points(dimension, time_x):
    """
    Input: -dimension = spatial dimension
           -time_x = [t_0,t_b,t_r,x_0,x_b,x_r,u_0,u_b]
    Output: display training points in either 1,2 or 3D
    """
    t_0, t_b, t_r, x_0, x_b, x_r, u_0, u_b = time_x
    if dimension == 1:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.scatter(t_0, x_0[:, 0], c=u_0, marker='X', vmin=-1, vmax=1)
        ax.scatter(t_b, x_b[:, 0], c=u_b, marker='X', vmin=-1, vmax=1)
        ax.scatter(t_r, x_r[:, 0], c='r', marker='.', alpha=0.1)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x1$')
    if dimension == 2:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(t_0, x_0[:, 0], x_0[:, 1], c=u_0[:, 0],
                   marker='X', vmin=-1, vmax=1)
        ax.scatter(t_b, x_b[:, 0], x_b[:, 1], c=u_b[:, 0],
                   marker='X', vmin=-1, vmax=1)
        ax.scatter(t_r, x_r[:, 0], x_r[:, 1], c='r', marker='.', alpha=0.1)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x1$')
        ax.set_zlabel('$x2$')
    if dimension == 3:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(t_0, x_0[:, 0], x_0[:, 1], c=u_0[:, 0],
                   marker='X', vmin=-1, vmax=1)
        ax.scatter(t_b, x_b[:, 0], x_b[:, 1], c=u_b[:, 0],
                   marker='X', vmin=-1, vmax=1)
        ax.scatter(t_r, x_r[:, 0], x_r[:, 1], c='r', marker='.', alpha=0.1)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x1$')
        ax.set_zlabel('$x2$')

    ax.set_title('Positions of collocation points and boundary data')
    plt.show()
