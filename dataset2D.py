import torch
import numpy as np
from config import DEFAULT_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_i = DEFAULT_CONFIG["N_i"]
########################################################### POINTS DEFINITION ###########################################################
#########################################################################################################################################


def define_points(N_i, N_b, N_r, l_b, u_b):
    t_i = torch.zeros(N_i, 1)
    x_i = torch.linspace(l_b, u_b, N_i).view(N_i, 1)
    y_i = torch.linspace(l_b, u_b, N_i).view(N_i, 1)
    #u_i = torch.sin(np.pi*x_i) + 0.5*torch.sin(4*np.pi*x_i)
    u_i = torch.zeros(N_i, 1)

    t_b = torch.linspace(l_b, u_b, N_b).view(N_b, 1)

    x_b = torch.bernoulli(0.5*torch.ones(N_b, 1))
    y_b = torch.bernoulli(0.5*torch.ones(N_b, 1))
    u_b = torch.zeros(N_b, 1)

    t_r = torch.rand(N_r, 1)
    x_r = torch.rand(N_r, 1)
    y_r = torch.rand(N_r, 1)
    return t_i, x_i, y_i, u_i, t_b, x_b, y_b, u_b, t_r, x_r, y_r


def define_points_begin(N_ri, l_b, u_b):
    t_ri = torch.rand(N_ri, 1) * 0.2 * (u_b - l_b)
    x_ri = torch.rand(N_ri, 1)
    y_ri = torch.rand(N_ri, 1)
    return t_ri, x_ri, y_ri

# Normalize data with min max


def normalize_data(x_r, y_r, t_r,
                   u_b, x_b, y_b, t_b,
                   u_i, x_i, y_i, t_i):
    x_r, t_r = 2*(x_r-x_r.min())/(x_r.max()-x_r.min()) - \
        1, 2*(t_r-t_r.min())/(t_r.max()-t_r.min())-1
    x_b, t_b = 2*(x_b-x_b.min())/(x_b.max()-x_b.min()) - \
        1, 2*(t_b-t_b.min())/(t_b.max()-t_b.min())-1
    x_i, t_i = 2*(x_i-x_i.min())/(x_i.max()-x_i.min())-1, -1*torch.ones(N_i, 1)
    y_r = 2*(y_r-y_r.min())/(y_r.max()-y_r.min())-1
    y_b = 2*(y_b-y_b.min())/(y_b.max()-y_b.min())-1
    y_i = 2*(y_i-y_i.min())/(y_i.max()-y_i.min())-1
    return x_r, y_r, t_r, u_b, x_b, y_b, t_b, u_i, x_i, y_i, t_i


def unnormalize_data(x_r, y_r, t_r,
                     u_b, x_b, y_b, t_b,
                     u_i, x_i, y_i, t_i,
                     x_r_min, x_r_max,
                     y_r_min, y_r_max,
                     t_r_min, t_r_max,
                     u_b_min, u_b_max,
                     x_b_min, x_b_max,
                     y_b_min, y_b_max,
                     t_b_min, t_b_max,
                     u_i_min, u_i_max,
                     x_i_min, x_i_max,
                     y_i_min, y_i_max,
                     t_i_min, t_i_max):
    x_r, t_r = x_r*(x_r_max-x_r_min)+x_r_min, t_r*(t_r_max-t_r_min)+t_r_min
    u_b, x_b, t_b = u_b*(u_b_max-u_b_min)+u_b_min, x_b * \
        (x_b_max-x_b_min)+x_b_min, t_b*(t_b_max-t_b_min)+t_b_min
    u_i, x_i, t_i = u_i*(u_i_max-u_i_min)+u_i_min, x_i * \
        (x_i_max-x_i_min)+x_i_min, t_i*(t_i_max-t_i_min)+t_i_min
    y_r = y_r*(y_r_max-y_r_min)+y_r_min
    y_b = y_b*(y_b_max-y_b_min)+y_b_min
    y_i = y_i*(y_i_max-y_i_min)+y_i_min
    return x_r, y_r, t_r, u_b, x_b, y_b, t_b, u_i, x_i, y_i, t_i


############################################################## TRAIN VAL SPLIT ###################################################################
def val_split(x_r, y_r, t_r, u_b, x_b, y_b, t_b, u_i, x_i, y_i, t_i, split=0.2):
    """Splits data into training and validation set with random order"""
    x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i = x_r.to(device), t_r.to(device), u_b.to(
        device), x_b.to(device), t_b.to(device), u_i.to(device), x_i.to(device), t_i.to(device)
    y_r = y_r.to(device)
    y_b = y_b.to(device)
    y_i = y_i.to(device)

    N_r = x_r.shape[0]
    N_b = x_b.shape[0]
    N_i = x_i.shape[0]
    N_r_val = int(N_r*split)
    N_b_val = int(N_b*split)
    N_i_val = int(N_i*split)
    N_r_train = N_r - N_r_val
    N_b_train = N_b - N_b_val
    N_i_train = N_i - N_i_val
    # Permet de mélanger pr que les données ne soient pas dans l'ordre pr l'entrainement
    idx_r = torch.randperm(N_r)
    idx_b = torch.randperm(N_b)
    idx_i = torch.randperm(N_i)
    x_r_train, t_r_train = x_r[idx_r[:N_r_train]], t_r[idx_r[:N_r_train]]
    y_r_train = y_r[idx_r[:N_r_train]]

    x_r_val, t_r_val = x_r[idx_r[N_r_train:]], t_r[idx_r[N_r_train:]]
    y_r_val = y_r[idx_r[N_r_train:]]

    u_b_train, x_b_train, t_b_train = u_b[idx_b[:N_b_train]
                                          ], x_b[idx_b[:N_b_train]], t_b[idx_b[:N_b_train]]
    y_b_train = y_b[idx_b[:N_b_train]]

    u_b_val, x_b_val, t_b_val = u_b[idx_b[N_b_train:]
                                    ], x_b[idx_b[N_b_train:]], t_b[idx_b[N_b_train:]]
    y_b_val = y_b[idx_b[N_b_train:]]

    u_i_train, x_i_train, t_i_train = u_i[idx_i[:N_i_train]
                                          ], x_i[idx_i[:N_i_train]], t_i[idx_i[:N_i_train]]
    y_i_train = y_i[idx_i[:N_i_train]]

    u_i_val, x_i_val, t_i_val = u_i[idx_i[N_i_train:]
                                    ], x_i[idx_i[N_i_train:]], t_i[idx_i[N_i_train:]]
    y_i_val = y_i[idx_i[N_i_train:]]

    return [x_r_train, y_r_train, t_r_train, u_b_train, x_b_train, y_b_train, t_b_train, u_i_train, x_i_train, y_i_train, t_i_train], [x_r_val, y_r_val, t_r_val, u_b_val, x_b_val, y_b_val, t_b_val, u_i_val, x_i_val, y_i_val, t_i_val]
