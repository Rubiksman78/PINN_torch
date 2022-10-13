import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim

from torch.autograd import Variable
from torch.autograd import grad
"""
x = np.linspace(-1,1,100)
t = np.linspace(0,1,100)

x_tensor = torch.from_numpy(x).float()
x_tensor = x_tensor.reshape(100,1)
x_tensor.requires_grad = True
t_tensor = torch.from_numpy(t).float()
t_tensor = t_tensor.reshape(100,1)
t_tensor.requires_grad = True

it0 = torch.zeros(100,dtype=torch.float,requires_grad=True).reshape(100,1)
ix1 = torch.zeros(100,dtype=torch.float,requires_grad=True).reshape(100,1)+1.0
ix1m = torch.zeros(100,dtype=torch.float,requires_grad=True).reshape(100,1)-1.0
"""

# Calculer résidu
def nth_gradient(f,wrt,n):
    for i in range(n):
        grads = grad(f,wrt,create_graph=True,allow_unused=True,)[0]
        f = grads
        if grads is None:
            print("Bad Grad")
            return torch.tensor(0.)
    return grads

def flat(x):
    m = x.shape[0]
    return [x[i] for i in range(m)]

def f(x,t):
    u = net(x,t) # net à définir
    u_tt = nth_gradient(flat(u),wrt=t,n=2)
    u_xx = nth_gradient(flat(u),wrt=x,n=2)
    residual = u_tt - 4*u_xx
    return residual 

# Calculer loss résidu + loss bords
def loss_fn(x_r,t_r,
            u_b,x_b,t_b,
            u_i,x_i,t_i):
    loss_residual = torch.mean(f(x_r,t_r)**2)
    u_pred_b = net(x_b,t_b)
    loss_bords = torch.mean((u_pred_b-u_b)**2)

    u_pred_i = net(x_i,t_i)
    loss_init = torch.mean((u_pred_i-u_i)**2)
    return loss_residual + loss_bords + loss_init

# Définir points bords + initial

# Entraîner modèle