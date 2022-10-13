import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim

from torch.autograd import Variable
from torch.autograd import grad
import numpy as np

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

# Réseau de neurones
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2,20),
            nn.Tanh(),
            nn.Linear(20,20),
            nn.Tanh(),
            nn.Linear(20,20),
            nn.Tanh(),
            nn.Linear(20,1)
        )

    def forward(self,x,t):
        return self.model(torch.cat((x,t),dim=1))

net = PINN()

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
N_i,N_b,N_r = 100,100,100

t_0 = torch.zeros(N_i,1)
x_0 = torch.linspace(0,1,N_i).view(N_i,1)
u_0 = torch.sin(np.pi*x_0)

t_b = torch.linspace(0,1,N_b).view(N_b,1)
#x_b evenly distributed in 0 or 1 with total N_b points
x_b = torch.zeros(N_b,1)
u_b = torch.zeros(N_b,1)

t_r = torch.linspace(0,1,N_r).view(N_r,1)
x_r = torch.linspace(0,1,N_r).view(N_r,1)

# Entraîner modèle
def train_step(model,optimizer,x_r,t_r,
               u_b,x_b,t_b,
               u_i,x_i,t_i):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(x_r,t_r,
                   u_b,x_b,t_b,
                   u_i,x_i,t_i)
    loss.backward()
    optimizer.step()
    return loss.item()

def train(model,optimizer,x_r,t_r,
        u_b,x_b,t_b,
        u_i,x_i,t_i,epochs):
    for epoch in range(epochs):
        loss = train_step(model,optimizer,x_r,t_r,
                          u_b,x_b,t_b,
                          u_i,x_i,t_i)
        if epoch%100 == 0:
            print("Epoch: {}, Loss: {}".format(epoch,loss))

opt = optim.Adam(net.parameters(),lr=0.001)
train(net,opt,x_r,t_r,
        u_b,x_b,t_b,
        u_0,x_0,t_0,epochs=10000)


    