import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.autograd import grad
from itertools import chain
import torchsummary
from real_sol import real_sol
from vrac.bails_sombres import RNN, Transformer
from variable_speed import c_fun
from config import DEFAULT_CONFIG
import numpy as np
from dataset import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Our following code was in 1D and we want to make it 2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()


def plot1dgrid_real(lb, ub, N, model, k):
    """Same for the real solution"""
    x1space = np.linspace(lb[0], ub[0], N)
    # We add y coordinates to make it 2D
    y1space = np.linspace(lb[0], ub[0], N)
    tspace = np.linspace(lb[1], ub[1], N)
    X1, Y1, T = np.meshgrid(tspace, x1space, y1space)
    T = torch.from_numpy(T).view(1, N**3, 1).to(device).float()
    X1 = torch.from_numpy(X1).view(1, N**3, 1).to(device).float()
    Y1 = torch.from_numpy(Y1).view(1, N**3, 1).to(device).float()
    T = T.transpose(0, 1).squeeze(-1)
    X1 = X1.transpose(0, 1).squeeze(-1)
    Y1 = Y1.transpose(0, 1).squeeze(-1)
    upred = model(torch.cat((X1, Y1, T), 1))
    U = torch.squeeze(upred).detach().cpu().numpy()
    U = upred.view(N, N, N).detach().cpu().numpy()
    T, X1, Y1 = T.view(N, N, N).detach().cpu().numpy(), X1.view(
        N, N, N).detach().cpu().numpy(), Y1.view(N, N, N).detach().cpu().numpy()
    z_array = np.zeros((N, N, N))
    for i in range(N):
        z_array[:, :, i] = U[i]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(T, X1, Y1, c=U, marker='X')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x1$')

    plt.savefig(f'results2Dnew/real_sol_{k}')
    writer.add_figure(f'real_sol_{k}', fig)
    plt.close()


def plot_loss(train_losses, val_losses):
    fig, ax1 = plt.subplots(1, 1)
    plt.style.use('dark_background')
    ax1.plot(train_losses, label='train')
    ax1.plot(val_losses, label='val')

    ax1.set(ylabel='Loss')
    plt.xlabel('Epoch')

    ax1.legend()
    plt.savefig(f'results2Dnew/loss')
    plt.close()

# Réseau de neurones


class CubicReLU(nn.Module):
    def __init__(self):
        super(CubicReLU, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        return torch.max(torch.tensor(0.0, device=self.device), x) ** 3


class network(torch.jit.ScriptModule):
    def __init__(self, N_neurons, N_layers):
        super().__init__()
        self.num_neurons = N_neurons
        self.num_layers = N_layers
        self.linear_input = nn.Linear(3, self.num_neurons)  # 3 inputs
        self.linear_hidden = nn.ModuleList(
            [nn.Linear(self.num_neurons, self.num_neurons) for _ in range(self.num_layers)])
        self.linear_output = nn.Linear(self.num_neurons, 1)
        self.activation = CubicReLU()  # nn.Tanh() if not working

    def forward(self, x):
        x = self.activation(self.linear_input(x))
        for i, linear in enumerate(self.linear_hidden):
            x = self.activation(linear(x))
        x = self.linear_output(x)
        return x


class PINN():
    def __init__(self, segments, N_neurons=64, N_layers=4):
        self.net = network(N_neurons, N_layers).to(device)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=DEFAULT_CONFIG['lr'])  # descente de gradient
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=DEFAULT_CONFIG['epochs'])
        self.loss_history = []
        self.loss_history_val = []
        self.segments = segments

    def _model_summary(self):
        print(torchsummary.summary(self.net, [(32, 1), (32, 1)]))

    # Calculer résidu
    def nth_gradient(self, f, wrt, n):
        for i in range(n):
            f = list(chain(*f))
            grads = grad(f, wrt, create_graph=True, allow_unused=True,)[0]
            f = grads
            if grads is None:
                print("Bad Grad")
                return None
        return grads

    def calculate_laplacian(self, model, tensor):
        laplacian_x = torch.zeros(tensor.shape[0], 1, device=device)
        laplacian_y = torch.zeros(tensor.shape[0], 1, device=device)
        laplacian_t = torch.zeros(tensor.shape[0], 1, device=device)
        for i, tensori in enumerate(tensor):
            hess = torch.autograd.functional.hessian(
                model, tensori.unsqueeze(0), create_graph=True)
            hess = hess.view(2, 2)
            laplacian_x[i] = hess[0, 0]
            laplacian_y[i] = hess[1, 1]
            laplacian_t[i] = hess[2, 2]
        return laplacian_x, laplacian_y, laplacian_t

    def flat(self, x):
        m = x.shape[0]
        return [x[i] for i in range(m)]

    def dist(self, x1, y1, t1, x2, y2, t2):
        return torch.sqrt((x1-x2)**2+(t1-t2)**2+(y1-y2)**2)

    def linseg(self, x, y, t, x1, y1, t1, x2, y2, t2):
        L = self.dist(x1, y1, t1, x2, y2, t2)
        xc = (x1+x2)*0.5
        tc = (t1+t2)*0.5
        yc = (y1+y2)*0.5
        # f = (1/L)*((x-x1)*(t2-t1) -(t-t1)*(x2-x1) )
        # Change the signed distance f with now 3 dimensions
        f = (1/L)*((x-x1)*(t2-t1) - (t-t1)*(x2-x1) +
                   (y-y1)*(x2-x1) - (x-x1)*(y2-y1) +
                   (t-t1)*(y2-y1) - (y-y1)*(t2-t1))
        t = (1/L)*((L/2.)**2 - self.dist(x, y, t, xc, yc, tc)**2)
        varphi = torch.sqrt(t**2 + f**4)
        phi = torch.sqrt(f**2 + 0.25*(varphi-t)**2)
        return phi

    def phi(self, x, y, t):  # segments is an array of all the segments composing the boundary
        m = 1.
        R = 0.
        for i in range(len(self.segments[:, 0])):
            phi_v = self.linseg(
                x, y, t, self.segments[i, 0], self.segments[i, 1], self.segments[i, 2], self.segments[i, 3], self.segments[i, 4], self.segments[i, 5])
            R += 1./phi_v**m
        R = 1/R**(1/m)
        return R

    def u(self, z):
        x, y, t = z[:, 0], z[:, 1], z[:, 2]
        x, y, t = x.unsqueeze(1), y.unsqueeze(1), t.unsqueeze(1)
        w = self.phi(x, y, t)*self.net(z)
        # add initial condition oftorch.sin(np.pi*x) + 0.5*torch.sin(4*np.pi*x) for t = 0
        # w[t==0] += torch.sin(np.pi*x[t==0]) + 0.5*torch.sin(4*np.pi*x[t==0])
        w[t == 0] = 5
        return w

    def loss(self, x, y, t):
        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True
        laplacian_u_x = self.nth_gradient(
            self.u(torch.cat((x, y, t), 1)), x, 2)
        laplacian_u_y = self.nth_gradient(
            self.u(torch.cat((x, y, t), 1)), y, 2)
        laplacian_u_t = self.nth_gradient(
            self.u(torch.cat((x, y, t), 1)), t, 2)
        # wave equation
        f = laplacian_u_t - 4*(laplacian_u_x+laplacian_u_y) - 3 * \
            (np.pi**2)*torch.sin(np.pi*x)*torch.sin(np.pi*t)
        loss = torch.mean(f ** 2)
        print(laplacian_u_x, laplacian_u_y, laplacian_u_t)
        return loss

    def train(self, x, y, t, x_val, y_val, t_val, epochs=DEFAULT_CONFIG['epochs']):
        progress_bar = tqdm(range(epochs))
        for epoch in progress_bar:
            self.optimizer.zero_grad()
            loss = self.loss(x, y, t)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            loss_val = self.loss(x_val, y_val, t_val)
            self.loss_history_val.append(loss_val.item())
            plot1dgrid_real([0, 0, 0], [1, 1, 1], 100, self.u, epoch)
            plot_loss(self.loss_history, self.loss_history_val)
            progress_bar.set_description(
                f"Loss: {loss.item():.4f}, Loss_val: {loss_val.item():.4f}")
        return self.loss_history, self.loss_history_val

    def predict(self, x, y, t):
        with torch.no_grad():
            x.requires_grad = False
            y.requires_grad = False
            t.requires_grad = False
            u = self.u(torch.cat((x, y, t), 1))
        return u


segments = torch.tensor(
    [[0, 0, 0, 1], [0, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0]], device=device)
# Segments2 doit etre un tensor de la forme [[x1,y1,t1,x2,y2,t2],...] et represente les 8 sommets d'un cube de cote 1
segments2 = torch.tensor([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0], [
                         0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1], [1, 1, 0, 1, 1, 1], [1, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]], device=device)


PINN = PINN(segments2, N_neurons=16, N_layers=2)
N_points = 20
x = torch.linspace(0.01, 0.99, N_points, device=device).unsqueeze(1)
y = torch.linspace(0.01, 0.99, N_points, device=device).unsqueeze(1)
t = torch.linspace(0.01, 0.99, N_points, device=device).unsqueeze(1)

x_train = x.repeat(N_points, 1)
y_train = y.repeat(N_points, 1)
t_train = t.repeat(N_points, 1).t().reshape(-1, 1)

x_val = torch.linspace(0.01, 0.99, 100, device=device).unsqueeze(1)
y_val = torch.linspace(0.01, 0.99, 100, device=device).unsqueeze(1)
t_val = torch.linspace(0.01, 0.99, 100, device=device).unsqueeze(1)

loss_history, loss_history_val = PINN.train(
    x, y, t, x_val, y_val, t_val, epochs=10000)

plt.plot(loss_history)
plt.plot(loss_history_val)
plt.show()
