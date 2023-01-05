import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.autograd import grad
from itertools import chain
import torchsummary
from real_sol2d import real_sol
from vrac.bails_sombres import RNN, Transformer
from variable_speed import c_fun
from config import DEFAULT_CONFIG
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Scaling_layer(nn.Module):  # Couche de normalisation des données entre -1 et 1
    def __init__(self):
        super(Scaling_layer, self).__init__()

        # On est maintenant en 2D donc on a 3 entrées (x,y,t)
        self.lb = torch.tensor([0.0, 0.0, 0.0]).to(device)  # lower bound
        self.ub = torch.tensor([1.0, 1.0, 1.0]).to(device)  # upper bound

    def forward(self, x):
        return 2 * (x - self.lb) / (self.ub - self.lb)

# Réseau de neurones


class network(torch.jit.ScriptModule):
    def __init__(self, N_neurons, N_layers):
        super().__init__()
        self.num_neurons = N_neurons
        self.num_layers = N_layers
        self.scaling_layer = Scaling_layer()

        # On est maintenant en 2D donc on a 3 entrées (x,y,t)
        self.linear_input = nn.Linear(3, self.num_neurons)
        self.linear_hidden = nn.ModuleList(
            [nn.Linear(self.num_neurons, self.num_neurons) for _ in range(self.num_layers)])
        self.linear_output = nn.Linear(self.num_neurons, 1)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(self.num_neurons)

    def forward(self, x):
        #x = self.scaling_layer(x)
        x = self.activation(self.linear_input(x))
        for i, linear in enumerate(self.linear_hidden):
            x = self.activation(linear(x))
            #x = self.bn(x)
            x = self.dropout(x)
        x = self.linear_output(x)
        return x


class PINN():
    def __init__(self, with_rnn=False, N_neurons=64, N_layers=4):
        if with_rnn == True:
            #self.net = RNN(2, 64, 1, num_layers=4).to(device)

            # On est maintenant en 2D donc on a 3 entrées (x,y,t)
            self.net = Transformer(3, 16, 1, num_layers=4).to(device)
        else:
            self.net = network(N_neurons, N_layers).to(device)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=DEFAULT_CONFIG['lr'])  # descente de gradient
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=DEFAULT_CONFIG['epochs'])
        self.loss_history = []
        self.loss_history_val = []

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
        laplacian_t = torch.zeros(tensor.shape[0], 1, device=device)

        # On est maintenant en 2D donc on a 3 entrées (x,y,t)
        laplacian_y = torch.zeros(tensor.shape[0], 1, device=device)

        for i, tensori in enumerate(tensor):
            hess = torch.autograd.functional.hessian(
                model, tensori.unsqueeze(0), create_graph=True)
            hess = hess.view(2, 2)
            laplacian_x[i] = hess[0, 0]
            laplacian_t[i] = hess[2, 2]

            # On est maintenant en 2D donc on a 3 entrées (x,y,t)
            laplacian_y[i] = hess[1, 1]
        return laplacian_x, laplacian_y, laplacian_t

    def flat(self, x):
        m = x.shape[0]
        return [x[i] for i in range(m)]

    def f(self, x, y, t, variable_speed=False):
        u_xx, u_yy, u_tt = self.calculate_laplacian(
            self.net, torch.cat([x, y, t], 1))
        if variable_speed:
            c = c_fun(x, y, t)
            #residual = u_tt - c*u_xx - (c**2-1)*(np.pi**2)*torch.sin(np.pi*x)*torch.sin(np.pi*t)

            # On est maintenant en 2D donc on a 3 entrées (x,y,t)
            residual = u_tt - c*(u_xx+u_yy)
        else:
            residual = u_tt - 4*(u_xx+u_yy)
            #lap,_ = self.calculate_laplacian(self.net, torch.cat([x, t], 1))
            #residual = lap - np.pi**2*torch.sin(np.pi*x)*torch.sin(np.pi*t)
        return residual

    # On est maintenant en 2D donc on a 3 entrées (x,y,t)
    def loss_first(self, x_ri, y_ri, t_ri):
        real_solution = real_sol(x_ri, y_ri, t_ri)
        u_pred_r = self.net(torch.cat([x_ri, y_ri, t_ri], 1))
        loss_residual = torch.mean(
            (u_pred_r-real_solution)**2)
        return loss_residual

    # On est maintenant en 2D donc on a 3 entrées (x,y,t)
    def loss_fn(self, x_r, y_r, t_r,
                u_b, x_b, y_b, t_b,
                u_i, x_i, y_i, t_i, validation=False):

        loss_residual = torch.mean(torch.abs(self.f(x_r, y_r, t_r)))

        u_pred_b = self.net(torch.cat([x_b, y_b, t_b], 1))
        loss_bords = torch.mean((u_pred_b-u_b)**2)
        u_pred_i = self.net(torch.cat([x_i, y_i, t_i], 1))
        loss_init = torch.mean((u_pred_i-u_i)**2)

        # Compute derivative of u_pred_b with respect to t_b
        #u_pred_b_t = torch.autograd.functional.jacobian(self.net, torch.cat([x_b, t_b], 1), create_graph=True)
        loss_bords_der = torch.zeros(1, device=device)
        #loss_bords_der = torch.mean((u_pred_b_t)**2)

        """
        #Add truncated boundary c u_tx - u_tt = 0 at t = 1
        t_r_1 = torch.ones_like(t_r, requires_grad=True)
        u_pred_t_1 = self.net(x_r, t_r_1)
        u_pred_t_1_x = self.nth_gradient(self.flat(u_pred_t_1), wrt=x_r, n=1)
        u_pred_t_1_x_t = self.nth_gradient(self.flat(u_pred_t_1_x), wrt=t_r_1, n=1)
        u_tt = self.nth_gradient(self.flat(u_pred_t_1), wrt=t_r_1, n=2)
        loss_trunc = torch.mean((u_pred_t_1_x_t - u_tt)**2)
        """
        loss_trunc = torch.zeros(1, device=device)
        return loss_residual, loss_bords, loss_init, loss_bords_der, loss_trunc

    def train_step(self, train_data, phase="later"):
        if phase == "beginning":
            t_ri, x_ri, y_ri = train_data
            self.net.train()
            self.optimizer.zero_grad()
            loss = self.loss_first(x_ri, y_ri, t_ri)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        else:
            x_r, y_r, t_r, u_b, x_b, y_b, t_b, u_i, x_i, y_i, t_i, = train_data
            self.net.train()
            self.optimizer.zero_grad()
            loss_residual, loss_bords, loss_init, loss_bords_der, loss_trunc = self.loss_fn(x_r, y_r, t_r,
                                                                                            u_b, x_b, y_b, t_b,
                                                                                            u_i, x_i, y_i, t_i)
            loss = loss_residual + loss_bords + loss_init + loss_bords_der + 0.5*loss_trunc
            loss.backward()
            self.optimizer.step()
            return loss_residual.item(), loss_bords.item(), loss_init.item(), loss_bords_der.item(), loss_trunc.item()

    def val_step(self, val_data, phase="later"):
        if phase == "beginning":
            x_ri, y_ri, t_ri = val_data
            loss = self.loss_first(x_ri, y_ri, t_ri)
            return loss.item()
        else:
            x_r, y_r, t_r, u_b, x_b, y_b, t_b, u_i, x_i, y_i, t_i = val_data
            self.net.eval()
            loss_residual, loss_bords, loss_init, loss_bords_der, loss_trunc = self.loss_fn(x_r, y_r, t_r,
                                                                                            u_b, x_b, y_b, t_b,
                                                                                            u_i, x_i, y_i, t_i, validation=False)
            loss = loss_residual + loss_bords + loss_init + loss_bords_der + loss_trunc
            return loss.item()

    def accuracy_step(self, val_data):
        x_r, y_r, t_r, _, _, _, _, _, _ = val_data
        self.net.eval()
        # Compute MSE between real_sol and net
        with torch.no_grad():
            u_pred = self.net(torch.cat([x_r, y_r, t_r], 1))
            real_u = real_sol(x_r, y_r, t_r)
            num = torch.mean(torch.square(u_pred-real_u))
            #den = torch.mean(torch.square(real_u))
        return num.item()
