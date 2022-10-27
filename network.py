import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torch import optim
from torch.autograd import grad
from itertools import chain
import torchsummary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Scaling_layer(nn.Module):
    def __init__(self):
        super(Scaling_layer, self).__init__()
        self.lb = torch.tensor([0.0,0.0]).to(device)
        self.ub = torch.tensor([1.0,1.0]).to(device)

    def forward(self, x):
        return 2.0*(x - self.lb)/(self.ub - self.lb) - 1.0

# Réseau de neurones
class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_neurons = 32
        self.num_layers = 4
        self.scaling_layer = Scaling_layer()
        self.linear_input = nn.Linear(2,self.num_neurons)
        self.linear_hidden = nn.ModuleList([nn.Linear(self.num_neurons,self.num_neurons) for i in range(self.num_layers)])
        self.linear_output = nn.Linear(self.num_neurons,1)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.2)

    def forward(self,x,t):
        x = self.scaling_layer(torch.cat([x,t],1))
        x = self.activation(self.linear_input(x))
        for i, linear in enumerate(self.linear_hidden):
            x = self.activation(linear(x))
            x = self.dropout(x)
        x = self.linear_output(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2,lstm=False):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = lstm
        if lstm:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self,x,t):
        x = torch.cat([x,t],dim=-1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        if self.lstm:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, (hn,cn) = self.rnn(x, (h0, c0))
            hn = hn.view(-1, self.hidden_size)
            out = nn.Tanh()(hn)
        else:
            out, _ = self.rnn(x, h0)
        out = nn.Tanh()(self.fc1(out)) 
        out = self.fc2(out[:, -1, :])
        return out

class PINN():
    def __init__(self,with_rnn=True):
        #self.net = network().to(device)
        if with_rnn == True:
            self.net = RNN(2, 512, 1, num_layers=4).to(device)
        else:
            self.net = network().to(device)
        self.optimizer = optim.Adam(self.net.parameters(),lr=1e-3)
        #self.optimizer = optim.LBFGS(self.net.parameters(), lr=1e-3)
        self.loss_history = []
        self.loss_history_val = []

    def _model_summary(self):
        print(torchsummary.summary(self.net, [(1,1), (1,1)]))

    # Calculer résidu
    def nth_gradient(self,f,wrt,n):
        for i in range(n):
            f = list(chain(*f))
            grads = grad(f,wrt,create_graph=True,allow_unused=True,)[0]
            f = grads
            if grads is None:
                print("Bad Grad")
                return torch.tensor(0.)
        return grads

    def flat(self,x):
        m = x.shape[0]
        return [x[i] for i in range(m)]

    def f(self,x,t):
        u = self.net(x,t) # net à définir
        u_tt = self.nth_gradient(self.flat(u),wrt=t,n=2)
        u_xx = self.nth_gradient(self.flat(u),wrt=x,n=2)
        residual = u_tt - 4*u_xx
        return residual 

    # Calculer loss résidu + loss bords
    def loss_fn_rnn(self,x_r,t_r,
                u_b,x_b,t_b,
                u_i,x_i,t_i,
                x_r_label,t_r_label,
                u_b_label,x_b_label,t_b_label,
                u_i_label,x_i_label,t_i_label):
        x_r,t_r = Variable(x_r,requires_grad=True).to(device),Variable(t_r,requires_grad=True).to(device)
        u_b,x_b,t_b = Variable(u_b,requires_grad=False).to(device),Variable(x_b,requires_grad=False).to(device),Variable(t_b,requires_grad=False).to(device)
        u_i,x_i,t_i = Variable(u_i,requires_grad=False).to(device),Variable(x_i,requires_grad=False).to(device),Variable(t_i,requires_grad=False).to(device)
        loss_residual = torch.mean(torch.square(self.f(x_r,t_r))) #+ u_capteur - u_pred
        u_pred_b = self.net(x_b,t_b)
        loss_bords = torch.mean((u_pred_b-u_b_label)**2)
        u_pred_i = self.net(x_i,t_i)
        loss_init = torch.mean((u_pred_i-u_i_label)**2)
        return loss_residual + loss_bords + loss_init

    def loss_fn(self,x_r,t_r,
                u_b,x_b,t_b,
                u_i,x_i,t_i,validation=False):
        if validation:
            x_r,t_r = Variable(x_r,requires_grad=False).to(device),Variable(t_r,requires_grad=False).to(device)
            u_b,x_b,t_b = Variable(u_b,requires_grad=False).to(device),Variable(x_b,requires_grad=False).to(device),Variable(t_b,requires_grad=False).to(device)
            u_i,x_i,t_i = Variable(u_i,requires_grad=False).to(device),Variable(x_i,requires_grad=False).to(device),Variable(t_i,requires_grad=False).to(device)
        else:
            x_r,t_r = Variable(x_r,requires_grad=True).to(device),Variable(t_r,requires_grad=True).to(device)
            u_b,x_b,t_b = Variable(u_b,requires_grad=False).to(device),Variable(x_b,requires_grad=False).to(device),Variable(t_b,requires_grad=False).to(device)
            u_i,x_i,t_i = Variable(u_i,requires_grad=False).to(device),Variable(x_i,requires_grad=False).to(device),Variable(t_i,requires_grad=False).to(device)
        loss_residual = torch.mean(torch.square(self.f(x_r,t_r))) #+ u_capteur - u_pred
        u_pred_b = self.net(x_b,t_b)
        loss_bords = torch.mean((u_pred_b-u_b)**2)
        u_pred_i = self.net(x_i,t_i)
        loss_init = torch.mean((u_pred_i-u_i)**2)
        return loss_residual + loss_bords + loss_init

    # Entraîner modèle
    def train_step_rnn(self,train_data):
        x_r,t_r,u_b,x_b,t_b,u_i,x_i,t_i,x_r_label,t_r_label,u_b_label,x_b_label,t_b_label,u_i_label,x_i_label,t_i_label = train_data
        self.net.train()
        self.optimizer.zero_grad()
        loss = self.loss_fn_rnn(x_r,t_r,
                    u_b,x_b,t_b,
                    u_i,x_i,t_i,
                    x_r_label,t_r_label,
                    u_b_label,x_b_label,t_b_label,
                    u_i_label,x_i_label,t_i_label)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_step(self,train_data):
        x_r,t_r,u_b,x_b,t_b, u_i,x_i,t_i, = train_data
        self.net.train()
        self.optimizer.zero_grad()
        loss = self.loss_fn(x_r,t_r,
                    u_b,x_b,t_b,
                    u_i,x_i,t_i)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self,val_data):
        x_r,t_r,u_b,x_b,t_b, u_i,x_i,t_i, = val_data
        self.net.eval()
        loss = self.loss_fn(x_r,t_r,
                    u_b,x_b,t_b,
                    u_i,x_i,t_i,validation=False)
        return loss.item()

    def val_step_rnn(self,val_data):
        x_r,t_r,u_b,x_b,t_b,u_i,x_i,t_i,x_r_label,t_r_label,u_b_label,x_b_label,t_b_label,u_i_label,x_i_label,t_i_label = val_data
        self.net.eval()
        loss = self.loss_fn_rnn(x_r,t_r,
                    u_b,x_b,t_b,
                    u_i,x_i,t_i,
                    x_r_label,t_r_label,
                    u_b_label,x_b_label,t_b_label,
                    u_i_label,x_i_label,t_i_label)
        return loss.item()