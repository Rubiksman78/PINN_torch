import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from network import PINN
import torch
import torch.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with_rnn = False
net = PINN(with_rnn=with_rnn)
net._model_summary()

N_i,N_b,N_r = 100,100,100

t_0 = torch.zeros(N_i,1)
x_0 = torch.linspace(0,1,N_i).view(N_i,1)
#u_0 = torch.sin(np.pi*x_0)
u_i = t_0 + 1*(torch.sin(np.pi*x_0) + 0.5*torch.sin(4*np.pi*x_0))

t_b = torch.linspace(0,1,N_b).view(N_b,1)
#x_b evenly distributed in 0 or 1 with total N_b points
x_b = torch.zeros(N_b,1)
u_b = torch.zeros(N_b,1)

t_r = torch.linspace(0,1,N_r).view(N_r,1)
x_r = torch.linspace(0,1,N_r).view(N_r,1)

def data_to_rnn_sequences(data,seq_len):
    """Converts data to sequences of length seq_len"""
    sequences = []
    for i in range(len(data)-seq_len):
        sequences.append(data[i:i+seq_len])
    return torch.stack(sequences)

def all_data_to_sequences(x_r,t_r,
        u_b,x_b,t_b,
        u_i,x_i,t_i,seq_len):
    x_r,t_r = data_to_rnn_sequences(x_r,seq_len),data_to_rnn_sequences(t_r,seq_len)
    u_b,x_b,t_b = data_to_rnn_sequences(u_b,seq_len),data_to_rnn_sequences(x_b,seq_len),data_to_rnn_sequences(t_b,seq_len)
    u_i,x_i,t_i = data_to_rnn_sequences(u_i,seq_len),data_to_rnn_sequences(x_i,seq_len),data_to_rnn_sequences(t_i,seq_len)
    return x_r,t_r,u_b,x_b,t_b,u_i,x_i,t_i

def sequence_to_label(sequence):
    """Converts a sequence to a label"""
    return sequence[:,-1,:]

def all_data_to_label(x_r,t_r,
        u_b,x_b,t_b,
        u_i,x_i,t_i):
    x_r,t_r = sequence_to_label(x_r),sequence_to_label(t_r)
    u_b,x_b,t_b = sequence_to_label(u_b),sequence_to_label(x_b),sequence_to_label(t_b)
    u_i,x_i,t_i = sequence_to_label(u_i),sequence_to_label(x_i),sequence_to_label(t_i)
    x_r,t_r,u_b,x_b,t_b,u_i,x_i,t_i = x_r.to(device),t_r.to(device),u_b.to(device),x_b.to(device),t_b.to(device),u_i.to(device),x_i.to(device),t_i.to(device)
    return x_r,t_r,u_b,x_b,t_b,u_i,x_i,t_i

if with_rnn:
    x_r,t_r,u_b,x_b,t_b,u_i,x_i,t_i = all_data_to_sequences(x_r,t_r,
            u_b,x_b,t_b,
            u_i,x_0,t_0,seq_len=10)
    x_r_label,t_r_label,u_b_label,x_b_label,t_b_label,u_i_label,x_i_label,t_i_label = all_data_to_label(x_r,t_r,
            u_b,x_b,t_b,
            u_i,x_i,t_i)

def plot1dgrid_real(lb,ub,N,model,k):
    """Same for the real solution"""
    model = model.net
    x1space = np.linspace(lb[1], ub[1], N)
    tspace = np.linspace(lb[0], ub[0], N )
    T,X1 = np.meshgrid(tspace,x1space)
    T = torch.from_numpy(T).view(1,N*N,1).to(device).float()
    X1 = torch.from_numpy(X1).view(1,N*N,1).to(device).float()
    T = T.transpose(0,1)
    X1 = X1.transpose(0,1)
    upred = model(T,X1)
    U = torch.squeeze(upred).cpu().detach().numpy()
    U = upred.view(N,N).detach().cpu().numpy()
    T,X1 = T.view(N,N).detach().cpu().numpy(),X1.view(N,N).detach().cpu().numpy()
    z_array = np.zeros((N,N))
    for i in range(N):
        z_array[:,i]= U[i]

    plt.style.use('dark_background')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(T,X1,c=U, marker='X', vmin=-1, vmax=1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x1$')
    plt.savefig(f'results/generated_{k}')
    plt.close()

def train(model,x_r,t_r,
        u_b,x_b,t_b,
        u_i,x_i,t_i,
        epochs):
    epochs = tqdm(range(epochs),desc="Training")
    for epoch in epochs:
        loss = model.train_step(x_r,t_r,
                          u_b,x_b,t_b,
                          u_i,x_i,t_i)
        epochs.set_postfix(loss=loss)
       

def train_rnn(model,x_r,t_r,
        u_b,x_b,t_b,
        u_i,x_i,t_i,
        x_r_label,t_r_label,u_b_label,x_b_label,t_b_label,u_i_label,x_i_label,t_i_label,
        epochs):
    epochs = tqdm(range(epochs),desc="Training")
    for epoch in epochs:
        loss = model.train_step_rnn(x_r,t_r,
                          u_b,x_b,t_b,
                          u_i,x_i,t_i,
                          x_r_label,t_r_label,u_b_label,x_b_label,t_b_label,u_i_label,x_i_label,t_i_label)
        epochs.set_postfix(loss=loss)
        if epoch%100==0:
            plot1dgrid_real(lb,ub,N,model,epoch)

lb = [0,0]
ub = [1,1]
N = 70
with torch.backends.cudnn.flags(enabled=False):
    if with_rnn:
        train_rnn(net,x_r,t_r,
                u_b,x_b,t_b,
                u_i,x_i,t_i,
                x_r_label,t_r_label,u_b_label,x_b_label,t_b_label,u_i_label,x_i_label,t_i_label
                ,epochs=10000)
    else:
        train(net,x_r,t_r,
                u_b,x_b,t_b,
                u_i,x_0,t_0,
                epochs=10000)
