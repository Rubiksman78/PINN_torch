import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from network import PINN
import torch.functional as F
from real_sol import real_sol
from config import DEFAULT_CONFIG
from dataset import *
from torch.utils.tensorboard import SummaryWriter
import cProfile

writer = SummaryWriter()

############################################################## POINTS PLOTTING #############################################################
############################################################################################################################################
def plot_training_points(t_0, t_b, t_r, x_0, x_b, x_r, u_0, u_b):
    """
    Input: -dimension = spatial dimension
           -time_x = [t_0,t_b,t_r,x_0,x_b,x_r,u_0,u_b]
    Output: display training points in either 1,2 or 3D
    """
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.scatter(t_0, x_0[:, 0], c=u_0, marker='X', vmin=-1, vmax=1)
    ax.scatter(t_b, x_b[:, 0], c=u_b, marker='X', vmin=-1, vmax=1)
    ax.scatter(t_r, x_r[:, 0], c='r', marker='.', alpha=0.1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x1$')
    ax.set_title('Positions of collocation points and boundary data')
    plt.show()

########################################################## PLOTTING FUNCTIONS ###########################################################
##########################################################################################################################################

def plot1dgrid_real(lb, ub, N, model, k, with_rnn=False):
    """Same for the real solution"""
    model = model.net
    x1space = np.linspace(lb[0], ub[0], N)
    tspace = np.linspace(lb[1], ub[1], N)
    T, X1 = np.meshgrid(tspace, x1space)
    T = torch.from_numpy(T).view(1, N*N, 1).to(device).float()
    X1 = torch.from_numpy(X1).view(1, N*N, 1).to(device).float()
    if not with_rnn:
        T = T.transpose(0, 1).squeeze(-1)
        X1 = X1.transpose(0, 1).squeeze(-1)
    else:
        T = T.transpose(0, 1)
        X1 = X1.transpose(0, 1)
    upred = model(torch.cat((X1,T), 1))
    U = torch.squeeze(upred).detach().cpu().numpy()
    U = upred.view(N, N).detach().cpu().numpy()
    T, X1 = T.view(N, N).detach().cpu().numpy(), X1.view(
        N, N).detach().cpu().numpy()
    z_array = np.zeros((N, N))
    for i in range(N):
        z_array[:, i] = U[i]

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.scatter(T, X1, c=U, marker='X')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x1$')

    # Partie 3d
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.scatter(T, X1, U, c=U, marker='X')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$x1$')

    plt.savefig(f'results/real_sol_{k}')
    writer.add_figure(f'real_sol_{k}', fig)
    plt.close()

# Plot train and val losses on same figure
def plot_loss(train_losses, val_losses, accuracy):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.style.use('dark_background')
    ax1.plot(train_losses, label='train')
    ax1.plot(val_losses, label='val')
    ax2.plot(accuracy, label="error", color='red')

    ax1.set(ylabel='Loss')
    ax2.set(ylabel='Error')
    plt.xlabel('Epoch')

    ax1.legend()
    ax2.legend()
    plt.savefig(f'results/loss')
    plt.close()

########################################################### TRAINING ###########################################################
################################################################################################################################
def train(model, train_data, val_data, train_data_begin,
          epochs):
    epochs = tqdm(range(epochs), desc="Training")
    losses = []
    val_losses = []
    acc = []
    for epoch in epochs:
        # Shuffle train_data
        index_shuf_r = torch.randperm(train_data[0].shape[0])
        index_shuf_b = torch.randperm(train_data[2].shape[0])
        index_shuf_i = torch.randperm(train_data[5].shape[0])
        x_r_train = train_data[0][index_shuf_r].to(device)
        t_r_train = train_data[1][index_shuf_r].to(device)
        u_b_train = train_data[2][index_shuf_b].to(device)
        x_b_train = train_data[3][index_shuf_b].to(device)
        t_b_train = train_data[4][index_shuf_b].to(device)
        u_i_train = train_data[5][index_shuf_i].to(device)
        x_i_train = train_data[6][index_shuf_i].to(device)
        t_i_train = train_data[7][index_shuf_i].to(device)
        train_data_new = [x_r_train, t_r_train, u_b_train, x_b_train, t_b_train, u_i_train, x_i_train, t_i_train]
        #Shuffle train_data_begin
        index_shuf_b = torch.randperm(train_data_begin[0].shape[0])
        x_b_train = train_data_begin[0][index_shuf_b].to(device)
        t_b_train = train_data_begin[1][index_shuf_b].to(device)
        train_data_begin = [t_b_train, x_b_train]
        train_data = train_data_new
        if epoch < 1000:
            loss_begin = model.train_step(train_data_begin, phase="beginning")
            epochs.set_postfix(loss=loss_begin)
        else:
            loss_residual,loss_bords,loss_init,loss_bords_der,loss_trunc = model.train_step(train_data)
            loss = loss_residual + loss_bords + loss_init + loss_bords_der + loss_trunc
            val_loss = model.val_step(val_data)
            accuracy = model.accuracy_step(val_data)
            epochs.set_postfix(loss_residual=loss_residual,
                loss_bords=loss_bords, 
                loss_init=loss_init, 
                loss_bords_der=loss_bords_der, 
                loss_trunc=loss_trunc, 
                loss=loss, 
                val_loss=val_loss, 
                accuracy=accuracy)

            #Scheduler step
            model.scheduler.step()

            #Append loss lists (and eventually log for Tensorboard)
            losses.append(loss)
            val_losses.append(val_loss)
            acc.append(accuracy)

            writer.add_scalar('Loss_residual', loss_residual, epoch)
            writer.add_scalar('Loss_bords', loss_bords, epoch)
            writer.add_scalar('Loss_init', loss_init, epoch)
            writer.add_scalar('Loss_bords_der', loss_bords_der, epoch)
            writer.add_scalar('Loss_trunc', loss_trunc, epoch)
            writer.add_scalar('Loss', loss, epoch)
            writer.add_scalar('Val_loss', val_loss, epoch)
            writer.add_scalar('Accuracy', accuracy, epoch)

            plot_loss(losses, val_losses, acc)

        if epoch % 100 == 0:
            plot1dgrid_real(lb, ub, N_plotting, model, epoch)
        if epoch % 1000 == 0:
            torch.save(model.net.state_dict(), f"results/model_{epoch}.pt")
       

if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    N_i,N_b,N_r = DEFAULT_CONFIG['N_i'],DEFAULT_CONFIG['N_b'],DEFAULT_CONFIG['N_r']
    l_b,u_b = DEFAULT_CONFIG['l_b'],DEFAULT_CONFIG['u_b']
    N_neurons, N_layers = DEFAULT_CONFIG['N_neurons'],DEFAULT_CONFIG['N_layers']
    
    with_rnn = False
    net = PINN(with_rnn=with_rnn, N_neurons=N_neurons, N_layers=N_layers)
    #net._model_summary()
 
    #Write config to tensorboard
    writer.add_text('Config', str(DEFAULT_CONFIG))

    t_ri,x_ri = define_points_begin(10000, l_b, u_b)
    t_i,x_i,u_i,t_b,x_b,u_b,t_r,x_r = define_points(N_i,N_b,N_r,l_b,u_b)
    #x_r,t_r,u_b,x_b,t_b,u_i,x_i,t_i = normalize_data(x_r,t_r,
    #    u_b,x_b,t_b,
    #    u_i,x_i,t_i)
    plot_training_points(t_i.data.numpy(),
                    t_b.data.numpy(),
                    t_r.data.numpy(),
                    x_i.data.numpy(),
                    x_b.data.numpy(),
                    x_r.data.numpy(),
                    u_i.data.numpy(),
                    u_b.data.numpy())
    
    train_data, val_data = val_split(
        x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i, split=0.1)
    
    train_data_begin = [t_ri.to(device),x_ri.to(device)]
    lb = [0,0]
    ub = [1,1]
    N_plotting = DEFAULT_CONFIG['N_plotting']
    epochs = DEFAULT_CONFIG['epochs']

    train(net, train_data, val_data,train_data_begin, epochs=epochs)

    writer.flush()
    writer.close()

    net.net.load_state_dict(torch.load("model_9000.pt"))
    plot1dgrid_real(lb,ub,N_plotting,net,10000,show=True)