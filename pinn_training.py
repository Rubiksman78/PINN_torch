import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from network import PINN
import torch
import torch.functional as F
from real_sol import real_sol
from config import DEFAULT_CONFIG


########################################################### POINTS DEFINITION ###########################################################
#########################################################################################################################################
def define_points(N_i, N_b, N_r, l_b, u_b):
    t_i = torch.zeros(N_i, 1)
    x_i = torch.linspace(l_b, u_b, N_i).view(N_i, 1)
    #u_0 = torch.sin(np.pi*x_0)
    u_i = t_i + 1*(torch.sin(np.pi*x_i) + 0.5*torch.sin(4*np.pi*x_i))

    t_b = torch.linspace(l_b, u_b, N_b).view(N_b, 1)
    # x_b evenly distributed in 0 or 1 with total N_b points
    x_b = torch.bernoulli(0.5*torch.ones(N_b, 1))
    u_b = torch.zeros(N_b, 1)

    # On génère des pts randoms dans le domaine sur lesquels on va calculer le residu
    t_r = torch.rand(N_r, 1)
    x_r = torch.rand(N_r, 1)
    return t_i, x_i, u_i, t_b, x_b, u_b, t_r, x_r

# Normalize data with min max


def normalize_data(x_r, t_r,
                   u_b, x_b, t_b,
                   u_i, x_i, t_i):
    x_r, t_r = 2*(x_r-x_r.min())/(x_r.max()-x_r.min()) - \
        1, 2*(t_r-t_r.min())/(t_r.max()-t_r.min())-1
    x_b, t_b = 2*(x_b-x_b.min())/(x_b.max()-x_b.min()) - \
        1, 2*(t_b-t_b.min())/(t_b.max()-t_b.min())-1
    x_i, t_i = 2*(x_i-x_i.min())/(x_i.max()-x_i.min())-1, -1*torch.ones(N_i, 1)
    return x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i


def unnormalize_data(x_r, t_r,
                     u_b, x_b, t_b,
                     u_i, x_i, t_i,
                     x_r_min, x_r_max,
                     t_r_min, t_r_max,
                     u_b_min, u_b_max,
                     x_b_min, x_b_max,
                     t_b_min, t_b_max,
                     u_i_min, u_i_max,
                     x_i_min, x_i_max,
                     t_i_min, t_i_max):
    x_r, t_r = x_r*(x_r_max-x_r_min)+x_r_min, t_r*(t_r_max-t_r_min)+t_r_min
    u_b, x_b, t_b = u_b*(u_b_max-u_b_min)+u_b_min, x_b * \
        (x_b_max-x_b_min)+x_b_min, t_b*(t_b_max-t_b_min)+t_b_min
    u_i, x_i, t_i = u_i*(u_i_max-u_i_min)+u_i_min, x_i * \
        (x_i_max-x_i_min)+x_i_min, t_i*(t_i_max-t_i_min)+t_i_min
    return x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i


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

############################################################## SEQUENCES FOR RNN ###################################################################
####################################################################################################################################################


def data_to_rnn_sequences(data, seq_len):
    """Converts data to sequences of length seq_len"""
    sequences = []
    for i in range(len(data)-seq_len):
        sequences.append(data[i:i+seq_len])
    return torch.stack(sequences)


def all_data_to_sequences(x_r, t_r,
                          u_b, x_b, t_b,
                          u_i, x_i, t_i, seq_len):
    x_r, t_r = data_to_rnn_sequences(
        x_r, seq_len), data_to_rnn_sequences(t_r, seq_len)
    u_b, x_b, t_b = data_to_rnn_sequences(u_b, seq_len), data_to_rnn_sequences(
        x_b, seq_len), data_to_rnn_sequences(t_b, seq_len)
    u_i, x_i, t_i = data_to_rnn_sequences(u_i, seq_len), data_to_rnn_sequences(
        x_i, seq_len), data_to_rnn_sequences(t_i, seq_len)
    return x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i


def sequence_to_label(sequence):
    """Converts a sequence to a label"""
    return sequence[:, -1, :]


def all_data_to_label(x_r, t_r,
                      u_b, x_b, t_b,
                      u_i, x_i, t_i):
    x_r, t_r = sequence_to_label(x_r), sequence_to_label(t_r)
    u_b, x_b, t_b = sequence_to_label(
        u_b), sequence_to_label(x_b), sequence_to_label(t_b)
    u_i, x_i, t_i = sequence_to_label(
        u_i), sequence_to_label(x_i), sequence_to_label(t_i)
    x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i = x_r.to(device), t_r.to(device), u_b.to(
        device), x_b.to(device), t_b.to(device), u_i.to(device), x_i.to(device), t_i.to(device)
    return x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i

############################################################## TRAIN VAL SPLIT ###################################################################


def val_split(x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i, split=0.2):
    """Splits data into training and validation set with random order"""
    x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i = x_r.to(device), t_r.to(device), u_b.to(
        device), x_b.to(device), t_b.to(device), u_i.to(device), x_i.to(device), t_i.to(device)
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
    x_r_val, t_r_val = x_r[idx_r[N_r_train:]], t_r[idx_r[N_r_train:]]
    u_b_train, x_b_train, t_b_train = u_b[idx_b[:N_b_train]
                                          ], x_b[idx_b[:N_b_train]], t_b[idx_b[:N_b_train]]
    u_b_val, x_b_val, t_b_val = u_b[idx_b[N_b_train:]
                                    ], x_b[idx_b[N_b_train:]], t_b[idx_b[N_b_train:]]
    u_i_train, x_i_train, t_i_train = u_i[idx_i[:N_i_train]
                                          ], x_i[idx_i[:N_i_train]], t_i[idx_i[:N_i_train]]
    u_i_val, x_i_val, t_i_val = u_i[idx_i[N_i_train:]
                                    ], x_i[idx_i[N_i_train:]], t_i[idx_i[N_i_train:]]
    return [x_r_train, t_r_train, u_b_train, x_b_train, t_b_train, u_i_train, x_i_train, t_i_train], [x_r_val, t_r_val, u_b_val, x_b_val, t_b_val, u_i_val, x_i_val, t_i_val]


def val_split_with_labels(x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i, split=0.2):
    """Splits data into training and validation set with random order"""
    x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i = x_r.to(device), t_r.to(device), u_b.to(
        device), x_b.to(device), t_b.to(device), u_i.to(device), x_i.to(device), t_i.to(device)
    N_r = x_r.shape[0]
    N_b = x_b.shape[0]
    N_i = x_i.shape[0]
    N_r_val = int(N_r*split)
    N_b_val = int(N_b*split)
    N_i_val = int(N_i*split)
    N_r_train = N_r - N_r_val
    N_b_train = N_b - N_b_val
    N_i_train = N_i - N_i_val
    idx_r = torch.randperm(N_r)
    idx_b = torch.randperm(N_b)
    idx_i = torch.randperm(N_i)
    x_r_train, t_r_train = x_r[idx_r[:N_r_train]], t_r[idx_r[:N_r_train]]
    x_r_val, t_r_val = x_r[idx_r[N_r_train:]], t_r[idx_r[N_r_train:]]
    u_b_train, x_b_train, t_b_train = u_b[idx_b[:N_b_train]
                                          ], x_b[idx_b[:N_b_train]], t_b[idx_b[:N_b_train]]
    u_b_val, x_b_val, t_b_val = u_b[idx_b[N_b_train:]
                                    ], x_b[idx_b[N_b_train:]], t_b[idx_b[N_b_train:]]
    u_i_train, x_i_train, t_i_train = u_i[idx_i[:N_i_train]
                                          ], x_i[idx_i[:N_i_train]], t_i[idx_i[:N_i_train]]
    u_i_val, x_i_val, t_i_val = u_i[idx_i[N_i_train:]
                                    ], x_i[idx_i[N_i_train:]], t_i[idx_i[N_i_train:]]
    x_r_train_label, t_r_train_label = sequence_to_label(
        x_r_train), sequence_to_label(t_r_train)
    x_r_val_label, t_r_val_label = sequence_to_label(
        x_r_val), sequence_to_label(t_r_val)
    u_b_train_label, x_b_train_label, t_b_train_label = sequence_to_label(
        u_b_train), sequence_to_label(x_b_train), sequence_to_label(t_b_train)
    u_b_val_label, x_b_val_label, t_b_val_label = sequence_to_label(
        u_b_val), sequence_to_label(x_b_val), sequence_to_label(t_b_val)
    u_i_train_label, x_i_train_label, t_i_train_label = sequence_to_label(
        u_i_train), sequence_to_label(x_i_train), sequence_to_label(t_i_train)
    u_i_val_label, x_i_val_label, t_i_val_label = sequence_to_label(
        u_i_val), sequence_to_label(x_i_val), sequence_to_label(t_i_val)
    return [x_r_train, t_r_train, u_b_train, x_b_train, t_b_train, u_i_train, x_i_train, t_i_train], [x_r_val, t_r_val, u_b_val, x_b_val, t_b_val, u_i_val, x_i_val, t_i_val], [x_r_train_label, t_r_train_label, u_b_train_label, x_b_train_label, t_b_train_label, u_i_train_label, x_i_train_label, t_i_train_label], [x_r_val_label, t_r_val_label, u_b_val_label, x_b_val_label, t_b_val_label, u_i_val_label, x_i_val_label, t_i_val_label]

########################################################### PLOTTING FUNCTIONS ###########################################################
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
    upred = model(X1, T)
    U = torch.squeeze(upred).detach().cpu().numpy()
    U = upred.view(N, N).detach().cpu().numpy()
    T, X1 = T.view(N, N).detach().cpu().numpy(), X1.view(
        N, N).detach().cpu().numpy()
    z_array = np.zeros((N, N))
    for i in range(N):
        z_array[:, i] = U[i]

    plt.style.use('dark_background')

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

    plt.savefig(f'results/generated_{k}')
    plt.close()

# Plot train and val losses on same figure


def plot_loss(train_losses, val_losses, accuracy):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.style.use('dark_background')
    ax1.plot(train_losses, label='train')
    ax1.plot(val_losses, label='val')
    ax2.plot(accuracy, label="accuracy", color='red')

    ax1.set(ylabel='Loss')
    ax2.set(ylabel='Accuracy')
    plt.xlabel('Epoch')

    ax1.legend()
    ax2.legend()
    plt.savefig(f'results/loss')
    plt.close()


# def plot_loss(train_losses, val_losses, accuracy):
#     plt.style.use('dark_background')
#     plt.plot(train_losses, label='train')
#     plt.plot(val_losses, label='val')
#     plt.plot(accuracy, label="accuracy")
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(f'results/loss')
#     plt.close()

########################################################### TRAINING ###########################################################
################################################################################################################################
def train(model, train_data, val_data,
          epochs):
    epochs = tqdm(range(epochs), desc="Training")
    losses = []
    val_losses = []
    acc = []
    for epoch in epochs:
        # Shuffle train_data
        # On remélange pr pas entrainer sur la mm chose ds le mm ordre
        index_shuf_r = torch.randperm(train_data[0].shape[0])
        index_shuf_b = torch.randperm(train_data[2].shape[0])
        index_shuf_i = torch.randperm(train_data[5].shape[0])
        x_r_train = train_data[0][index_shuf_r]
        t_r_train = train_data[1][index_shuf_r]
        u_b_train = train_data[2][index_shuf_b]
        x_b_train = train_data[3][index_shuf_b]
        t_b_train = train_data[4][index_shuf_b]
        u_i_train = train_data[5][index_shuf_i]
        x_i_train = train_data[6][index_shuf_i]
        t_i_train = train_data[7][index_shuf_i]
        train_data_new = [x_r_train, t_r_train, u_b_train,
                          x_b_train, t_b_train, u_i_train, x_i_train, t_i_train]
        train_data = train_data_new
        loss = model.train_step(train_data)
        val_loss = model.val_step(val_data)
        accuracy = model.accuracy_step(val_data)
        epochs.set_postfix(loss=loss, epochs=epoch, val_loss=val_loss)
        losses.append(loss)
        val_losses.append(val_loss)
        acc.append(accuracy)
        if epoch % 100 == 0:
            plot1dgrid_real(lb, ub, N_plotting, model, epoch)
        if epoch % 1000 == 0:
            torch.save(model.net.state_dict(), f"results/model_{epoch}.pt")
        # Plot_losses
        plot_loss(losses, val_losses, acc)


def train_rnn(model, train_data, val_data, epochs):
    epochs = tqdm(range(epochs), desc="Training")
    losses = []
    val_losses = []
    for epoch in epochs:
        # Shuffle train_data
        index_shuf = torch.randperm(train_data[0].shape[0])
        train_data_new = [train_data[0][index_shuf], train_data[1][index_shuf], train_data[2][index_shuf], train_data[3][index_shuf], train_data[4][index_shuf], train_data[5][index_shuf], train_data[6][index_shuf], train_data[7][index_shuf],
                          train_data[8][index_shuf], train_data[9][index_shuf], train_data[10][index_shuf], train_data[11][index_shuf], train_data[12][index_shuf], train_data[13][index_shuf], train_data[14][index_shuf], train_data[15][index_shuf]]
        train_data = train_data_new
        loss = model.train_step_rnn(train_data)
        val_loss = model.val_step_rnn(val_data)
        epochs.set_postfix(loss=loss, epochs=epoch, val_loss=val_loss)
        losses.append(loss)
        val_losses.append(val_loss)
        if epoch % 100 == 0:
            plot1dgrid_real(lb, ub, N_plotting, model, epoch, True)
        if epoch+1 % 1000 == 0:
            model.net.save_weights(f'weights/weights_{epoch}')
        plot_loss(losses, val_losses)


if __name__ == '__main__':
    # pour utiliser le gpu au lieu de cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    with_rnn = False
    net = PINN(with_rnn=with_rnn)
    # net._model_summary()
    N_i, N_b, N_r = DEFAULT_CONFIG['N_i'], DEFAULT_CONFIG['N_b'], DEFAULT_CONFIG['N_r']
    l_b, u_b = DEFAULT_CONFIG['l_b'], DEFAULT_CONFIG['u_b']
    t_i, x_i, u_i, t_b, x_b, u_b, t_r, x_r = define_points(
        N_i, N_b, N_r, l_b, u_b)
    x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i = normalize_data(x_r, t_r,
                                                            u_b, x_b, t_b,
                                                            u_i, x_i, t_i)
    plot_training_points(t_i.data.numpy(),
                         t_b.data.numpy(),
                         t_r.data.numpy(),
                         x_i.data.numpy(),
                         x_b.data.numpy(),
                         x_r.data.numpy(),
                         u_i.data.numpy(),
                         u_b.data.numpy())
    if with_rnn:
        x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i = all_data_to_sequences(x_r, t_r,
                                                                       u_b, x_b, t_b,
                                                                       u_i, x_i, t_i, seq_len=10)
        x_r_label, t_r_label, u_b_label, x_b_label, t_b_label, u_i_label, x_i_label, t_i_label = all_data_to_label(x_r, t_r,
                                                                                                                   u_b, x_b, t_b,
                                                                                                                   u_i, x_i, t_i)
    if not with_rnn:
        train_data, val_data = val_split(
            x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i, split=0.1)
    else:
        train_data, val_data, train_data_labels, val_data_labels = val_split_with_labels(
            x_r, t_r, u_b, x_b, t_b, u_i, x_i, t_i, split=0.1)
        train_data = train_data + train_data_labels
        val_data = val_data + val_data_labels
    lb = [-1, -1]
    ub = [1, 1]
    N_plotting = DEFAULT_CONFIG['N_plotting']
    epochs = DEFAULT_CONFIG['epochs']

    with torch.backends.cudnn.flags(enabled=False):
        if with_rnn:
            train_rnn(net, train_data, val_data, epochs=epochs)
        else:
            train(net, train_data, val_data, epochs=epochs)

    net.net.load_state_dict(torch.load("model_9000.pt"))
    plot1dgrid_real(lb, ub, N_plotting, net, 10000, show=True)
