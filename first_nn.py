import numpy as np
import matplotlib.pyplot as plt
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

# Define the network
class NN(nn.Module):
    def __init__(self,num_layers=3,num_neurons=128,dimension=2):
        super(NN,self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dimension,num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons,num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons,num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons,10),
        )

    def forward(self,x):
        last_layer = self.model(x)
        return F.log_softmax(last_layer,dim=1)

# Train step
def train_step(model,loss_fn,opt,x,y):
    opt.zero_grad()
    out = model(x)
    loss = loss_fn(out,y)
    loss.backward()
    opt.step()
    return loss.item(),out

def train(model,loss_fn,opt,train_loader,epochs=10):
    for epoch in range(epochs):
        correct = 0
        bar = tqdm(train_loader)
        for x,y in bar:
            loss,out = train_step(model,loss_fn,opt,x,y)
            bar.set_description(f"Epoch {epoch} Loss {loss}")
            out = torch.argmax(out,dim=1)
            correct += (out == y).float().sum()
        print(f"Epoch {epoch} Loss {loss}, Accuracy {correct/len(train_loader.dataset)}")
    
#Load data
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transforms)
test_dataset = datasets.MNIST(root='./data',train=False,download=True,transform=transforms)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)

# Define the network
model = NN(dimension=28*28)
loss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=1e-2)
train(model,loss,opt,train_loader,epochs=1)

im = train_dataset[1][0].reshape(28,28)
im = im.view(1,28*28)
out = model(im)
pred = torch.argmax(out,dim=1)
plt.imshow(im.reshape(28,28))
plt.title(f"Prediction {pred.item()}")
plt.show()

