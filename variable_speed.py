import torch 
import torch.nn as nn

def c_fun(x,t):
    #Parabolic profile
    c = 4*x**2-4*x+3
    return c