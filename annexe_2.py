# imports
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
from ipywidgets import *

f = lambda x : np.sin((x*np.pi*2)/10)

t= 10 #t temps final
x= 10 #x longueur du fil
# Amount of elements
N = 10
# Amount of nodes
n = N - 1
# Element size
h = x/N
# La vitesse de propagation
c = 0.4
# Delta time
nx,nt = 100,100
dx=(x/nx) # dx pas spatial
dt=(t/nt) #dt pas temporel
X=np.arange(0,t+dt,dt) #Vecteurs allant de dt à
Y=np.arange(0,x+dx,dx)
U_0= np.array(list(map(f,np.arange(h,x,h)))) #Condition initiale u(x,0),sinusoide
U_1= np.array(list(map(f,np.arange(h,x,h)))) #Condition initiale u(x,-dt), sinusoide
#nt= 100
Resultat= []
# Amount of iterations
iterations = 20000
# Stepsize
# stepSize = 100
# Matrix construction
# Time coefficient matrix
A = np.zeros((n, n))
# A[0, 0] = 1; # Left boundary
# A[N, N] = 1; # Right boundary
for i in range(n):
    for j in range(n):
        if(i==j):
            A[i, j] = (2.0/3.0)*h
        if(abs(i-j) == 1):
            A[i, j] = (1.0/6.0)*h
# Space coefficient matrix
B = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if(i==j):
            B[i, j] = (2.0/h)
        if(abs(i-j)== 1):
            B[i, j] = -(1.0/h)
# A single time step
invA = linalg.inv(A)
M = -(c**2)*dt*invA.dot(B)
U_n = U_0
U_n_1 = U_1
Resultat= []
D=np.concatenate((np.array([0]),U_0),axis = 0)
D=np.concatenate((D,np.array([0])),axis = 0)
Resultat.append(D.tolist())
for i in range(nt):
    U__n= np.dot(M+ 2*np.identity(n),U_n)- U_n_1
    U_n_1=U_n
    U_n=U__n
    D=np.concatenate((np.array([0]),U_n),axis = 0)
    D=np.concatenate((D,np.array([0])),axis = 0)
    Resultat.append(D.tolist())
Resultat = np.array(Resultat)

X=np.arange(0,t+dt,dt)
Y=np.arange(0,x+h,h)

X, Y = np.meshgrid(Y,X)
#X est l'axe spatial
#Y est l'axe temporel

ax = plt.axes(projection = '3d')
ax.plot_surface(X,Y,Resultat,cmap='plasma')
plt.show()


