import matplotlib.pyplot as plt
import numpy as np

# 'Solarize_Light2'

t= 10 #t temps final
nt = 10000
x= 10 #x longueur du fil
nx = 100 #nx , nt nombres de pas interieurs
dx=(x/nx) # dx pas spatial
dt=(t/nt) #dt pas temporel
c= 0.5 # c célerité de l'onde
alphacarre = ((c*dt)/dx)**2
T=5
f = lambda x : np.sin((x*np.pi*2)/10)

X=np.arange(0,t+dt,dt) #Vecteurs allant de dt
Y=np.arange(0,x+dx,dx)
U_0= np.array(list(map(f,np.arange(dx,x,dx)))) #Condition initiale u(x,0),sinusoide
U_1= np.array(list(map(f,np.arange(dx,x,dx)))) #Condition initiale u(x,-dt), sinusoide

L = np.zeros(nx-1) #vecteur limite
M = np.zeros((nx-1, nx-1));
for i in range(0,nx-1):
    for j in range(0,nx-1):
        if(i==j):
            M[i, j] = -2.0;
        if(abs(i-j) == 1):
            M[i, j] = (1.0);
Resultat= []
D=np.concatenate((np.array([0]),U_0),axis = 0)
D=np.concatenate((D,np.array([0])),axis = 0)
Resultat.append(D.tolist())
U_n = U_0
U_n_1 = U_1
B= 2*np.identity(nx-1) + alphacarre*M
C =alphacarre*M
for i in range(nt):
    U__n=  np.dot(B,U_n) - U_n_1 + np.dot(C,L)
    U_n_1=U_n
    U_n=U__n
    D=np.concatenate((np.array([0]),U_n),axis = 0)
    D=np.concatenate((D,np.array([0])),axis = 0)
    Resultat.append (D.tolist())
    if i * dt <= T and T<= (i+1)*dt :
        U_T = D
Resultat = np.array(Resultat)

#X est l'axe temporel
#Y est l'axe spatial
X, Y = np.meshgrid(Y,X)
# X est l'axe spatial
# Y est l'axe temporel
ax = plt.axes(projection = '3d')
ax.plot_surface(X,Y,Resultat,cmap='plasma')
plt.show()

