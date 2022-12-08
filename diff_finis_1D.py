import numpy as np
import matplotlib.pyplot as plt

# Implementation 1D differences finis:

L = 1;   # length of the string element
t = np.linspace(0.,1,101)   # time
x = np.linspace(0., L, 51) # space
c = 2;   # wavespeed parameter
# c = 2*(1 + 2*(x**2-x*L));   # wavespeed parameter

dx = x[1] - x[0]
dt = t[1] - t[0]
C = c*dt/dx       # Courant number
Nt = len(t) - 1   # last time index
Nx = len(x) - 1   # last space index

# Defining the displacement
u = np.zeros((Nx+1,Nt+1))

# Initial conditions (respecting boundary conditions) (t = 0)
# u[:,0] = np.sin(x/L*2*np.pi)
# u[:,0] = np.sin(x/L*np.pi)
u[:,0] = 1*(np.sin(np.pi*x) + 0.5*np.sin(4*np.pi*x))
u[0,:]  = 0  # boundary conditions
u[Nx,:] = 0  # boundary conditions

# Special formula for the 1º Step (lack of n-1 term) (t = 1)
for i in range(1,Nx):
    u[i,1] = u[i,0] - 0.5*(C**2)*(u[i+1,0] - 2*u[i,0] + u[i-1,0])

# For every other instant (t > 1 until 'inf')
for n in range(1, Nt):
    # Update all inner mesh points at time t[n+1]
    for i in range(1, Nx):
        u[i,n+1] = 2*u[i,n] - u[i,n-1] + (C**2)*(u[i+1,n] - 2*u[i,n] + u[i-1,n])

# Plot
X, T = np.meshgrid(x, t)
map_plot = plt.contourf(np.transpose(T), np.transpose(X),u)
cbar = plt.colorbar(map_plot)
plt.title('1D constant c initial condition response')
plt.show()



