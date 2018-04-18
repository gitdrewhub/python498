'''
ALUC2_HW13 - Andrew Luc, Homework 13


'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Constants
v = 100
L = 1
d = 0.1
C = 1
sigma=0.3
N = 100
a = L / N
h = 1e-7
epsilon = h / 1000



tx = [0.001 , 0.003, 0.005, 0.007, 0.009, 0.011, 0.1]
tend = tx[np.size(tx)-1] + epsilon

x = np.linspace(0,L, N+1)
psi = C*(x*(L-x)/(L**2))*np.exp(-(x-d)**2/(2*sigma**2))



t = 0.0

phi = np.zeros(N+1,float)
while t < tend:

    # calculate the new values of T
    phi[1:N] = phi[1:N]+(h*psi[1:N])
    psi[1:N] = psi[1:N]+((h*v**2/a**2)*(phi[2:N+1]+phi[0:N-1]-2*phi[1:N]))
    t += h

    # make plots at the given time
    for i in range(0, np.size(tx)):
        if abs(t - tx[i]) < epsilon:
            print(tx[i])
            plt.plot(phi)


plt.xlabel("x")
plt.ylabel("T")
plt.show()


print(tx[0])
print(tx[np.size(tx)-1])