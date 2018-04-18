# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:12:59 2017

@author: allee
"""

# example 9.1 LaPlace Jacobi

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Constants
M = 100 # grid squares on a side
a = 1 # 1 m grid
eps = 8.854e-12 # just for demonstration, actually 8.854e-12 F/m
k = a**2/(4*eps)
target = 1e-6 # target accuracy
w = 0.9

# Create arrays to hold potential values
phi = np.zeros([M+1,M+1],float)
rho = np.zeros([M+1,M+1],float)
rho[25,25] = 10e-12 # 10 pC/m2
rho[75,75] = -10e-12 # -10 pC/m2 

#Main loop
flag = True
while flag == True:
    flag = False
    #calculate new values of the potential
    for i in range(M+1):
        for j in range(M+1):
            if i==0 or i==M or j==0 or j==M:
                phi[i,j] = phi[i,j] # all edges remain grounded
            else:
                phiold = phi[i,j]
                phi[i,j] = (1+w)*((phi[i+1,j] + phi[i-1,j] \
                                + phi[i,j+1] + phi[i,j-1])/4 + k*rho[i,j]) \
                                -w*phi[i,j]
                diff = np.abs(phi[i,j]-phiold)
                if diff > target: flag = True
    

    

plt.imshow(phi,cmap=cm.hot)
plt.colorbar()
plt.show()