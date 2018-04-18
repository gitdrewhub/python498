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
V1 = 1.0 # voltage at top wall
target = 1e-6 # target accuracy

# Create arrays to hold potential values
phi = np.zeros([M+1,M+1],float)
phi[0,:] = V1
phiprime = np.empty([M+1,M+1],float)

#Main loop
delta = 1.0
while delta > target:
    #calculate new values of the potential
    for i in range(M+1):
        for j in range(M+1):
            if i==0 or i==M or j==0 or j==M:
                phiprime[i,j] = phi[i,j]
            else:
                phiprime[i,j] = (phi[i+1,j] + phi[i-1,j] \
                                + phi[i,j+1] + phi[i,j-1])/4
    
    # calculate maximum difference from old values
    delta = np.max(np.abs(phi-phiprime))
    
    # swap the two arrays
    phi,phiprime = phiprime,phi
    

plt.imshow(phi,cmap=cm.hot)
plt.colorbar()
plt.show()