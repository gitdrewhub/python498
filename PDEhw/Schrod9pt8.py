# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 07:52:10 2017

@author: allee
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

# constants
h = 1e-18 # s, time step
hbar = 1.055e-34 # Planck's constant in Js/rad
m = 9.109e-31 # mass of electron in kg
sig = 1e-10 # std dev wave function in m
L = 1e-8 # width of box in m 
k = 5e10 # wave vector in m-1
N = 1000 # number of spatial steps
a = L/N
x0 = L/2.0

# initial wave function
psi = np.empty(N+1,complex)
psinew = np.empty(N+1,complex)
psi[0] = 0.0
psi[N] = 0.0
for ii in range(1,N):
    x = ii*a
    exponent = -(x-x0)*(x-x0)/(2*sig*sig)
    psi[ii] = np.exp(exponent)*np.exp(1j*k*x)
 
#create A matrix   
A = np.zeros((N+1,N+1),complex)
a1 = 1.0 + h*1j*hbar/(2*m*a*a)
a2 = -h*1j*hbar/(4*m*a*a)
A = np.eye(N+1,N+1,k=-1)*a2 + np.eye(N+1,N+1)*a1 + np.eye(N+1,N+1,k=1)*a2
    
#create B matrix
B = np.zeros((N+1,N+1),complex)
b1 = 1.0 - h*1j*hbar/(2*m*a*a)
b2 = h*1j*hbar/(4*m*a*a)
B = np.eye(N+1,N+1,k=-1)*b2 + np.eye(N+1,N+1)*b1 + np.eye(N+1,N+1,k=1)*b2


#main loop
t = 0.0
tend = 1000.0*h
epsilon = 1e-20
t1 = 1000e-18

while t < tend:
    #calculate the new values of psi
    v = B.dot(psi.T)
    psinew = solve(A,v)
    psi = psinew
    t += h
temp = abs(psi)
plt.plot(temp*temp)     
#plt.xlabel("x")
#plt.ylabel("psi")
plt.show()


