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
cnst = np.pi*np.pi*hbar/(2.0*m*L*L) #problem here not the right k!

# initial wave function
psi = np.empty(N+1,complex)
psinew = np.empty(N+1,complex)
psi[0] = 0.0
psi[N] = 0.0
for ii in range(1,N):
    x = ii*a
    exponent = -(x-x0)*(x-x0)/(2*sig*sig)
    psi[ii] = np.exp(exponent)*np.exp(1j*k*x)
 
realpsi = psi.real
imagpsi =  psi.imag
 
c = np.arange(N+1)

#Discrete sine transform
#Scipy has dst and idst
from scipy import fftpack

alpha = fftpack.dst(realpsi)
eta = fftpack.dst(imagpsi)

#pick a time
t = 1500.0*h
coeffr = np.empty(N+1,float)
coeffr = alpha*np.cos(cnst*c*c*t) - eta*np.sin(cnst*c*c*t)

psiratt = np.empty(N+1,float)
psiratt = fftpack.idst(coeffr)

coeffi = np.empty(N+1,float)
coeffi = alpha*np.sin(cnst*c*c*t) + eta*np.cos(cnst*c*c*t)

psiiatt = np.empty(N+1,float)
psiiatt = fftpack.idst(coeffi)

psiatt = np.empty(N+1,complex)
psiatt = psiratt +1j*psiiatt

temp = abs(psiatt)

plt.plot(temp*temp)  
#plt.plot(imagpsi)   
#plt.xlabel("x")
#plt.ylabel("psi")
plt.show()


