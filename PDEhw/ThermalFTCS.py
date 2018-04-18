# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 21:07:33 2017

@author: allee
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Constants
L = 0.01
D = 4.25e-6
N = 100
a = L/N
h = 1e-4
epsilon = h/1000

Tlo = 0.0
Tmid = 20.0
Thi = 50.0

t1 = 0.01
t2 = 0.1
t3 = 0.4
t4 = 1.0
t5 = 10.0
tend = t5 + epsilon

#create arrays
T = np.empty(N+1,float)
T[0] = Thi
T[N] = Tlo
T[1:N] = Tmid
Tp = np.empty(N+1,float)
Tp[0] = Thi
Tp[N] = Tlo

#main loop
t = 0.0
c = h*D/(a*a)
while t < tend:
    
    #calculate the new values of T
    #for i in range(1,N):
    #   Tp[i] = T[i] + c*(T[i+1]+T[i-1]-2*T[i])
    Tp[1:N] = T[1:N] + c*(T[0:N-1]+T[2:N+1]-2*T[1:N]) # much faster!
    T,Tp = Tp,T
    t += h
    
    # make plots at the given time
    if abs(t-t1)<epsilon:
        plt.plot(T)
    if abs(t-t2)<epsilon:
        plt.plot(T)
    if abs(t-t3)<epsilon:
        plt.plot(T)
    if abs(t-t4)<epsilon:
        plt.plot(T)
    if abs(t-t5)<epsilon:
        plt.plot(T)
        
plt.xlabel("x")
plt.ylabel("T")
plt.show()
