'''
ALUC2_HW14 - Andrew Luc, Homework 14

This code solves for the distance D inside of a sphere of 235U such that a neutron has a uniform probability of causing
a fission. A quick Google search will yield that the critical mass for a 235U sphere of 17cm diameter is 52kg.
We are able to approach this value by:
1)Picking a random position within the 17cm diameter sphere
2)Picking two random directions
3)Picking a random distance between 0 - D
4)Using previously mentioned random position, 2 random directions, and a random distance, calculate 2 new positions
5)Check to see if the positions of the new neutrons are within the sphere
6)If the position of the neutron is within the sphere, add that neutron to the running total to find the ratio of
    reactions to neutrons
7)Repeat for ~1,000,000 neutrons



Running the 'D' for loop with 100,000 neutrons gave a uniform fission probability of 0.9982 at D=8.8cm

###########################

Increasing the number of neutrons for better accuracy to 1,000,000 , we see that D = 8.9cm gives a uniform fission
probability of 0.9969.

###########################

'''

import numpy as np
import math
import matplotlib.pyplot as plt

#Radius of sphere
R = 17/2

#Finding random position
def randPos():
    xyz = np.random.random_sample(3) * 17 - 17 / 2
    Rprime = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)
    #Check if position is within the sphere
    if Rprime > R:
        xyz = np.random.random_sample(3) * 17 - 17 / 2
    return xyz

#Finding random direction
def randDirect():
    phi = np.random.random_sample() * 2 * np.pi
    costheta = np.random.random_sample() * 2 - 1
    theta = math.acos(costheta)
    return [phi, theta]

#Calculating new position given initial position, direction, and distance
def newPos(xyz, phitheta, d):
    x0 = xyz[0]
    y0 = xyz[1]
    z0 = xyz[2]
    phi = phitheta[0]
    theta = phitheta[1]

    x1 = x0 + d * math.sin(theta) * math.cos(phi)
    y1 = y0 + d * math.sin(theta) * math.sin(phi)
    z1 = z0 + d * math.sin(theta)
    return [x1, y1, z1]

dlist=[]
rlist=[]
# Main loop

D=8.9
numer = 0
denom = 0
for neutron in range(1, 1000000, 1):
    d = np.random.random_sample(2) * D
    xyz0 = randPos()
    phitheta0 = randDirect()
    phitheta1 = randDirect()
    xyz1 = newPos(xyz0, phitheta0, d[0])
    xyz2 = newPos(xyz0, phitheta1, d[1])

    Rprime = np.sqrt(xyz1[0] ** 2 + xyz1[1] ** 2 + xyz1[2] ** 2)
    if Rprime < R:
        numer += 2
    Rprime = np.sqrt(xyz2[0] ** 2 + xyz2[1] ** 2 + xyz2[2] ** 2)
    if Rprime < R:
        numer += 2
    denom += 2

ratio=numer/denom
dlist.append(D)
rlist.append(ratio)
print('Optimum D: ' +str(D)+ ' Ratio: '+str(ratio))

'''
# Using a for loop to sweep 'D' in order to find the right D value that corresponds to uniform fission probability

Drange=np.arange(8.6,9.01,0.1)
for i in range(0,len(Drange)):
    D=Drange[i]
    numer = 0
    denom = 0
    for neutron in range(1, 100000, 1):
        d = np.random.random_sample(2) * D      #Finding random distances between 0 - D
        xyz0 = randPos()                        #Finding random initial position in sphere
        phitheta0 = randDirect()                #Finding random directions 
        phitheta1 = randDirect()
        xyz1 = newPos(xyz0, phitheta0, d[0])    #Calculating new positions for two atoms
        xyz2 = newPos(xyz0, phitheta1, d[1])

        # Count the number of escaped 235U atoms
        Rprime = np.sqrt(xyz1[0] ** 2 + xyz1[1] ** 2 + xyz1[2] ** 2)
        if Rprime < R:
            numer += 2
        Rprime = np.sqrt(xyz2[0] ** 2 + xyz2[1] ** 2 + xyz2[2] ** 2)
        if Rprime < R:
            numer += 2
        denom += 2
        
    #Calculate and store fission probabilities
    ratio=numer/denom
    rlist.append(ratio)
    print('D: ' +str(D)+ ' Ratio: '+str(ratio))

#Plotting waveforms
plt.plot(Drange,rlist)
plt.xlabel("D")
plt.ylabel("Chain Rxn Ratio")
plt.show()
'''


