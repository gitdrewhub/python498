import numpy as np
import matplotlib.pyplot as plt



months,spots = np.loadtxt('sunspots.txt', delimiter = '	', usecols=(0,1), unpack = True)

months = months.astype(int)
print(months)

plt.plot(months, spots)
plt.xlabel('time (months)')
plt.ylabel('Sun Spots')
plt.show()