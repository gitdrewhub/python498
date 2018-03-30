import numpy as np
import matplotlib.pyplot as plt


## A
months,spots = np.loadtxt('sunspots.txt', delimiter = '	', usecols=(0,1), unpack = True)

months = months.astype(int)



#time_step = 1/44100 # 44100 samples per second

plt.plot(months, spots)
plt.xlabel('time (months)')
plt.ylabel('Sun Spots')
plt.show()

cycle = 148-10 # Estimate via speculation, finding the difference in time between first two peaks
print(cycle)

## B
#perform a fast Fourier transform
from scipy import fftpack
sample_freq = fftpack.fftfreq(spots.size)#, d=time_step)
spots_fft = fftpack.fft(spots)

#Since the signal is real, just plot the postive frequencies
pidxs = np.where(sample_freq > 0)
freqs = sample_freq[pidxs]
power = np.square(np.abs(spots_fft)[pidxs])
#powerdB = np.log10(power)
plt.plot(freqs[1:1000],power[1:1000])
plt.xlabel('k')
plt.ylabel('|ck|^2')
plt.show()

cycle = 1/0.007636  #taking the inverse of the peak frequency
print(cycle)