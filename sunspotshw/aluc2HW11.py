import numpy as np
import matplotlib.pyplot as plt


# Load sunspot data
months,spots = np.loadtxt('sunspots.txt', delimiter = '	', usecols=(0,1), unpack = True)
months = months.astype(int)

# Plot the Sun Spot signal
plt.plot(months, spots)
plt.xlabel('time (months)')
plt.ylabel('Sun Spots')
plt.show()

# Estimate via speculation, finding the difference in time between first two peaks(Time Domain)
cycle = 148-10
print("Sunspot Period (Initial Estimate) : " + str(cycle))


## B
# Perform a fast Fourier transform on the signal
from scipy import fftpack
sample_freq = fftpack.fftfreq(spots.size)
spots_fft = fftpack.fft(spots)

# Omit negative frequencies
pidxs = np.where(sample_freq > 0)
freqs = sample_freq[pidxs]
power = np.square(np.abs(spots_fft)[pidxs])

# Plotting the signal in the frequency domain
plt.plot(freqs[1:1000],power[1:1000])
plt.xlabel('k')
plt.ylabel('|ck|^2')
plt.show()

fft_cycle = 1/0.007636  #taking the inverse of the peak frequency
print("Sunspot Period (From fft) : " + str(fft_cycle))