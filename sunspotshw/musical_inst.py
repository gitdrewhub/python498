# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 16:40:54 2017

@author: allee
"""
import numpy as np
import matplotlib.pyplot as plt

#read in musical instrument



with open('sunspots.txt') as f:
    musicdata = []
    for line in f:
        line = line.split() # to deal with blank
        if line:            # lines (ie skip them)
            line = [int(i) for i in line] #can read in more than 1 data per line
            musicdata.append(line)

musicdata = np.array(musicdata)
#musicdata = musicdata.astype(float)
mdata = np.ndarray.flatten(musicdata)

time_step = 1/44100 # 44100 samples per second
plt.plot(mdata)
plt.xlabel('time (s)')
plt.ylabel('signal')
plt.show()

#perform a fast Fourier transform
from scipy import fftpack
sample_freq = fftpack.fftfreq(mdata.size, d=time_step)
mdata_fft = fftpack.fft(mdata)

#Since the signal is real, just plot the postive frequencies
pidxs = np.where(sample_freq > 0)
freqs = sample_freq[pidxs]
power = np.square(np.abs(mdata_fft)[pidxs])
#powerdB = np.log10(power)
plt.plot(freqs[1:1000],power[1:1000])
plt.xlabel('frequency (Hz)')
plt.ylabel('amplitude')
plt.show()

