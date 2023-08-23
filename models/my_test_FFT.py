# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 00:41:36 2023

@author: Bcsic_abcd
"""

from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np
import math


path = r'C:\Users\chihk\projects\apuiv2'
samplerate, data = wavfile.read(path+'./recording.wav')

data = np.array([i for i in data])
# 1/32768 

N_len = len(data)

dt = 1/samplerate
time = np.linspace(0, len(data)-1, len(data))
NFFT = 2048*100
offset =int((1-0.95)*NFFT)

win = np.hanning(NFFT)

N_win = 1+ int((N_len-NFFT)/offset)

fft_spectrum = np.zeros(NFFT);
for i in range(N_win):
    s = i*offset
    idx = [k for k in range(s,s+NFFT,1)]
    sw = data[idx]*win
    fft_spectrum = fft_spectrum + abs(fft(sw)*4/NFFT)

idx = [k for k in range(0,int(NFFT/2),1)]

myspec = fft_spectrum[idx]
myfreq = [samplerate*k/NFFT for k in range(0,int(NFFT/2),1)]

sensor_spectrum_dB = [20*math.log10(i) for i in myspec]

plt.plot(myfreq,sensor_spectrum_dB,linewidth=0.3)
plt.xscale('log',base=10) 
plt.ylim(-100,60)
plt.xlim([15,11000])
plt.grid()
plt.show()


##

# plt.psd(data, NFFT=2048*75, Fs=samplerate, noverlap=int((1-0.5)*NFFT), window=win,linewidth=0.5)
# plt.xscale('log',base=10) 
# plt.xlim([15,11000])
# plt.show()
