# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:04:08 2023

@author: Bcsic_abcd
"""

from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np


path = r'D:\AI-A job\20230201_FFT\Wave'
samplerate, data = wavfile.read(path+'./bad_partial.wav')

data = np.array([i for i in data])
#plt.plot(data)
# 1/32768 

N_len = len(data)

dt = 1/samplerate
time = np.linspace(0, len(data)-1, len(data))
NFFT = 2048*75
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

# plt.plot(myfreq,myspec,linewidth=0.5)
# plt.xlim([15,11000])


myspec[0:sum(np.array(myfreq)<20)]=0

idx = np.argmax(myspec)
print(myfreq[idx])



from scipy.signal import savgol_filter
myspec_hat = savgol_filter(myspec, 200, 3) 
plt.plot(myfreq,myspec)
plt.plot(myfreq,myspec_hat)


import mo





###
import sys

path = r'D:\AI-A job\20230201_FFT\NEW_UI\apui_beta\apui_beta_0201\widgets'
sys.path.append(path)

import training as mytr

aa = mytr.Training.get_models(1)









