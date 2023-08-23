# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:36:18 2023

@author: Bcsic_abcd
"""


# input_dict = Wav_result
# origin_wav = utils.lps_rebuild(input_dict['origin_lps'],input_dict['origin_info'])
# extract_wav = utils.lps_rebuild(input_dict['atten_lps'],input_dict['origin_info'])

# plt.plot(origin_wav)
# plt.plot(extract_wav)

# sig = origin_wav
# fs = input_dict['sampling_rate']


import sys
sys.path.append('..')

# To get inline plots (specific to Jupyter notebook)

# Import numpy
import numpy as np
# Import plot function
import matplotlib.pyplot as plt
# Import mosqito functions
from mosqito.utils import load
# Import spectrum computation tool
from scipy.fft import fft, fftfreq
from mosqito.sq_metrics import loudness_zwst_perseg
from mosqito.sq_metrics import sharpness_din_st
from mosqito.sq_metrics import sharpness_din_perseg
from mosqito.sq_metrics import sharpness_din_from_loudness
from mosqito.sq_metrics import sharpness_din_freq

# Import MOSQITO color sheme [Optional]
from mosqito import COLORS

# To get inline plots (specific to Jupyter notebook)

# Define path to the .wav file
# To be replaced by your own path
path = r"D:\AI-A job\20230201_FFT\Wave\bad_partial.wav"
# load signal
sig, fs = load(path, wav_calib=2 * 2 **0.5)


sharpness = sharpness_din_st(sig, fs, weighting="din")
print("Sharpness = {:.1f} acum".format(sharpness) )

# ref: https://github.com/Eomys/MoSQITo/blob/master/tutorials/tuto_sharpness_din.ipynb




### load 




path = r"D:\AI-A job\20230201_FFT\Cam_result\zero\motor_cat1_12_16_2022_13_44_04\wav"


# load signal
sig, fs = load(path+r'\motor_cat1_12_16_2022_13_44_04_extract.wav', wav_calib=2 * 2 **0.5)


samplerate, data = wavfile.read(path+r'\motor_cat1_12_16_2022_13_44_04_extract.wav')

