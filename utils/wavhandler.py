import scipy
from scipy.io import wavfile
import numpy as np
import math
from scipy.fftpack import fft, fftfreq
from mosqito.sq_metrics import sharpness_din_st


class WavHandler():

    def __init__(self,fpath):

        self.sr, self.data = wavfile.read(fpath)
        self.fft_data = {}

    def get_time_domain(self):

        normalized_y = [(ele/2**16.)*2-1 for ele in self.data]
        xt = np.linspace(0,len(self.data)/self.sr, len(self.data))
        return xt, normalized_y

    def get_freq_domain(self):
        
        self.data = np.array([i for i in self.data])
        # 1/32768 
        N_len = len(self.data)

        dt = 1 / self.sr
        time = np.linspace(0, len(self.data)-1, len(self.data))
        NFFT = 2048*100
        offset =int((1-0.95)*NFFT)

        win = np.hanning(NFFT)

        N_win = 1+ int((N_len-NFFT)/offset)

        fft_spectrum = np.zeros(NFFT);
        for i in range(N_win):
            s = i*offset
            idx = [k for k in range(s,s+NFFT,1)]
            sw = self.data[idx]*win
            fft_spectrum = fft_spectrum + abs(fft(sw)*4/NFFT)

        idx = [k for k in range(0,int(NFFT/2),1)]

        myspec = fft_spectrum[idx]
        myfreq = [self.sr*k/NFFT for k in range(0,int(NFFT/2),1)]
        self.fft_data['freq'] = myfreq

        sensor_spectrum_dB = [20*math.log10(i) for i in myspec]
        self.fft_data['amplitude'] = sensor_spectrum_dB

        myspec[0:sum(np.array(myfreq)<20)]=0
        idx = np.argmax(myspec)
        self.fft_data['max_freq'] = myfreq[idx]

        sharpness = sharpness_din_st(self.data, self.sr, weighting="din")
        print("Sharpness = {:.1f} acum".format(sharpness) )
        self.fft_data['sharpness'] = f'{sharpness:.1f}'

        return self.fft_data


