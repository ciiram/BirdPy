import numpy as np
import scipy as sp
import pylab as pb
import sys
from scipy import signal
from scikits.audiolab import wavread
from scikits.audiolab import wavwrite
import time
import bob
import bird_song_segmentation_functions as  bss_funcs

'''
This code produces the bird song spectrogram of a particular bird song recording.

'''

#MFCC parameters
win_length_ms = 25 # The window length of the cepstral analysis in milliseconds
win_shift_ms = 10 # The window shift of the cepstral analysis in milliseconds
n_filters = 41 # The number of filter bands
n_ceps = 12 # The number of cepstral coefficients
f_min = 200. # The minimal frequency of the filter bank
f_max = 8000. # The maximal frequency of the filter bank
delta_win = 2 # The integer delta value used for computing the first and second order derivatives
pre_emphasis_coef = 0.97 # The coefficient used for the pre-emphasis
dct_norm = True # A factor by which the cepstral coefficients are multiplied 
mel_scale = True # Tell whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale
search_win=500# search for a change point over 500ms


if len(sys.argv) != 2:  
	sys.exit("Usage: run plot_single_file.py [file_name (eg PC1_20090606_050012_0010)]")

x=wavread('../Data/mlsp_contest_dataset/essential_data/src_wavs/'+sys.argv[1]+'.wav')
fs=float(x[1])
x1=x[0]

pb.figure()
Pxx, freqs, t, plot = pb.specgram(
    x1,
    NFFT=512, 
    Fs=fs, 
    noverlap=int(512 * 0.4))
pb.xlabel('Time (s)')
pb.ylabel('Frequency (Hz)')



#tests on high pass filtering
order=10
fc=2000.
b, a = sp.signal.butter(order, fc / (fs/2.), btype='high')
h, w = sp.signal.freqz(b,a)

pb.figure()
pb.title('Digital filter frequency response')
pb.semilogy(h*((fs/2.)/np.pi), np.abs(w), 'b')
pb.ylabel('Amplitude (dB)')
pb.xlabel('Frequency (Hz)')
pb.grid()
pb.legend()

#filter
x2 = signal.lfilter(b, a, x1)
pb.figure()
Pxx, freqs, t, plot = pb.specgram(
    x2,
    NFFT=512, 
    Fs=fs, 
    noverlap=int(512 * 0.4))
pb.xlabel('Time (s)')
pb.ylabel('Frequency (Hz)')

wavwrite(x2,'test.wav',fs)


c = bob.ap.Ceps(fs, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
c.with_energy=False
mfccs = c(x2*2**15)#normalize to integer
change_points= bss_funcs.change_point_detect(mfccs,search_win,win_shift_ms)
current_num_clust,new_seg_boundary,new_clust_label_2=bss_funcs.agglomerative_clustering(mfccs,change_points)

pb.figure()
Pxx, freqs, t, plot = pb.specgram(
    x2,
    NFFT=512, 
    Fs=fs, 
    noverlap=int(512 * 0.4))

for i in range(len(change_points)-1):
	t1=(change_points[i]*win_shift_ms+0.5*win_length_ms)*0.001
	t2=(change_points[i+1]*win_shift_ms+0.5*win_length_ms)*0.001
	#pb.hlines(3000,t1,t2,linewidth=4,color=pb.cm.gist_ncar(np.random.random()))#np.random.random()
	pb.vlines(t2,fs*.125,fs*.25)#linewidth=2)
pb.xlabel('Time (s)')
pb.ylabel('Frequency (Hz)')



pb.figure()
Pxx, freqs, t, plot = pb.specgram(
    x2,
    NFFT=512, 
    Fs=fs, 
    noverlap=int(512 * 0.4))

for i in range(len(new_seg_boundary)-1):
	t1=(new_seg_boundary[i]*win_shift_ms+0.5*win_length_ms)*0.001
	t2=(new_seg_boundary[i+1]*win_shift_ms+0.5*win_length_ms)*0.001
	#pb.hlines(3000,t1,t2,linewidth=4,color=pb.cm.gist_ncar(np.random.random()))#np.random.random()
	pb.vlines(t2,fs*.125,fs*.375)#linewidth=2)
	pb.text(0.5*(t1+t2),fs*.25,str(new_clust_label_2[i]),fontsize=8)

pb.xlabel('Time (s)')
pb.ylabel('Frequency (Hz)')

pb.show()



