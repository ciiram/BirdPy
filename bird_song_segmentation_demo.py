import numpy as np
import scipy as sp
import pylab as pb
from scipy import signal
from scikits.audiolab import wavread
import time
import bob
import bird_song_segmentation_functions as  bss_funcs

'''
This code produces the bird song spectrogram plots in the paper
'''


rec1='PC4_20090804_070000_0020'#rec_id=194, labels=[17]  Stellar's Jay
rec2='PC1_20100705_070002_0010'#rec_id=13, labels=[15,17] Warbling Vireo and Stellar's Jay

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


#Plot the spectrograms corresponding to rec1 and rec2

x=wavread('../Data/mlsp_contest_dataset/essential_data/src_wavs/'+rec1+'.wav')
fs=float(x[1])
x2=x[0]

pb.figure()
Pxx, freqs, t, plot = pb.specgram(
    x2,
    NFFT=512, 
    Fs=fs, 
    noverlap=int(512 * 0.4))
pb.xlabel('Time (s)')
pb.ylabel('Frequency (Hz)')
pb.savefig('rec1.png',dpi=300)


c = bob.ap.Ceps(fs, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
c.with_energy=False
mfccs = c(x[0]*2**15)#normalize to integer
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
pb.savefig('rec1_seg.png',dpi=300)


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
pb.savefig('rec1_clust.png',dpi=300)

x=wavread('../Data/mlsp_contest_dataset/essential_data/src_wavs/'+rec2+'.wav')
fs=float(x[1])
x2=x[0]

pb.figure()
Pxx, freqs, t, plot = pb.specgram(
    x2,
    NFFT=512, 
    Fs=fs, 
    noverlap=int(512 * 0.4))
pb.xlabel('Time (s)')
pb.ylabel('Frequency (Hz)')
pb.savefig('rec2.png',dpi=300)

c = bob.ap.Ceps(fs, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
c.with_energy=False
mfccs = c(x[0]*2**15)#normalize to integer
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
pb.savefig('rec2_seg.png',dpi=300)

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
pb.savefig('rec2_clust.png',dpi=300)

pb.show()
