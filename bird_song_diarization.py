import numpy as np
import scipy as sp
import pylab as pb
from scipy import signal
from scikits.audiolab import wavread
import time
import bob
import bird_song_segmentation_functions as  bss_funcs
from scipy.stats.stats import pearsonr


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

numspeciesVSnumclust=np.array([])
file1=open('num_species_ids.txt',"r")

while file1:
	
	line1=file1.readline()
	s1=line1.split(' ')
	if len(line1)==0:
		break
	
	x=wavread('mlsp_contest_dataset/essential_data/src_wavs/'+s1[2].split('\n')[0]+'.wav')
	fs=float(x[1])#Sampling frequency
	c = bob.ap.Ceps(fs, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
	c.with_energy=False
	mfccs = c(x[0]*2**15)#normalize to integer
	change_points=bss_funcs.change_point_detect(mfccs,search_win,win_shift_ms)

	if len(change_points)>2:#Only cluster if a change point was discovered within the audio
		clust_res=bss_funcs.agglomerative_clustering(mfccs,change_points)
	else:
		clust_res=(1,change_points,np.array([0]))


	numspeciesVSnumclust= np.vstack([numspeciesVSnumclust, np.array([float(s1[1]),clust_res[0]])]) if numspeciesVSnumclust.size else np.array([float(s1[1]),clust_res[0]])

	print s1[0],s1[2].split('\n')[0],s1[1],clust_res[0]
	
	
np.savetxt('numspeciesVSnumclust_nohpf.txt',numspeciesVSnumclust)
file1.close()

