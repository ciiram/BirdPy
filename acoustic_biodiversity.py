import PIL
import numpy as np
import scipy as sp
import pylab as pb
from scipy import signal
from scikits.audiolab import wavread
import time
from matplotlib import gridspec

def spectral_ent(x,N):

	'''
	This function compute the spectral entropy
	The inputs are:
		x: the sampled waveform
		N: Bin length
	It returns the spectral entropy computed according to 
		@article{sueur2008rapid,
		  title={Rapid acoustic survey for biodiversity appraisal},
		  author={Sueur, J{\'e}r{\^o}me and Pavoine, Sandrine and Hamerlynck, Olivier and Duvail, St{\'e}phanie},
		  journal={PLoS One},
		  volume={3},
		  number={12},
		  pages={e4065},
		  year={2008},
		  publisher={Public Library of Science}
		}
	'''
	num_block=int(np.floor(float(len(x))/N))
	spec=np.zeros((N/2+1,num_block), dtype=complex)
	

	for i in range(num_block):
		spec[:,i]=np.fft.fft(x[i*N:(i+1)*N])[0:N/2+1]

	Sabs=np.abs(spec)
	
	
	Sf=np.mean(Sabs,1)
	Sf=Sf/np.sum(Sf)

	

	return -np.sum(Sf[Sf!=0]*np.log2(Sf[Sf!=0]))/np.log2(len(Sf))

def mask_spec_ent(Sf,mask):
	'''
	This function computes the spectral entropy given the spectrogram and the 
	associated mask with bird/non bird sound classification
	'''
	Sf=Sf*mask
	Sf=np.mean(Sf,1)
	Sf=Sf/np.sum(Sf)

	return -np.sum(Sf[Sf!=0]*np.log2(Sf[Sf!=0]))/np.log2(len(Sf))
	

	


def temporal_ent(x):

	'''
	This function computes the temporal entropy
	The inputs are:
		x: the sampled waveform
		
	It returns the temporal entropy computed according to 
		@article{sueur2008rapid,
		  title={Rapid acoustic survey for biodiversity appraisal},
		  author={Sueur, J{\'e}r{\^o}me and Pavoine, Sandrine and Hamerlynck, Olivier and Duvail, St{\'e}phanie},
		  journal={PLoS One},
		  volume={3},
		  number={12},
		  pages={e4065},
		  year={2008},
		  publisher={Public Library of Science}
		}
	'''

	#hilbert transfrom
	y=sp.signal.hilbert(x)
	yabs=np.abs(y)
	At=yabs/np.sum(yabs)

	return -np.sum(At[At!=0]*np.log2(At[At!=0]))/np.log2(len(At))

def spec(x,N,alpha=0):

	'''
	This function computes the magnitude spectrum of the file
	The inputs are:
		x: The sampled audio
		N: Block length in samples
		alpha: overlap, default is 0 (no overlap)
	The function returns the magnitude spectrum
	'''
	Na=(1-alpha)*N
	num_block=int((float(len(x))/N-1)/(1-alpha))
	spec=np.zeros((N/2+1,num_block), dtype=complex)
	

	for i in range(num_block):
		spec[:,i]=np.fft.fft(x[i*Na:i*Na+N])[0:N/2+1]

	Sabs=np.abs(spec)
	
	return Sabs


file1=open('mlsp_contest_dataset/supplemental_data/segment_rectangles.txt',"r")#File containing segmentation of the spectrograms according to bird/non bird sounds
#Obtain the segments
line1=file1.readline()
line1=file1.readline()
s1=line1.split(',')
segs=np.zeros(6)
for i in range(6):
	segs[i]=int(s1[i])
num_segs=1

while file1:
	
	line1=file1.readline()
	s1=line1.split(',')
	
	if len(line1)==0:
		break
	temp_segs=np.zeros(6)
	for i in range(6):
		temp_segs[i]=int(s1[i])
	segs=np.concatenate((segs,temp_segs))
	num_segs+=1
	
file1.close()

segs=np.reshape(segs,(num_segs,6))
file1=open('num_species_ids.txt',"r")

N=512#block size for the spectrogram

res=np.array([])
f=1
while file1:
	
	line1=file1.readline()
	s1=line1.split(' ')
	if len(line1)==0:
		break
	x=wavread('mlsp_contest_dataset/essential_data/src_wavs/'+s1[2].split('\n')[0]+'.wav')
	Sf=spec(x[0],N,.75)#spectrogram
	mask=np.zeros(Sf.T.shape)#mask indicating bird sound regions of the spectrogram
	
	audio_mask=np.zeros(len(x[0]))#mask indicating bird sound samples
	#get the segments of the spectrogram with bird sounds
	for i in range(num_segs):
		if segs[i,0]==int(s1[0]):
						
			mask[segs[i,2]:segs[i,3],segs[i,4]:segs[i,5]]=1
			#convert the x-coordinate of the spectrogram to a sample number of the audio file
			#sample=time*frequency
			#window=32ms,overlap=75%
			#time=(N-1)*(1-alpha)*window+window/2
			n1=int((16.0+(segs[i,2]-1)*8.0)/(1.0/16))
			n2=int((16.0+(segs[i,3]-1)*8.0)/(1.0/16))
			audio_mask[n1:n2]=1
			
	
	#compute acoustic index with and without segmentation
	#Check that the audio has some bird sound
	if np.max(mask)>0:
		entropy_res=np.array([s1[1],spectral_ent(x[0],N)*temporal_ent(x[0]),mask_spec_ent(Sf,mask.T)*temporal_ent(x[0]*audio_mask)], dtype=float)
		res= np.vstack([res, entropy_res]) if res.size else entropy_res
	print 'Processing file:',f
	f+=1

	
	
	

np.savetxt('res.txt',res)


	
