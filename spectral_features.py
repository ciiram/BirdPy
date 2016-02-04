import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io.wavfile
import time
import sys
import os
import datetime

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
		spec[:,i]=np.fft.fft(x[i*Na:i*Na+N]*np.hanning(N))[0:N/2+1]

	Sabs=np.abs(spec)
	
	return Sabs

def centroid(Sxx,N,fs,fmin,fmax):
	
	num_frames=Sxx.shape[1]
	centroids=np.zeros(num_frames)
	band_width=np.zeros(num_frames)
	freqs=np.arange(Sxx.shape[0])*fs/float(N)

	for i in range(num_frames):
		centroids[i]=np.sum(Sxx[(freqs>fmin)&(freqs<fmax),i]*freqs[(freqs>fmin)&(freqs<fmax)])/np.sum(Sxx[(freqs>fmin)&(freqs<fmax),i])
		band_width[i]=np.sum(np.abs(freqs[(freqs>fmin)&(freqs<fmax)]-centroids[i])*Sxx[(freqs>fmin)&(freqs<fmax),i])/np.sum(Sxx[(freqs>fmin)&(freqs<fmax),i])

	return centroids,band_width


def spectral_rolloff(Sxx,N,fs,percent):
	num_frames=Sxx.shape[1]
	rolloff=np.zeros(num_frames)
	S=Sxx/np.sum(Sxx,axis=0)
	freqs=np.arange(Sxx.shape[0])*fs/float(N)
	#compute cumulative sum
	CummulativeSum=np.zeros(S.shape)
	for i in range(Sxx.shape[0]):
		CummulativeSum[i,:]=np.sum(S[:i,:],axis=0)

	for i in range(num_frames):
		rolloff[i]=freqs[sum(CummulativeSum[:,i]<percent)]

	return rolloff

def band_energy_ratio(Sxx,N,fs,num_bands):
	num_frames=Sxx.shape[1]
	BER=np.zeros((num_bands,num_frames))
	band_lim=.5**np.arange(num_bands)*N*.5
	band_lim=np.concatenate((np.array([0]),band_lim[::-1]))

	for i in range(num_frames):
		E=sum(Sxx[:,i]**2)
		for j in range(num_bands):
			BER[j,i]=sum(Sxx[band_lim[j]:band_lim[j+1],i]**2)/E


	return BER

def spectral_flux(Sxx):
	num_frames=Sxx.shape[1]
	S=Sxx/np.sum(Sxx,axis=0)
	spectral_flux=np.zeros(num_frames-1)
	for i in range(num_frames-1):
		spectral_flux[i]=sum((S[:,i+1]-S[:,i])**2)

	return spectral_flux

def spectral_flux_bands(Sxx,N,num_bands):
	num_frames=Sxx.shape[1]
	spectral_flux=np.zeros((num_bands,num_frames-1))
	band_lim=.5**np.arange(num_bands)*N*.5
	band_lim=np.concatenate((np.array([0]),band_lim[::-1]))
	S=Sxx/np.sum(Sxx,axis=0)
	spectral_flux=np.zeros((num_bands,num_frames-1))
	for i in range(1,num_frames):
		for j in range(num_bands):
			spectral_flux[j,i-1]=sum((S[band_lim[j]:band_lim[j+1],i]-S[band_lim[j]:band_lim[j+1],i-1])**2)
			

	return spectral_flux
	
		
	
