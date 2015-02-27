import numpy as np
import scipy as sp
import pylab as pb
from scipy import signal
from scikits.audiolab import wavread
import time
import bob


def change_point_detect(mfccs,search_win,win_shift_ms):
	'''
	This function takes a sequence of MFCC featurses and determines the locations of change points.
	We implement the algorithm in 

	@inproceedings{chen1998speaker,
	  title={Speaker, environment and channel change detection and clustering via the bayesian information criterion},
	  author={Chen, Scott and Gopalakrishnan, Ponani},
	  booktitle={Proc. DARPA Broadcast News Transcription and Understanding Workshop},
	  pages={8},
	  year={1998},
	  organization={Virginia, USA}
	}

	The inputs are 
		mfccs: The MFCCs derived from the audion
		search_win: The length of the window over which we search for a change point
		win_shift_ms: The frame rate in miliseconds
	The output is
		change_points: A vector containing the frame numbers of the change points
	'''


	min_frame=mfccs.shape[1]+3#we must obtain at least this number of frames to get an invertible covariance matrix
	search_win_frame=search_win/win_shift_ms
	last_change_point=0
	change_points=np.array([0])

	while last_change_point+search_win_frame<mfccs.shape[0]:
		R=np.zeros(search_win_frame)#Maximum likelihood ratio statistic
		BIC=np.zeros(search_win_frame)#Bayesian Information Criterion
		P=0.5*(n_ceps+0.5*n_ceps*(n_ceps+1))*np.log(search_win_frame) #BIC penalty
		for i in range(min_frame,search_win_frame-min_frame):

			cov_all=np.cov(mfccs[last_change_point:last_change_point+search_win_frame,:].T)#Covariance of all frames in window
			cov_1=np.cov(mfccs[last_change_point:i+last_change_point,:].T)#Covariance of frames before hypothesised change
			cov_2=np.cov(mfccs[i+last_change_point:last_change_point+search_win_frame,:].T)#Covariance of frames after hypothesised change
			R[i]=search_win_frame*np.log(np.linalg.det(cov_all))-i*np.log(np.linalg.det(cov_1))-(search_win_frame-i)*np.log(np.linalg.det(cov_2))
			BIC[i]=R[i]-P
		last_change_point+=np.argmax(BIC)
		if np.argmax(BIC)==0:
			search_win_frame+=10#Increase the size of the window if the entire window is homogeneous and no change point is detected
		else:
			search_win_frame=search_win/win_shift_ms
			change_points= np.vstack([change_points,last_change_point]) if change_points.size else last_change_point
	change_points=np.vstack([change_points,mfccs.shape[0]])
	change_points=change_points[:,0]


	return change_points

def agglomerative_clustering(mfccs,change_points):
	'''
	This function performs agglomerative clustering on the initial clusters using the Bayesian information criterion
	The algorithm implemented here is as described in 
	
	@article{tranter2006overview,
	  title={An overview of automatic speaker diarization systems},
	  author={Tranter, Sue E and Reynolds, Douglas A},
	  journal={Audio, Speech, and Language Processing, IEEE Transactions on},
	  volume={14},
	  number={5},
	  pages={1557--1565},
	  year={2006},
	  publisher={IEEE}
	}
	
	The inputs are 
		mfccs: The mfccs derived from the audio
		change_points: The frame numbers at which changes occur, the first and last segment are included

	'''

	#determine the initial clusters
	current_num_clust=len(change_points)-1
	clust_num=np.arange(current_num_clust)
	current_clust_label=np.arange(current_num_clust)
	n_ceps=mfccs.shape[1]

	curr_min_bic=-np.inf
	while (current_num_clust>2)&(curr_min_bic<0):#len(seg_diff)-6:#2:

		#initialize the covariances
		covs=np.zeros((n_ceps,n_ceps,current_num_clust))
		clust_size=np.zeros(current_num_clust)
		current_clust_unique_labels=np.unique(current_clust_label)
		for i in range(current_num_clust):
			#get the segment numbers corresponding to each cluster
			seg_label=clust_num[current_clust_label==current_clust_unique_labels[i]]
			#Now get the corresponding data
			A=np.array([])
			for j in range(len(seg_label)):
				A=np.vstack((A,mfccs[change_points[seg_label[j]]:change_points[seg_label[j]+1],:])) if A.size else mfccs[change_points[seg_label[j]]:change_points[seg_label[j]+1],:]
			
	
			covs[:,:,i]=np.cov(A.T)
			clust_size[i]=A.shape[0]

		#Compute the delta bic that will result if two clusters are merged. We consider all the pairs			
		delta_bics=np.zeros(current_num_clust*(current_num_clust-1)*0.5)
		clust_dist=np.zeros(current_num_clust*(current_num_clust-1)*0.5)

		indx=[]
		k=0
		for i in range(current_num_clust):
			for j in range(i+1,current_num_clust):
				P=0.5*(n_ceps+0.5*n_ceps*(n_ceps+1))*np.log(clust_size[i]+clust_size[j])

				#compute covariance from merging
				seg_label=clust_num[current_clust_label==current_clust_unique_labels[i]]
				#print seg_label
				A=np.array([])
				for l in range(len(seg_label)):
					A=np.vstack((A,mfccs[change_points[seg_label[l]]:change_points[seg_label[l]+1],:])) if A.size else mfccs[change_points[seg_label[l]]:change_points[seg_label[l]+1],:]
				seg_label=clust_num[current_clust_label==current_clust_unique_labels[j]]
				for l in range(len(seg_label)):
					A=np.vstack((A,mfccs[change_points[seg_label[l]]:change_points[seg_label[l]+1],:])) if A.size else mfccs[change_points[seg_label[l]]:change_points[seg_label[l]+1],:]
			
			
				delta_bic=0.5*((clust_size[i]+clust_size[j])*np.log(np.linalg.det(np.cov(A.T)))-clust_size[i]*np.log(np.linalg.det(covs[:,:,i]))-clust_size[j]*np.log(np.linalg.det(covs[:,:,j])))-P
				delta_bics[k]=delta_bic
				clust_dist[k]=(clust_size[i]+clust_size[j])*np.log(np.linalg.det(np.cov(A.T)))-clust_size[i]*np.log(np.linalg.det(covs[:,:,i]))-clust_size[j]*np.log(np.linalg.det(covs[:,:,j]))
				indx.append([current_clust_unique_labels[i],current_clust_unique_labels[j]])
				k+=1

		max_indx=indx[np.argmin(clust_dist)]
		curr_min_bic=np.min(delta_bics)
		#print max_indx	,np.min(delta_bics),np.linalg.det(np.cov(A.T))

		#merge clusters
		current_clust_label[current_clust_label==np.max(max_indx)]=np.min(max_indx)
		current_num_clust-=1


	#get new segment boundaries

	new_seg_boundary=np.array([0])
	new_clust_label=np.array([])
	for i in range(1,len(current_clust_label)):
		if current_clust_label[i]!=current_clust_label[i-1]:
			new_seg_boundary=np.vstack([new_seg_boundary,change_points[i]])
			new_clust_label=np.vstack([new_clust_label,current_clust_label[i-1]]) if new_clust_label.size else np.array([current_clust_label[i-1]])

	if current_clust_label[-1]!=new_clust_label[-1]:#current_clust_label[-2]:
		new_clust_label=np.vstack([new_clust_label,current_clust_label[-1]])
	new_seg_boundary=np.vstack([new_seg_boundary,change_points[-1]])

	#reassign segment names
	current_clust_unique_labels=np.unique(current_clust_label)
	new_current_clust_label=np.zeros(len(current_clust_label))
	new_clust_label=new_clust_label[:,0]
	new_clust_label_2=np.zeros(len(new_clust_label),dtype=int)

	for i in range(len(current_clust_unique_labels)):
		new_current_clust_label[current_clust_label==current_clust_unique_labels[i]]=i
		new_clust_label_2[new_clust_label==current_clust_unique_labels[i]]=i

	return current_num_clust,new_seg_boundary,new_clust_label_2



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
f=1
while file1:
	
	line1=file1.readline()
	s1=line1.split(' ')
	if len(line1)==0:
		break
	print 'Processing file',f,s1[2].split('\n')[0]+'.wav'
	x=wavread('mlsp_contest_dataset/essential_data/src_wavs/'+s1[2].split('\n')[0]+'.wav')
	fs=float(x[1])#Sampling frequency
	c = bob.ap.Ceps(fs, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
	c.with_energy=False
	mfccs = c(x[0]*2**15)#normalize to integer
	change_points=change_point_detect(mfccs,search_win,win_shift_ms)

	if len(change_points)>2:#Only cluster if a change point was discovered within the audio
		clust_res=agglomerative_clustering(mfccs,change_points)
	else:
		clust_res=(1,change_points,np.array([0]))


	numspeciesVSnumclust= np.vstack([numspeciesVSnumclust, np.array([float(s1[1]),clust_res[0]])]) if numspeciesVSnumclust.size else np.array([float(s1[1]),clust_res[0]])
	
	f+=1
np.savetxt('numspeciesVSnumclust.txt',numspeciesVSnumclust)

pb.figure()
pb.plot(numspeciesVSnumclust[:,0],numspeciesVSnumclust[:,1]-1,'bx',markersize=10)
pb.xlabel('Number of Species')
pb.ylabel('Number of Clusters')
pb.xlim([0,np.max(numspeciesVSnumclust)+2])
pb.ylim([0,np.max(numspeciesVSnumclust)+2])
pb.show()

	





