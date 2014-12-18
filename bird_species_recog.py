import cPickle, gzip
import PIL
import numpy as np
import scipy as sp
import pylab as pb
from scipy import signal
from scikits.audiolab import wavread
import time
import bob
from matplotlib import gridspec
from sklearn import metrics
from sklearn.metrics import pairwise_distances,accuracy_score
from sklearn.cross_validation import KFold,train_test_split


#MFCC parameters
win_length_ms = 25 # The window length of the cepstral analysis in milliseconds
win_shift_ms = 10 # The window shift of the cepstral analysis in milliseconds
n_filters = 41 # The number of filter bands
n_ceps = 19 # The number of cepstral coefficients
f_min = 200. # The minimal frequency of the filter bank
f_max = 8000. # The maximal frequency of the filter bank
delta_win = 2 # The integer delta value used for computing the first and second order derivatives
pre_emphasis_coef = 0.97 # The coefficient used for the pre-emphasis
dct_norm = True # A factor by which the cepstral coefficients are multiplied
mel_scale = True # Tell whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale



#segment single species into training and test
single_ids=np.array([1,6,9,10,18])#species with more that 7 single species recordings
num_files=np.zeros(len(single_ids))
file1=open('single_species.txt',"r")
train_id=[]
test_id=[]
while file1:
	line1=file1.readline()
	s1=line1.split()
	if len(line1)==0:
		break
	if np.sum(single_ids==int(s1[3])):
		#print s1
		num_files[single_ids==int(s1[3])]+=1
	if num_files[single_ids==int(s1[3])]<=5:
		#print 'Train',s1[0],s1[1],s1[2],s1[3]
		train_id.append(int(s1[0]))
	elif num_files[single_ids==int(s1[3])]>5:
		#print 'Test',s1[0],s1[1],s1[2],s1[3]
		test_id.append(int(s1[0]))

file1.close()

#obain segmentation 

file1=open('mlsp_contest_dataset/supplemental_data/segment_rectangles.txt',"r")
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



#get data for UBM that doesn't include single species data
wav_dir='mlsp_contest_dataset/essential_data/src_wavs/'
file1=open('mlsp_contest_dataset/essential_data/rec_id2filename.txt')
line1=file1.readline()#header
idx=0

w_old=32.0#32 ms window used to create spectrograms
alpha1=0.75
alpha=1.0-float(win_shift_ms)/win_length_ms

Mfcc_bird=np.array([])#bird call segments

print 'Obtaining Data for UBM training'
t1=time.clock()
while file1:
	line1=file1.readline()
	s1=line1.split(',')
	if len(line1)==0:
		break
	if (np.sum(np.array(test_id)==int(s1[0]))==0)&(np.sum(np.array(train_id)==int(s1[0]))==0):
		
		x=wavread(wav_dir+s1[1].split('\n')[0]+'.wav')
		c = bob.ap.Ceps(x[1], win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
		mfcc = c(x[0]*2**15)#normalize to integer
		for i in range(num_segs):
			if segs[i,0]==int(s1[0]):
				#print segs[i,:]
				n1=int(((1-alpha1)*segs[i,2]*w_old+(w_old/2)-(float(win_length_ms)/2))/((1-alpha)*float(win_length_ms)))
				n2=int(((1-alpha1)*segs[i,3]*w_old+(w_old/2)-(float(win_length_ms)/2))/((1-alpha)*float(win_length_ms)))
				#n1,n2=0,mfcc.shape[0]
				Mfcc_bird= np.vstack([Mfcc_bird, mfcc[n1:n2,:]]) if Mfcc_bird.size else mfcc[n1:n2,:]
		idx+=1

print('UBM data collected in %.2f minutes'%((time.clock()-t1)/60))

file1.close()


#get single species data for MAP training and testing
file1=open('single_species.txt',"r")
train_mfcc=np.array([])#bird call segments
test_mfcc=np.array([])#bird call segments
train_labels=[]
test_labels=[]
print 'Obtaining single species Training and Test Data'
t1=time.clock()
while file1:
	line1=file1.readline()
	s1=line1.split()
	if len(line1)==0:
		break
	if np.sum(np.array(train_id)==int(s1[0])):
		x=wavread(wav_dir+s1[2]+'.wav')
		c = bob.ap.Ceps(x[1], win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
		mfcc = c(x[0]*2**15)#normalize to integer
		for i in range(num_segs):
			if segs[i,0]==int(s1[0]):
				
				n1=int(((1-alpha1)*segs[i,2]*w_old+(w_old/2)-(float(win_length_ms)/2))/((1-alpha)*float(win_length_ms)))
				n2=int(((1-alpha1)*segs[i,3]*w_old+(w_old/2)-(float(win_length_ms)/2))/((1-alpha)*float(win_length_ms)))
				
				train_mfcc= np.vstack([train_mfcc, mfcc[n1:n2,:]]) if train_mfcc.size else mfcc[n1:n2,:]
				train_labels=np.concatenate((train_labels,np.ones(mfcc[n1:n2,:].shape[0])*int(s1[3])))
	elif np.sum(np.array(test_id)==int(s1[0])):
		x=wavread(wav_dir+s1[2]+'.wav')
		c = bob.ap.Ceps(x[1], win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
		mfcc = c(x[0]*2**15)#normalize to integer
		for i in range(num_segs):
			if segs[i,0]==int(s1[0]):
				
				n1=int(((1-alpha1)*segs[i,2]*w_old+(w_old/2)-(float(win_length_ms)/2))/((1-alpha)*float(win_length_ms)))
				n2=int(((1-alpha1)*segs[i,3]*w_old+(w_old/2)-(float(win_length_ms)/2))/((1-alpha)*float(win_length_ms)))
				test_mfcc= np.vstack([test_mfcc, mfcc[n1:n2,:]]) if test_mfcc.size else mfcc[n1:n2,:]
				test_labels=np.concatenate((test_labels,np.ones(mfcc[n1:n2,:].shape[0])*int(s1[3])))

file1.close()
print('single species data collected in %.2f minutes'%((time.clock()-t1)/60))

#train the gmm ubm with kmeans initialization
print 'Training and testing models...'
t1=time.clock() 
num_clust=32
max_iterations=20
kmeans = bob.machine.KMeansMachine(num_clust, len(Mfcc_bird[0]))
ubm = bob.machine.GMMMachine(num_clust, len(Mfcc_bird[0]))
kmeans_trainer = bob.trainer.KMeansTrainer()
kmeans_trainer.convergence_threshold = 0.0005
kmeans_trainer.max_iterations = max_iterations
kmeans_trainer.check_no_duplicate = True

# Trains using the KMeansTrainer
kmeans_trainer.train(kmeans, Mfcc_bird)
[variances, weights] = kmeans.get_variances_and_weights_for_each_cluster(Mfcc_bird)
means = kmeans.means

#train the gmm
# Initializes the GMM
ubm.means = means

ubm.variances = variances
ubm.weights = weights
ubm.set_variance_thresholds(0.0005)

trainer = bob.trainer.ML_GMMTrainer(True, True, True)
trainer.convergence_threshold = 0.0005
trainer.max_iterations = 25
trainer.train(ubm, Mfcc_bird)
means = ubm.means
weights = ubm.weights


#MAP
relevance_factor = 4.
trainer = bob.trainer.MAP_GMMTrainer(relevance_factor, True, False, False) # mean adaptation only
trainer.convergence_threshold = 1e-5
trainer.max_iterations = 200
trainer.set_prior_gmm(ubm)
models=[]
for i in range(len(single_ids)):
	gmmAdapted = bob.machine.GMMMachine(num_clust,len(Mfcc_bird[0])) # Create a new machine for the MAP estimate
	trainer.train(gmmAdapted, train_mfcc[train_labels==single_ids[i],:])
	models.append(gmmAdapted)

#evaluate loglikelihoods on training data
log_lik=np.zeros((train_mfcc.shape[0],len(models)))
for i in range(train_mfcc.shape[0]):
	for j in range(len(models)):
		log_lik[i,j]=models[j].log_likelihood(train_mfcc[i])

for i in range(len(models)):
	res=np.zeros(len(models))
	for j in range(len(models)):
		res[j]=np.mean(log_lik[train_labels==single_ids[i],j])

	#print single_ids[i] ,single_ids[np.argmax(res)]



print("Framewise classification on training data:\n%s\n" % (
    metrics.classification_report(
        train_labels,single_ids[np.argmax(log_lik,1)])))

print("Accuracy: %.2f"
		      % accuracy_score(train_labels,single_ids[np.argmax(log_lik,1)]))

#evaluate loglikelihoods on test data
log_lik=np.zeros((test_mfcc.shape[0],len(models)))
for i in range(test_mfcc.shape[0]):
	for j in range(len(models)):
		log_lik[i,j]=models[j].log_likelihood(test_mfcc[i])

for i in range(len(models)):
	res=np.zeros(len(models))
	for j in range(len(models)):
		res[j]=np.mean(log_lik[test_labels==single_ids[i],j])

	#print single_ids[i] ,single_ids[np.argmax(res)]

print("Framewise classification on test data:\n%s\n" % (
    metrics.classification_report(
        test_labels,single_ids[np.argmax(log_lik,1)])))

print("Accuracy: %.2f"
		      % accuracy_score(test_labels,single_ids[np.argmax(log_lik,1)]))

print('Training and testing completed in %.2f minutes'%((time.clock()-t1)/60))
#classify entire files
#get single species data for MAP training and testing
file1=open('single_species.txt',"r")
print 'Classiying entire utterance for training files'
t1=time.clock()
true_label=[]
pred_label=[]

while file1:
	line1=file1.readline()
	s1=line1.split()
	if len(line1)==0:
		break
	if np.sum(np.array(train_id)==int(s1[0])):
		x=wavread(wav_dir+s1[2]+'.wav')
		c = bob.ap.Ceps(x[1], win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
		mfcc = c(x[0]*2**15)#normalize to integer
		file_mfcc=np.array([])#bird call segments for particular file
		for i in range(num_segs):
			if segs[i,0]==int(s1[0]):
				#print segs[i,:]
				n1=int(((1-alpha1)*segs[i,2]*w_old+(w_old/2)-(float(win_length_ms)/2))/((1-alpha)*float(win_length_ms)))
				n2=int(((1-alpha1)*segs[i,3]*w_old+(w_old/2)-(float(win_length_ms)/2))/((1-alpha)*float(win_length_ms)))
				#n1,n2=0,mfcc.shape[0]
				file_mfcc= np.vstack([file_mfcc, mfcc[n1:n2,:]]) if file_mfcc.size else mfcc[n1:n2,:]
				train_labels=np.concatenate((train_labels,np.ones(mfcc[n1:n2,:].shape[0])*int(s1[3])))
		#evaluate loglikelihoods
		log_lik=np.zeros((file_mfcc.shape[0],len(models)))
		for i in range(file_mfcc.shape[0]):
			for j in range(len(models)):
				log_lik[i,j]=models[j].log_likelihood(file_mfcc[i])
		#print s1[3], single_ids[np.argmax(np.mean(log_lik,0))]
		true_label.append(int(s1[3]))
		pred_label.append(single_ids[np.argmax(np.mean(log_lik,0))])

file1.close()

print("File Accuracy for training data: %.2f"
		      % accuracy_score(true_label,pred_label))

print("File classification for training data:\n%s\n" % (
    metrics.classification_report(
        true_label,pred_label)))

file1=open('single_species.txt',"r")
print 'Classiying entire utterance for test files'
t1=time.clock()
true_label=[]
pred_label=[]

while file1:
	line1=file1.readline()
	s1=line1.split()
	if len(line1)==0:
		break
	if np.sum(np.array(test_id)==int(s1[0])):
		x=wavread(wav_dir+s1[2]+'.wav')
		c = bob.ap.Ceps(x[1], win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
		mfcc = c(x[0]*2**15)#normalize to integer
		file_mfcc=np.array([])#bird call segments for particular file
		for i in range(num_segs):
			if segs[i,0]==int(s1[0]):
				#print segs[i,:]
				n1=int(((1-alpha1)*segs[i,2]*w_old+(w_old/2)-(float(win_length_ms)/2))/((1-alpha)*float(win_length_ms)))
				n2=int(((1-alpha1)*segs[i,3]*w_old+(w_old/2)-(float(win_length_ms)/2))/((1-alpha)*float(win_length_ms)))
				#n1,n2=0,mfcc.shape[0]
				file_mfcc= np.vstack([file_mfcc, mfcc[n1:n2,:]]) if file_mfcc.size else mfcc[n1:n2,:]
				train_labels=np.concatenate((train_labels,np.ones(mfcc[n1:n2,:].shape[0])*int(s1[3])))
		#evaluate loglikelihoods
		log_lik=np.zeros((file_mfcc.shape[0],len(models)))
		for i in range(file_mfcc.shape[0]):
			for j in range(len(models)):
				log_lik[i,j]=models[j].log_likelihood(file_mfcc[i])
		#print s1[3], single_ids[np.argmax(np.mean(log_lik,0))]
		true_label.append(int(s1[3]))
		pred_label.append(single_ids[np.argmax(np.mean(log_lik,0))])

file1.close()

print("File Accuracy for test data: %.2f"
		      % accuracy_score(true_label,pred_label))

print("File classification for test data:\n%s\n" % (
    metrics.classification_report(
        true_label,pred_label)))



