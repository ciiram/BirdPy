import numpy as np
import scipy as sp
import pylab as pb
from scipy import signal
import scipy.io.wavfile
import time
import sys
import os
import bob
import spectral_features as sf
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import pairwise_distances,accuracy_score
from sklearn.cross_validation import KFold,train_test_split
from sklearn.metrics import roc_curve,auc

#Random seed
np.random.seed(12)
#Split the data into random training and testing sets
#Split data containing HTuraco vocalisation into training and testing
file1=open('HTuracoData/Files/turaco_annotation.csv',"r")#Filenames of wav files
#read header 

file1.readline()
file1.readline()
file1.readline()
audio_files=[]
labels=[]
while file1:
	
	line1=file1.readline()
	s1=line1.split(',')
	
	if len(line1)==0:
		break

	audio_files.append(s1[0])
	labels.append(s1[1])

file1.close()

indx=np.arange(len(audio_files))

#shuffle indx to obtain training and test data 50-50 split

np.random.shuffle(indx)


train_indx=indx[np.arange(0,len(audio_files),2)]
test_indx=indx[np.arange(1,len(audio_files),2)]

file2=open('HTuracoData/Files/turaco-training.txt','w')
for i in range(len(train_indx)):
	file2.write(audio_files[train_indx[i]])
	file2.write('\t')
	file2.write(labels[train_indx[i]])
	file2.write('\n')
file2.close()
file2=open('HTuracoData/Files/turaco-test.txt','w')
for i in range(len(test_indx)):
	file2.write(audio_files[test_indx[i]])
	file2.write('\t')
	file2.write(labels[test_indx[i]])
	file2.write('\n')
file2.close()


file1=open('HTuracoData/Files/noturaco_annotation.csv',"r")#Filenames of wav files
#read header 

file1.readline()
file1.readline()
file1.readline()
audio_files=[]
labels=[]
while file1:
	
	line1=file1.readline()
	s1=line1.split(',')
	
	if len(line1)==0:
		break

	audio_files.append(s1[0])
	labels.append(s1[1])

file1.close()

indx=np.arange(len(audio_files))

#shuffle indx to obtain training and test data 50-50 split

np.random.shuffle(indx)


train_indx=indx[np.arange(0,len(audio_files),2)]
test_indx=indx[np.arange(1,len(audio_files),2)]

file2=open('HTuracoData/Files/noturaco-training.txt','w')
for i in range(len(train_indx)):
	file2.write(audio_files[train_indx[i]])
	file2.write('\t')
	file2.write(labels[train_indx[i]])
	file2.write('\n')
file2.close()
file2=open('HTuracoData/Files/noturaco-test.txt','w')
for i in range(len(test_indx)):
	file2.write(audio_files[test_indx[i]])
	file2.write('\t')
	file2.write(labels[test_indx[i]])
	file2.write('\n')
file2.close()


#Obtain training data

file1=open('HTuracoData/Files/turaco-training.txt',"r")#Filenames of wav files

#parameters
N=512#window size 32ms at 16kHz
alpha=.5#% overlap
fmin=500 #minimum frequency when computing the centroid and bandwidth
fmax=5000  #maximum frequency when computing the centroid and bandwidth
num_bands=6 # number of logarithmically spaced specral bands
num_files=0
start_time=time.time()
spectral_features=np.array([])
train_labels=[]
print 'Obtaining training data...'
while file1:
	
	line1=file1.readline()
	s1=line1.split()
	
	if len(line1)==0:
		break

	
	x= scipy.io.wavfile.read('HTuracoData/Turaco/'+s1[0])
	fs=float(x[0])
	x2=x[1]/2.**15
	x2=x2/np.max(np.abs(x2)) 
	file_feature=np.array([])

	#compute features
	Sxx=sf.spec(x2,N,alpha)
	centroids,band_width=sf.centroid(Sxx,N,fs,fmin,fmax)
	file_feature= np.vstack([file_feature, centroids[1:]]) if file_feature.size else centroids[1:]
	file_feature= np.vstack([file_feature, band_width[1:]])
	rolloff=sf.spectral_rolloff(Sxx,N,fs,.85)
	file_feature= np.vstack([file_feature, rolloff[1:]])
	BER=sf.band_energy_ratio(Sxx,N,fs,num_bands)
	file_feature= np.vstack([file_feature, BER[:,1:]])
	spec_flux=sf.spectral_flux_bands(Sxx,N,num_bands)
	file_feature= np.vstack([file_feature, spec_flux])

	train_labels=np.concatenate((train_labels,np.ones(spec_flux.shape[1])*int(s1[1])))
	spectral_features= np.vstack([spectral_features, file_feature.T]) if spectral_features.size else file_feature.T

	
	num_files+=1

print 'Turaco Features calculated in ',(time.time()-start_time)/(60.0*num_files),' minutes per file'

file1.close()

file1=open('HTuracoData/Files/noturaco-training.txt',"r")#Filenames of wav files
#read header 


num_files=0
start_time=time.time()
while file1:
	
	line1=file1.readline()
	s1=line1.split()
	
	if len(line1)==0:
		break

	
	x= scipy.io.wavfile.read('HTuracoData/NoTuraco/'+s1[0])
	fs=float(x[0])
	x2=x[1]/2.**15
	x2=x2/np.max(np.abs(x2)) 
	file_feature=np.array([])

	#compute features
	Sxx=sf.spec(x2,N,alpha)
	centroids,band_width=sf.centroid(Sxx,N,fs,fmin,fmax)
	file_feature= np.vstack([file_feature, centroids[1:]]) if file_feature.size else centroids[1:]
	file_feature= np.vstack([file_feature, band_width[1:]])
	rolloff=sf.spectral_rolloff(Sxx,N,fs,.85)
	file_feature= np.vstack([file_feature, rolloff[1:]])
	BER=sf.band_energy_ratio(Sxx,N,fs,num_bands)
	file_feature= np.vstack([file_feature, BER[:,1:]])
	spec_flux=sf.spectral_flux_bands(Sxx,N,num_bands)
	file_feature= np.vstack([file_feature, spec_flux])

	train_labels=np.concatenate((train_labels,np.ones(spec_flux.shape[1])*int(s1[1])))
	spectral_features= np.vstack([spectral_features, file_feature.T]) if spectral_features.size else file_feature.T

	
	num_files+=1

print 'No Turaco Features calculated in ',(time.time()-start_time)/(60.0*num_files),' minutes per file'

file1.close()


#Train models

num_clust=32
max_iterations=20
kmeans_turaco = bob.machine.KMeansMachine(num_clust, spectral_features.shape[1])
kmeans_noturaco = bob.machine.KMeansMachine(num_clust, spectral_features.shape[1])
gmm_turaco = bob.machine.GMMMachine(num_clust, spectral_features.shape[1])
gmm_noturaco = bob.machine.GMMMachine(num_clust, spectral_features.shape[1])
kmeans_trainer = bob.trainer.KMeansTrainer()
kmeans_trainer.convergence_threshold = 0.0005
kmeans_trainer.max_iterations = max_iterations
kmeans_trainer.check_no_duplicate = True

# Trains using the KMeansTrainer
print 'Running Kmeans...'
start_time=time.time()
kmeans_trainer.train(kmeans_turaco, spectral_features[train_labels==1,:])
[variances_turaco, weights_turaco] = kmeans_turaco.get_variances_and_weights_for_each_cluster(spectral_features[train_labels==1,:])
means_turaco = kmeans_turaco.means

kmeans_trainer.train(kmeans_noturaco, spectral_features[train_labels==0,:])
[variances_noturaco, weights_noturaco] = kmeans_noturaco.get_variances_and_weights_for_each_cluster(spectral_features[train_labels==0,:])
means_noturaco = kmeans_noturaco.means

print 'Run Kmeans in ',(time.time()-start_time)/(60.0),' minutes'

#train the gmm
# Initializes the GMM
gmm_turaco.means = means_turaco
gmm_turaco.variances = variances_turaco
gmm_turaco.weights = weights_turaco
gmm_turaco.set_variance_thresholds(0.0005)

gmm_noturaco.means = means_noturaco
gmm_noturaco.variances = variances_noturaco
gmm_noturaco.weights = weights_noturaco
gmm_noturaco.set_variance_thresholds(0.0005)

trainer = bob.trainer.ML_GMMTrainer(True, True, True)
trainer.convergence_threshold = 0.0005
trainer.max_iterations = 25

print 'Training GMMs...'
start_time=time.time()
trainer.train(gmm_turaco, spectral_features[train_labels==1,:])
trainer.train(gmm_noturaco, spectral_features[train_labels==0,:])
print 'Trained GMMs in ',(time.time()-start_time)/(60.0),' minutes'


#evaluate performance on test data


file1=open('HTuracoData/Files/turaco-test.txt',"r")#Filenames of wav files
llr=np.array([])
test_labels=[]
while file1:
	
	line1=file1.readline()
	s1=line1.split()
	
	if len(line1)==0:
		break

	
	x= scipy.io.wavfile.read('HTuracoData/Turaco/'+s1[0])
	fs=float(x[0])
	x2=x[1]/2.**15
	x2=x2/np.max(np.abs(x2)) 
	file_feature=np.array([])

	#compute features
	Sxx=sf.spec(x2,N,alpha)
	centroids,band_width=sf.centroid(Sxx,N,fs,fmin,fmax)
	file_feature= np.vstack([file_feature, centroids[1:]]) if file_feature.size else centroids[1:]
	file_feature= np.vstack([file_feature, band_width[1:]])
	rolloff=sf.spectral_rolloff(Sxx,N,fs,.85)
	file_feature= np.vstack([file_feature, rolloff[1:]])
	BER=sf.band_energy_ratio(Sxx,N,fs,num_bands)
	file_feature= np.vstack([file_feature, BER[:,1:]])
	spec_flux=sf.spectral_flux_bands(Sxx,N,num_bands)
	file_feature= np.vstack([file_feature, spec_flux])

	ll_turaco=0
	ll_noturaco=0
	for i in range(file_feature.shape[1]):
		ll_turaco+=gmm_turaco.log_likelihood(file_feature[:,i])
		ll_noturaco+=gmm_noturaco.log_likelihood(file_feature[:,i])

	
	llr=np.concatenate((llr,np.array([(ll_turaco-ll_noturaco)/file_feature.shape[1]])))
	test_labels.append(int(s1[1]))
	num_files+=1

file1.close()


file1=open('HTuracoData/Files/noturaco-test.txt',"r")#Filenames of wav files
while file1:
	
	line1=file1.readline()
	s1=line1.split()
	
	if len(line1)==0:
		break

	
	x= scipy.io.wavfile.read('HTuracoData/NoTuraco/'+s1[0])
	fs=float(x[0])
	x2=x[1]/2.**15
	x2=x2/np.max(np.abs(x2)) 
	file_feature=np.array([])

	#compute features
	Sxx=sf.spec(x2,N,alpha)
	centroids,band_width=sf.centroid(Sxx,N,fs,fmin,fmax)
	file_feature= np.vstack([file_feature, centroids[1:]]) if file_feature.size else centroids[1:]
	file_feature= np.vstack([file_feature, band_width[1:]])
	rolloff=sf.spectral_rolloff(Sxx,N,fs,.85)
	file_feature= np.vstack([file_feature, rolloff[1:]])
	BER=sf.band_energy_ratio(Sxx,N,fs,num_bands)
	file_feature= np.vstack([file_feature, BER[:,1:]])
	spec_flux=sf.spectral_flux_bands(Sxx,N,num_bands)
	file_feature= np.vstack([file_feature, spec_flux])

	ll_turaco=0
	ll_noturaco=0
	for i in range(file_feature.shape[1]):
		ll_turaco+=gmm_turaco.log_likelihood(file_feature[:,i])
		ll_noturaco+=gmm_noturaco.log_likelihood(file_feature[:,i])

	
	llr=np.concatenate((llr,np.array([(ll_turaco-ll_noturaco)/file_feature.shape[1]])))
	test_labels.append(int(s1[1]))
	num_files+=1

file1.close()


#save models
myh5_file = bob.io.HDF5File('HTuracoData/Models/TuracoModel.hdf5', 'w')
gmm_turaco.save(myh5_file)
del myh5_file #close
myh5_file = bob.io.HDF5File('HTuracoData/Models/NoTuracoModel.hdf5', 'w')
gmm_noturaco.save(myh5_file)
del myh5_file #close

fpr,tpr,thresh=roc_curve(test_labels,llr)
AUC=auc(fpr,tpr)
xx=tpr+fpr
print "True Positive Rate",tpr[np.argmin((np.abs(1-xx)))]
print "False Positive Rate",fpr[np.argmin((np.abs(1-xx)))]
print "Threshold",thresh[np.argmin((np.abs(1-xx)))]

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % AUC,linewidth=2)
plt.plot([0, 1], [0, 1], 'k--',linewidth=2)
plt.plot(fpr[np.argmin((np.abs(1-xx)))],tpr[np.argmin((np.abs(1-xx)))],'bo',markersize=12)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic', fontsize=14)
plt.legend(loc="lower right")
pb.savefig('HTuracoData/Files/hturaco_roc.png')
plt.show()




