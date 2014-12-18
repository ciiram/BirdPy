import PIL
import numpy as np
import scipy as sp
import pylab as pb
from scipy import signal
from scikits.audiolab import wavread
from scipy.stats.stats import pearsonr
import time

'''
This code plots the acoustic biodiversity index versus the number of species
'''

A=np.genfromtxt('res.txt')


num_species=6

y1=np.zeros(num_species)
y2=np.zeros(num_species)
x=np.arange(1,7)

for i in range(1,7):
	y1[i-1]=np.mean(A[A[:,0]==i,1])
	y2[i-1]=np.mean(A[A[:,0]==i,2])

print 'Correlation without segmentation',pearsonr(x,y1)[0]
print 'Correlation with segmentation',pearsonr(x,y2)[0]
pb.figure()
pb.plot(A[:,0],A[:,2],'bx')
pb.plot(x,y2,'bs',markersize=10)
pb.plot(A[:,0],A[:,1],'ro')
pb.plot(x,y1,'rd',markersize=10)
pb.xlim([0,7])
pb.ylim([0,1])
pb.xlabel('Number of Species')
pb.ylabel('Acoustic Entropy')
pb.legend(['With Segmentation','Mean With Segmentation','Without Segmentation','Mean Without Segmentation'],loc='lower right')
pb.savefig('ent.png',dpi=300)

	
