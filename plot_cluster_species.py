import numpy as np
import scipy as sp
import pylab as pb
from scipy import signal
from scikits.audiolab import wavread
from scipy.stats.stats import pearsonr
import time
import sys

'''
This code plots the the number of species versus number of clusters determined 
'''

if len(sys.argv) != 2:  
	sys.exit("Usage: run plot_cluster_species.py [filename]")
numspeciesVSnumclust=np.genfromtxt(sys.argv[1])



print "Number of correct estimates", sum(numspeciesVSnumclust[:,0]==numspeciesVSnumclust[:,1]-1), "Percentage correct:",float(sum(numspeciesVSnumclust[:,0]==numspeciesVSnumclust[:,1]-1))/numspeciesVSnumclust.shape[0]

pb.figure()
pb.plot(numspeciesVSnumclust[:,0],numspeciesVSnumclust[:,1]-1,'bo',markersize=10)
pb.xlabel('Number of Species')
pb.ylabel('Number of Clusters')
pb.xlim([0,np.max(numspeciesVSnumclust)+2])
pb.ylim([0,np.max(numspeciesVSnumclust)+2])
pb.savefig('numspeciesVSnumclust.png',dpi=300)
pb.show()
