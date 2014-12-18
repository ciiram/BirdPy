OVERVIEW
========

This folder contains Python code used to study acoustic indices of biodiversity monitoring and bird species recognition using bird calls and songs. The associated manuscript “Bioacoustic Approaches to Biodiversity Monitoring and Conservation in  Kenya,” 
by Ciira wa Maina has been submitted to [IST Africa 2015](http://www.ist-africa.org/home/)


REQUIREMENTS
============
The programs require Python 2.7 or later and the following python libraries

1. numpy >= 1.6.1
2. scipy >= 0.9.0
3. pylab
4. scikits.audiolab >= 0.11.0
5. [bob >= 1.2.2](https://github.com/idiap/bob) bob is a machine learning and signal processing toolkit developed at IDIAP.
6. PIL
7. Matplotlib


INSTALLATION
============

Download the repository and the data for the MLSP 2013 Bird Classification Challenge and place the folder mlsp_contest_dataset in the folder containing the code. The data are available [here](https://www.kaggle.com/c/mlsp-2013-birds/data)

The file *num_species_ids.txt* contains the file IDs and identifiers for the 179 file names in the dataset for which the number of species present in the data set is available. It is derived from the files in the *essential_data* folder.


USE
=======

Biodiversity Monitoring
-----------------------

To compute the acoustic indices for these files open an Ipython shell and type
	
	run acoustic_biodiversity.py

or type

	python acoustic_biodiversity.py	

in the commandline.

To plot the results type 

	run plot_acoustic_index.py

A file *ent.png* is saved which plots the acoustic entropy index with and without audio segmentation. This is Figure 3 in the manuscript. The correlation between the acoustic index and the number of species in the recording is 0.89 with segmentation and 0.17 without it.


Bird species recognition
------------------------

The file *bird_species_recog.py* performs all the operations of bird species recognition. Here we focus on species for which more than 7 single species recordings are available. The corresponding species IDs are 1,6,9,12 and 18. All single species files as well as the species ID are contained in the file *single_species.txt*. We use 5 files for training the species specific models and the rest for testing. In total we have 25 training files and 24 testing files.
The step implemented are:

1. Train a UBM using all the data that contains more than one species per recording. 19 dimensional MFCCs are used and are computed using [bob](https://github.com/idiap/bob). The model is trained using only segments labelled as bird sound.
2. Adapt the speaker specific models using bob using MAP
3. Obtain accuracy measures on both training and test data. 


To run the code open an Ipython shell and type
	
	run bird_species_recog.py

The output is 

	Obtaining Data for UBM training
	UBM data collected in 0.71 minutes
	Obtaining single species Training and Test Data
	single species data collected in 0.06 minutes
	Training and testing models...
	Framewise classification on training data:
		     precision    recall  f1-score   support

		1.0       0.98      0.88      0.93      1100
		6.0       0.84      0.88      0.86       469
		9.0       0.94      0.96      0.95       365
	       10.0       0.31      0.97      0.47        38
	       18.0       0.93      0.92      0.93       535

	avg / total       0.93      0.90      0.91      2507


	Accuracy: 0.90
	Framewise classification on test data:
		     precision    recall  f1-score   support

		1.0       0.65      0.41      0.50       928
		6.0       0.30      0.34      0.32       263
		9.0       0.11      0.13      0.12       324
	       10.0       0.43      0.41      0.42       599
	       18.0       0.16      0.84      0.27        68

	avg / total       0.45      0.37      0.39      2182


	Accuracy: 0.37
	Training and testing completed in 0.40 minutes
	Classiying entire utterance for training files
	File Accuracy for training data: 0.68
	File classification for training data:
		     precision    recall  f1-score   support

		  1       0.38      1.00      0.56         5
		  6       1.00      0.60      0.75         5
		  9       1.00      0.60      0.75         5
		 10       1.00      0.20      0.33         5
		 18       1.00      1.00      1.00         5

	avg / total       0.88      0.68      0.68        25


	Classiying entire utterance for test files
	File Accuracy for test data: 0.50
	File classification for test data:
		     precision    recall  f1-score   support

		  1       0.27      0.60      0.37         5
		  6       1.00      0.25      0.40         4
		  9       0.33      0.20      0.25         5
		 10       0.83      0.71      0.77         7
		 18       0.67      0.67      0.67         3

	avg / total       0.62      0.50      0.50        24

	

