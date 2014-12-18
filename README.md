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

The file num_species_ids.txt contains the file IDs and identifiers for the 179 file names in the dataset for which the number of species present in the data set is available. It is derived from the files in the essential_data folder.

To compute the acoustic indices for these files open an Ipython shell and type
	
	run acoustic_biodiversity.py

or type

	python acoustic_biodiversity.py	

in the commandline.

To plot the results type 

	run plot_acoustic_index.py

A file ent.png is saved which plots the acoustic entropy index with and without audio segmentation. We see that the correlation between the acoustic index and the number of species in the recording is 0.89 with segmentation and 0.17 without it.
	

