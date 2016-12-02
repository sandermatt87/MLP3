import numpy as np
import nibabel as nib
import csv
import os
import preprocess

#functions related to readng from files
def voxels_from_image(filename,ncubes,smoothening=True,stride=[4,4,4],smoothening_width=2.2):
	img=nib.load(filename)
	array=preprocess.crop(img)
	if(smoothening):
		array=preprocess.smoothe(array,smoothening_width)
	array=preprocess.reduce_resolution(array,stride)
	return array

def read_targets(ntargets,nclasses):
	targets=np.zeros((ntargets,nclasses))
	csv_str='../targets.csv'
		
	with open(csv_str, 'rb') as csvfile:
		infile = csv.reader(csvfile)
		i=0
		for row in infile:
			for j in range(0,len(row)):
				targets[i,j]=int(row[j])
			i+=1
			if(i==ntargets):
				break
	return targets


