import numpy as np
import nibabel as nib
import csv
import os
import preprocess

import consts

import time

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
	if consts.modeKCT:
		csv_str='C:/phd/MLcourse/MLP3_stuff/targets.csv'
	else:
		csv_str = '../targets.csv'
		
	with open(csv_str, 'r') as csvfile:
		infile = csv.reader(csvfile)
		i=0
		for row in infile:
			for j in range(0,len(row)):
				targets[i,j]=int(row[j])
			i+=1
			if(i==ntargets):
				break
	return targets


