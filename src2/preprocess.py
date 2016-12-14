import numpy as np
import nibabel as nib
import math
from scipy.ndimage.filters import gaussian_filter

#functions regarding the preprocessing of the features
def reduce_resolution(array,factor):
	new_array=array[0::factor[0],0::factor[1],0::factor[2]]
	return new_array
	
def smoothe(array,smoothening_width):
	new_array = gaussian_filter(array,smoothening_width)
	return new_array
	
def reshape(img):
	array=img.get_data()
	new_array=np.reshape(array,array.size)
	return new_array
	
def crop(img,box_boundary=[20,153,20,187,14,149]):
	array=img.get_data()
	return array[box_boundary[0]:box_boundary[1]+1,box_boundary[2]:box_boundary[3]+1,box_boundary[4]:box_boundary[5]+1]
	
def get_nonzero_variance(array):
	#removes all zero variance elements
	diff=np.zeros(array.shape)
	for i in range(0,array.shape[0]):
		diff[i,:]=array[i,:]-array[0,:]
	absmax=np.amax(diff,axis=0)
	nonzeros=np.nonzero(absmax)
	return nonzeros[0]

def remove_zero_variance(array,nonzeros):
	print( "removing zero variance elements")
	new_array=np.zeros((array.shape[0],nonzeros.shape[0]))
	for i in range(0,nonzeros.shape[0]):
		new_array[:,i]=array[:,nonzeros[i]]
	print( "removed "+str(array.shape[1]-nonzeros.shape[0]) + " of " + str(array.shape[1]) + " voxels")
	return new_array
	
def remove_zero_variance_vector(vector,nonzeros):
	new_vector=np.zeros(nonzeros.shape[0])
	for i in range(0,nonzeros.shape[0]):
		new_vector[i]=vector[nonzeros[i]]
	return new_vector
