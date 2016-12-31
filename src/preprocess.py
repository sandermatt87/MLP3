import numpy as np
import nibabel as nib
import math
from sklearn.decomposition import PCA
from scipy.ndimage.filters import gaussian_filter

#functions regarding the preprocessing of the features
def reduce_resolution(array,factor):
	new_array=array[0::factor[0],0::factor[1],0::factor[2]]
	return new_array
	
def smoothe(array,smoothening_width):
	new_array=gaussian_filter(array,smoothening_width)
	return new_array
	
def reshape(img):
	array=img.get_data()
	new_array=np.reshape(array,array.size)
	return new_array
	
def crop(img,box_boundary=[20,153,20,187,14,149]):
	array=img.get_data()
	new_array=array[box_boundary[0]:box_boundary[1]+1,box_boundary[2]:box_boundary[3]+1,box_boundary[4]:box_boundary[5]+1]
	return new_array
	
def get_nonzero_variance(array):
	#returns all zero variance elements
	diff=np.zeros(array.shape)
	for i in range(0,array.shape[0]):
		diff[i,:]=array[i,:]-array[0,:]
	absmax=np.amax(diff,axis=0)
	nonzeros=np.nonzero(absmax)
	return nonzeros[0]

def remove_zero_variance(array,nonzeros):
	print "removing zero variance elements"
	new_array=np.zeros((array.shape[0],nonzeros.shape[0]))
	for i in range(0,nonzeros.shape[0]):
		new_array[:,i]=array[:,nonzeros[i]]
	print "removed "+str(array.shape[1]-nonzeros.shape[0]) + " of " + str(array.shape[1]) + " voxels"
	return new_array
	
def remove_zero_variance_vector(vector,nonzeros):
	new_vector=np.zeros(nonzeros.shape[0])
	for i in range(0,nonzeros.shape[0]):
		new_vector[i]=vector[nonzeros[i]]
	return new_vector
	
def get_cube(array,ncubes,pos):
	low_bound=[pos[i]*(array.shape[i]/ncubes) for i in range(0,3)]
	high_bound=[(pos[i]+1)*(array.shape[i]/ncubes) for i in range(0,3)]
	new_array=array[low_bound[0]:high_bound[0],low_bound[1]:high_bound[1],low_bound[2]:high_bound[2]]
	return new_array
	
def features1D(array,ncubes):
	#creates a 1d array of the features, with the cubes listed consecutively on the array
	cubes=[]
	for x in range(0,ncubes):
		for y in range(0,ncubes):
			for z in range(0,ncubes):
				cubes.append(get_cube(array,ncubes,[x,y,z]))
	result=np.zeros(cubes[0].size*ncubes**3)
	for i in range(0,len(cubes)):
		result[i*cubes[0].size:(i+1)*cubes[0].size]=np.reshape(cubes[i],cubes[i].size)
	return result
	
def fft_features1D(array,ncubes):
	#creates a 1d array of the features, with the cubes listed consecutively on the array
	cubes=[]
	for x in range(0,ncubes):
		for y in range(0,ncubes):
			for z in range(0,ncubes):
				cubes.append(get_cube(array,ncubes,[x,y,z]))
	result=np.zeros(2*cubes[0].size*ncubes**3)
	for i in range(0,len(cubes)):
		result[2*i*cubes[0].size:(2*i+1)*cubes[0].size]=np.real(np.fft.fft(np.reshape(cubes[i],cubes[i].size)))
		result[(2*i+1)*cubes[0].size:(2*i+2)*cubes[0].size]=np.imag(np.fft.fft(np.reshape(cubes[i],cubes[i].size)))
	return result
	
def segwise_pca(train_features,test_features,ncubes,ntrain,ntest):
	#pca transforms the features cube by cube
	nfeatures=train_features.shape[1]
	features_per_cube=nfeatures/(ncubes**3)
	out_train=np.zeros((ntrain,ntrain*(ncubes**3)))
	out_test=np.zeros((ntest,ntrain*(ncubes**3)))
	for i in range(0,ncubes**3):
		pca = PCA(n_components=ntrain, svd_solver='full')
		out_train[:,i*ntrain:(i+1)*ntrain]=pca.fit_transform(train_features[:,i*features_per_cube:(i+1)*features_per_cube])
		out_test[:,i*ntrain:(i+1)*ntrain]=pca.transform(test_features[:,i*features_per_cube:(i+1)*features_per_cube])
	return out_train,out_test

		
				
