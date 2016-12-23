import nibabel as nib
import numpy as np
from sklearn.model_selection import KFold

import preprocess
import cube_predictor
import model
import parse

#this model calculates the closest distance of each voxel to a voxel that is not part of the segment
class distance_segmentation(model.model):

	invert = False
	ncubes=1

	
	def __init__(self,ntrain,ntest,seg,gamma,slack,ncubes,cname,nclasses,invert=False):
		model.model.__init__(self,ntrain,ntest,seg,gamma,slack,cname,nclasses)
		self.ncubes=ncubes
		if(ncubes>1):
			for i in range(0,nclasses):
				self.predictor[i]=cube_predictor.cube_predictor(ncubes,gamma[i],slack[i],cv_opt=True)
		
	def read_features(self,path):
		tmp=[]
		train_path=path+"set_train/c"+str(self.seg)+"train_"
		print "reading train images"
		for i in range(0,self.ntrain):
			filename=train_path+str(i+1)+".nii"
			array=parse.voxels_from_image(filename,smoothening=False)
			if(self.invert):
				array[array==0]=-1
				array[array>0]=0
				array[array==-1]=10000
			else:
				array[array>0]=10000
			array=closest_distance(array)
			tmp.append(preprocess.features1D(array,self.ncubes))
		self.train_features=np.zeros((self.ntrain,tmp[0].size))
		for i in range (0,self.ntrain):
			self.train_features[i,:]=tmp[i]
		tmp=[]
		test_path=path+"set_test/c"+str(self.seg)+"test_"
		print "reading test images"
		for i in range(0,self.ntest):
			filename=test_path+str(i+1)+".nii"
			array=parse.voxels_from_image(filename,smoothening=False)
			if(self.invert):
				array[array==0]=-1
				array[array>0]=0
				array[array==-1]=10000
			else:
				array[array>0]=10000
			array=closest_distance(array)
			tmp.append(preprocess.features1D(array,self.ncubes))
		self.test_features=np.zeros((self.ntest,tmp[0].size))
		if(self.ncubes==1):
			#Remove zero vaiance elements, to make the pca cheaper. This can only be done without cubes, since otherwise the format gets messed up
			nonzeros=preprocess.get_nonzero_variance(self.train_features)
			self.train_features=preprocess.remove_zero_variance(self.train_features,nonzeros)
			self.test_features=preprocess.remove_zero_variance(self.test_features,nonzeros)
		self.train_features,self.test_features=preprocess.segwise_pca(self.train_features,self.test_features,self.ncubes,self.ntrain,self.ntest)
			
def closest_distance(array):
	xdim=array.shape[0]+2
	ydim=array.shape[1]+2
	zdim=array.shape[2]+2
	centered=np.zeros((xdim,ydim,zdim))
	right=np.copy(centered)
	left=np.copy(centered)
	bottom=np.copy(centered)
	top=np.copy(centered)
	front=np.copy(centered)
	back=np.copy(centered)
	centered[1:xdim-1,1:ydim-1,1:zdim-1]=array
	diff=1
	i=0
	while (abs(diff)>0):
		old_centered=np.copy(centered)
		right[2:,1:ydim-1,1:zdim-1]=np.copy(array)+1
		left[:xdim-2,1:ydim-1,1:zdim-1]=np.copy(array)+1
		back[1:xdim-1,:ydim-2,1:zdim-1]=np.copy(array)+1
		front[1:xdim-1,2:ydim,1:zdim-1]=np.copy(array)+1
		bottom[1:xdim-1,1:ydim-1,:zdim-2]=np.copy(array)+1
		top[1:xdim-1,1:ydim-1,2:zdim]=np.copy(array)+1
		centered=np.minimum(centered,left)
		centered=np.minimum(centered,right)
		centered=np.minimum(centered,top)
		centered=np.minimum(centered,bottom)
		centered=np.minimum(centered,front)
		centered=np.minimum(centered,back)
		diff=np.sum(centered-old_centered)
		array=np.copy(centered[1:xdim-1,1:ydim-1,1:zdim-1])
		#print "iteration: ", i
		i+=1
	return array
		
	
		
	
	
		
