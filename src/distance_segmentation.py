import model
import parse
import preprocess
import nibabel as nib
import numpy as np
from sklearn.model_selection import KFold

import preprocess

#this model calculates the closest distance of each voxel to a voxel that is not part of the segment
class distance_segmentation(model.model):

	invert = False
	ncubes=1

	
	def __init__(self,ntrain,ntest,seg,gamma_scale,slack,ncubes,cname,nclasses,invert=False):
		model.model.__init__(self,ntrain,ntest,seg,gamma_scale,slack,cname,nclasses)
		self.invert=invert
		self.ncubes=ncubes
		
	def read_features(self,path):
		tmp=[]
		train_path=path+"set_train/c"+str(self.seg)+"train_"
		for i in range(0,self.ntrain):
			print "reading train image "+str(i)
			filename=train_path+str(i+1)+".nii"
			array=parse.voxels_from_image(filename,self.ncubes,smoothening=False)
			if(self.invert):
				array[array==0]=-1
				array[array>0]=0
				array[array==-1]=10000
			else:
				array[array>0]=10000
			array=closest_distance(array)
			tmp.append(np.reshape(array,array.size))
		self.train_features=np.zeros((self.ntrain,tmp[0].size))
		for i in range (0,self.ntrain):
			self.train_features[i,:]=tmp[i]
		tmp=[]
		test_path=path+"set_test/c"+str(self.seg)+"test_"
		for i in range(0,self.ntest):
			print "reading test image "+str(i)
			filename=test_path+str(i+1)+".nii"
			array=parse.voxels_from_image(filename,self.ncubes,smoothening=False)
			if(self.invert):
				array[array==0]=-1
				array[array>0]=0
				array[array==-1]=10000
			else:
				array[array>0]=10000
			array=closest_distance(array)
			tmp.append(np.reshape(array,array.size))
		self.test_features=np.zeros((self.ntest,tmp[0].size))
		for i in range (0,self.ntest):
			self.test_features[i,:]=tmp[i]
			
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
		
	
		
	
	
		
