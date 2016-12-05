import model
import parse
import preprocess
import nibabel as nib
import numpy as np
from sklearn.model_selection import KFold

import preprocess

#this model uses the voxels as input
class voxel_model(model.model):

	ncubes=1

	def __init__(self,ntrain,ntest,seg,gamma_scale,ncubes,cname):
		model.model.__init__(self,ntrain,ntest,seg,gamma_scale,cname)
		self.ncubes=ncubes
		
	def read_features(self,path):
		tmp=[]
		train_path=path+"set_train/c"+str(self.seg)+"train_"
		for i in range(0,self.ntrain):
			print "reading train image "+str(i)
			filename=train_path+str(i+1)+".nii"
			array=parse.voxels_from_image(filename,self.ncubes)
			tmp.append(np.reshape(array,array.size))
		self.train_features=np.zeros((self.ntrain,tmp[0].size))
		for i in range (0,self.ntrain):
			self.train_features[i,:]=tmp[i]
		tmp=[]
		test_path=path+"set_test/c"+str(self.seg)+"test_"
		for i in range(0,self.ntest):
			print "reading test image "+str(i)
			filename=test_path+str(i+1)+".nii"
			array=parse.voxels_from_image(filename,self.ncubes)
			tmp.append(np.reshape(array,array.size))
		self.test_features=np.zeros((self.ntest,tmp[0].size))
		for i in range (0,self.ntest):
			self.test_features[i,:]=tmp[i]
