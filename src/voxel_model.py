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
	smootening=-1

	def __init__(self,ntrain,ntest,seg,gamma_scale,slack,ncubes,smoothening,cname):
		model.model.__init__(self,ntrain,ntest,seg,gamma_scale,slack,cname)
		self.ncubes=ncubes
		self.smoothening=smoothening
		
	def read_features(self,path):
		tmp=[]
		train_path=path+"set_train/c"+str(self.seg)+"train_"
		print "reading train images"
		for i in range(0,self.ntrain):
			filename=train_path+str(i+1)+".nii"
			array=parse.voxels_from_image(filename,self.ncubes,smoothening_width=self.smootening)
			tmp.append(np.reshape(array,array.size))
		self.train_features=np.zeros((self.ntrain,tmp[0].size))
		for i in range (0,self.ntrain):
			self.train_features[i,:]=tmp[i]
		tmp=[]
		test_path=path+"set_test/c"+str(self.seg)+"test_"
		print "reading test images"
		for i in range(0,self.ntest):
			filename=test_path+str(i+1)+".nii"
			array=parse.voxels_from_image(filename,self.ncubes,smoothening_width=self.smootening)
			tmp.append(np.reshape(array,array.size))
		self.test_features=np.zeros((self.ntest,tmp[0].size))
		for i in range (0,self.ntest):
			self.test_features[i,:]=tmp[i]
