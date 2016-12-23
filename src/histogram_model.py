import model
import parse
import preprocess
import nibabel as nib
import numpy as np
from sklearn.model_selection import KFold
from scipy.ndimage.filters import gaussian_filter
from sklearn.svm import SVC

import preprocess

#this model uses the voxels as input
class histogram_model(model.model):

	ncubes=-1
	smoothening=-1

	def __init__(self,ntrain,ntest,seg,gamma,slack,ncubes,smoothening,cname,nclasses):
		model.model.__init__(self,ntrain,ntest,seg,gamma,slack,cname,nclasses)
		self.custom_svm=True
		self.ncubes=ncubes
		self.smoothening=smoothening
		if(ncubes>1):
			for i in range(0,nclasses):
				self.predictor[i]=cube_predictor.cube_predictor(ncubes,gamma[i],slack[i],cv_opt=True)
				
	def read_features(self,path):
		tmp=[]
		train_path=path+"set_train/c"+str(self.seg)+"train_" #grey matter mask
		ref_path=path+"set_train/"+"train_"  #original image
		print "reading train images"
		for i in range(0,self.ntrain):
			filename=train_path+str(i+1)+".nii"
			ref_name=ref_path+str(i+1)+".nii"
			hist=get_histogram(filename,ref_name,self.ncubes,self.smoothening)
			tmp.append(hist)
		self.train_features=np.zeros((self.ntrain,tmp[0].size))
		for i in range (0,self.ntrain):
			self.train_features[i,:]=tmp[i]
		tmp=[]
		test_path=path+"set_test/c"+str(self.seg)+"test_"
		ref_path=path+"set_test/"+"test_"
		print "reading train images"
		for i in range(0,self.ntest):
			filename=test_path+str(i+1)+".nii"
			hist=get_histogram(filename,ref_name,self.ncubes,self.smoothening)
			tmp.append(hist)
		self.test_features=np.zeros((self.ntest,tmp[0].size))
		for i in range (0,self.ntest):
			self.test_features[i,:]=tmp[i]
			
def get_histogram(filename,ref_name,ncubes,smoothening):
	array=parse.voxels_from_image(filename,smoothening_width=smoothening)
	array=preprocess.features1D(array,ncubes)
	ref_array=parse.voxels_from_image(filename,smoothening_width=smoothening)
	ref_array=preprocess.features1D(ref_array,ncubes)
	true_array=np.multiply(array,ref_array)
	hist=np.histogram(true_array,bins=100,range=(0,1000))
	return hist[0]
