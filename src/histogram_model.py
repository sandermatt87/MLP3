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
	def __init__(self,ntrain,ntest,seg,ncubes,pos,cname,weight):
		model.model.__init__(self,ntrain,ntest,seg,ncubes,pos,cname,weight)
		self.predictor = SVC(probability=True,C=2,gamma=0.000000005)
		self.custom_svm=True
		
	def read_features(self,path):
		print "parsing features at ",str(self.seg),self.ncubes,self.pos
		tmp=[]
		train_path=path+"set_train/c"+str(self.seg)+"train_"
		ref_path=path+"set_train/"+"train_"
		for i in range(0,self.ntrain):
			print "reading train image "+str(i)
			filename=train_path+str(i+1)+".nii"
			ref_name=ref_path+str(i+1)+".nii"
			hist=get_unmasked_values(filename,ref_name,self.ncubes,self.pos)
			tmp.append(hist)
		self.train_features=np.zeros((self.ntrain,tmp[0].size))
		for i in range (0,self.ntrain):
			self.train_features[i,:]=tmp[i]
		tmp=[]
		test_path=path+"set_test/c"+str(self.seg)+"test_"
		ref_path=path+"set_test/"+"test_"
		for i in range(0,self.ntest):
			print "reading test image "+str(i)
			filename=test_path+str(i+1)+".nii"
			ref_name=ref_path+str(i+1)+".nii"
			hist=get_unmasked_values(filename,ref_name,self.ncubes,self.pos)
			tmp.append(hist)
		self.test_features=np.zeros((self.ntest,tmp[0].size))
		for i in range (0,self.ntest):
			self.test_features[i,:]=tmp[i]
			
def get_unmasked_values(filename,ref_name,ncubes,pos):
	array=parse.voxels_from_image(filename,ncubes,pos,smoothening=False,stride=[1,1,1])
	ref_array=parse.voxels_from_image(ref_name,ncubes,pos,smoothening=False,stride=[1,1,1])
	array=np.round(array)
	true_array=np.multiply(array,ref_array[:,:,:,0])
	#true_array=gaussian_filter(true_array,2.2)
	hist=np.histogram(true_array,bins=100,range=(0,1000))
	return hist[0]
