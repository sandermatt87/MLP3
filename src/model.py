import numpy as np
from sklearn.model_selection import KFold
import os.path
import os
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss
from sklearn.multiclass import OneVsRestClassifier

import preprocess
import utils

#the parent class for all models
class model:
	cv_score=0
	ntrain=0
	ntest=0
	train_features=None
	test_features=None
	predictor=[]
	predictions=None
	cv_predictions=None
	fname_cache_train=None
	fname_cache_test=None
	fname_cache_pred=None
	fname_cache_cv_pred=None
	feature_cache_exists=False
	prediction_cache_exists=False
	gamma_scale=-1
	seg=-1
	slack=-1
	
	def __init__(self,ntrain,ntest,seg,gamma_scale,slack,cname):
		self.ntrain=ntrain
		self.ntest=ntest
		self.fname_cache_train="./cache/features/"+cname+"train.npy"
		self.fname_cache_test="./cache/features/"+cname+"test.npy"
		self.fname_cache_pred="./cache/predictions/"+cname+"pred.npy"
		self.fname_cache_cv_pred="./cache/predictions/"+cname+"cv_pred.npy"
		self.seg=seg
		self.gamma_scale=gamma_scale
		self.slack=slack
	
	def train(self,targets,nsplits,nclasses):
		#new_targets=utils.to_single_class(targets)
		
		print "training model with: ",self.fname_cache_train
		if(self.prediction_cache_exists):
			self.read_cache_predictions()
			print targets,self.cv_predictions
			self.cv_score+=hamming_loss(targets,self.cv_predictions)
			print "cv_hamming_loss: "+str(self.cv_score)
		else:
			for i in range(0,nclasses):
				self.predictor.append(SVC(C=self.slack[i],gamma=self.gamma_scale[i]/self.train_features.shape[1]))
			np.random.seed(1231)
			self.cv_predictions=np.copy(targets)*0
			self.cv_score=0
			kf = KFold(n_splits=nsplits)
			for train, test in kf.split(self.train_features):
				for i in range(0,nclasses):
					self.predictor[i].fit(self.train_features[train],targets[train,i])
					self.cv_predictions[test,i]=self.predictor[i].predict(self.train_features[test])
				self.cv_score+=hamming_loss(targets[test],self.cv_predictions[test])
			self.cv_score/=nsplits
			print "cv_hamming_loss: "+str(self.cv_score)
			for i in range(0,nclasses):
				self.predictor[i].fit(self.train_features, targets[:,i])
				
		self.predict(nclasses)
		
	def get_features(self,path):
		self.check_feature_cache()
		if(self.feature_cache_exists):
			if(not self.prediction_cache_exists): # if we have the predictions we do not need the features
				self.read_cache_features()
		else:
			self.check_prediction_cache()
			self.read_features(path)
			nonzeros=preprocess.get_nonzero_variance(self.train_features)
			self.train_features=preprocess.remove_zero_variance(self.train_features,nonzeros)
			self.test_features=preprocess.remove_zero_variance(self.test_features,nonzeros)
			self.cache_features()
		
	def predict(self,nclasses):
		self.check_prediction_cache()
		if(self.prediction_cache_exists):
			self.read_cache_predictions()
		else:
			self.predictions=np.zeros((self.ntest,nclasses))
			for i in range(0,nclasses):
				self.predictions[:,i]=self.predictor[i].predict(self.test_features)
				self.cache_predictions()
			
	def cache_features(self):
		np.save(self.fname_cache_train,self.train_features)
		np.save(self.fname_cache_test,self.test_features)
	
	def cache_predictions(self):
		np.save(self.fname_cache_cv_pred,self.cv_predictions)
		np.save(self.fname_cache_pred,self.predictions)
	
	def check_feature_cache(self):
		self.feature_cache_exists=(os.path.isfile(self.fname_cache_train) and os.path.isfile(self.fname_cache_test))
		
	def check_prediction_cache(self):
		self.prediction_cache_exists=(os.path.isfile(self.fname_cache_pred) and os.path.isfile(self.fname_cache_cv_pred))
	
	def read_cache_features(self):
		self.train_features=np.load(self.fname_cache_train)
		self.test_features=np.load(self.fname_cache_test)
		
	def clear_cache_features(self):
		if(os.path.isfile(self.fname_cache_train)):
			os.remove(self.fname_cache_train)
		if(os.path.isfile(self.fname_cache_test)):
			os.remove(self.fname_cache_test)
		
	def read_cache_predictions(self):
		self.cv_predictions=np.load(self.fname_cache_cv_pred)
		self.predictions=np.load(self.fname_cache_pred)
		
	def clear_cache_predictions(self):
		if(os.path.isfile(self.fname_cache_cv_pred)):
			os.remove(self.fname_cache_cv_pred)
		if(os.path.isfile(self.fname_cache_pred)):
			os.remove(self.fname_cache_pred)

