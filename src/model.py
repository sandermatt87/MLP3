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
	predictor=None
	predictions=None
	cv_predictions=None
	fname_cache_train=None
	fname_cache_test=None
	fname_cache_pred=None
	fname_cache_cv_pred=None
	feature_cache_exists=False
	prediction_cache_exists=False
	gamma=-1
	seg=-1
	slack=-1
	
	def __init__(self,ntrain,ntest,seg,gamma,slack,cname,nclasses):
		self.ntrain=ntrain
		self.ntest=ntest
		self.fname_cache_train="./cache/features/"+cname+"train.npy"
		self.fname_cache_test="./cache/features/"+cname+"test.npy"
		self.fname_cache_pred="./cache/predictions/"+cname+"pred.npy"
		self.fname_cache_cv_pred="./cache/predictions/"+cname+"cv_pred.npy"
		self.seg=seg
		self.gamma=gamma
		self.slack=slack
		self.nclasses=nclasses
		self.predictor=[]
		for i in range(0,nclasses):
			self.predictor.append(None)
	
	def train(self,targets,nsplits):
		#new_targets=utils.to_single_class(targets)
		
		print "training model with: ",self.fname_cache_train
		self.check_prediction_cache()
		if(self.prediction_cache_exists):
			self.read_cache_predictions()
			self.cv_score+=hamming_loss(targets,np.round(self.cv_predictions).astype(int))
			print "cv_hamming_loss: "+str(self.cv_score)
		else:
			for i in range(0,self.nclasses):
				if(self.predictor[i] is None):
					self.predictor[i]=SVC(C=self.slack[i],gamma=self.gamma[i],probability=True)
			np.random.seed(1237)
			self.cv_predictions=np.copy(targets)*0
			self.cv_score=0
			kf = KFold(n_splits=nsplits)
			for train, test in kf.split(self.train_features):
				for i in range(0,self.nclasses):
					self.predictor[i].fit(self.train_features[train],targets[train,i])
					self.cv_predictions[test,i]=self.predictor[i].predict_proba(self.train_features[test])[:,1]
					#self.cv_predictions[test,i]=self.predictor[i].predict(self.train_features[test])
			print "cv_hamming_loss: "+str(hamming_loss(targets[:,0],np.round(self.cv_predictions[:,0]).astype(int))),str(hamming_loss(targets[:,1],np.round(self.cv_predictions[:,1]).astype(int))),str(hamming_loss(targets[:,2],np.round(self.cv_predictions[:,2]).astype(int)))
			for i in range(0,self.nclasses):
				self.predictor[i].fit(self.train_features, targets[:,i])
				
		self.predict(targets)
		
	def get_features(self,path):
		self.check_feature_cache()
		if(self.feature_cache_exists):
			if(not self.prediction_cache_exists): # if we have the predictions we do not need the features
				self.read_cache_features()
		else:
			self.check_prediction_cache()
			self.read_features(path)
			self.cache_features()
		
	def predict(self,targets):
		if(self.prediction_cache_exists):
			self.read_cache_predictions()
		else:
			self.predictions=np.zeros((self.ntest,self.nclasses))
			for i in range(0,self.nclasses):
				self.predictions[:,i]=self.predictor[i].predict_proba(self.test_features)[:,1]
				#self.predictions[:,i]=self.predictor[i].predict(self.test_features)
				if(True):
					#print self.predictor[i].predict(self.train_features)-targets[:,i]
					#print np.sum(np.abs(self.predictor[i].predict(self.train_features)-targets[:,i]))
					#print self.cv_predictions[:,i]-targets[:,i]
					print np.sum(np.abs(self.cv_predictions[:,i]-targets[:,i]))
					#print self.predictions[:,i]
					#print self.predictions.shape
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

