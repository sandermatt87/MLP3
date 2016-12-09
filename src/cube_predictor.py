import numpy as np
from sklearn.svm import SVC


class cube_predictor():
	ncubes=-1
	gamma_scale=-1
	slack=-1
	predictors=[]
	shape=None
	cube_size=-1
	
	def __init__(self,ncubes,gamma_scale,slack):
		self.ncubes=ncubes**3
		self.slack=slack
		self.gamma_scale=gamma_scale
		
	def fit(self,features,targets,probability=False):
		cube_size=int(round(float(features.shape[1])/(self.ncubes)))
		for i in range(0,self.ncubes):
			self.predictors.append(SVC(C=self.slack,gamma=self.gamma_scale/features.shape[1],probability=probability))
			self.predictors[i].fit(features[:,i*cube_size:(i+1)*cube_size],targets)
	def predict(self,features):
		cube_size=int(round(float(features.shape[1])/(self.ncubes)))
		results=[]
		for i in range(0,self.ncubes):
			results.append(self.predictors[i].predict(features[:,i*cube_size:(i+1)*cube_size]))
		final_result=np.zeros(results[0].shape)
		for i in range(0,results[0].shape[0]):
			for j in range(0,self.ncubes):
				final_result[i]+=results[j][i]
			final_result[i]=int(round(float(final_result[i])/self.ncubes))
		return final_result
		
	def predict_proba(self,features):
		cube_size=int(round(float(features.shape[1])/(self.ncubes)))
		results=[]
		for i in range(0,self.ncubes):
			results.append(self.predictors[i].predict_proba(features[:,i*cube_size:(i+1)*cube_size]))
		final_result=np.zeros(results[0].shape)
		for i in range(0,results[0].shape[0]):
			for j in range(0,self.ncubes):
				final_result+=result[j][i]
			final_result[i]=final_result[i]/self.ncubes
		return final_result
