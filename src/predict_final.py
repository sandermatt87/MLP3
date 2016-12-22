#!/usr/bin/python

import sys
from sklearn.decomposition import PCA
import numpy as np
import csv
import timeit
import os
import scipy
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

import parse
import model
import init_models
import mixing

import matplotlib.pyplot as plt


plt.ion()

def main():
	ntrain=278 # number of training points
	ntest=138 # number of test points
	nclasses=3 # number of classes
	max_cubes=2 # maximum number of cubes in each direction
	nseg=1   #number of segments of the brain
	nmodel=1 #number of models to fit
	kfold_splits=10 # number of splits in the kfold cross validation
	
	print "number of models: ", nmodel
	
	#initialize all models
	models=init_models.init(ntrain,ntest,max_cubes,nseg,nclasses)

	#loop over all models, and calculate the features
	path = "../data/"
	for imodel in range(0,nmodel):
		models[imodel].get_features(path)
	
	#train the models
	targets=parse.read_targets(ntrain,nclasses)
	for imodel in range(0,nmodel): 
		models[imodel].train(targets,kfold_splits)
		
	#mix the models
	if(nmodel>1):
		final_prediction=np.zeros((ntest,nclasses))
		for iclass in range(0,nclasses):
			cv_predictions=np.zeros((ntrain,nmodel))
			for imodel in range(0,nmodel):
				cv_predictions[:,imodel]=models[imodel].cv_predictions[:,iclass]
			print cv_predictions.shape
			weights=mixing.cv_optimization(cv_predictions,targets[:,iclass],nmodel)
	else:
		final_prediction=np.copy(models[0].predictions)
	#write the final predictions to the csv file
	fpredictions = open("../predictions.csv", 'w')
	fpredictions.write("ID,Sample,Label,Predicted\n")
	for i in range(0,ntest):
		fpredictions.write(str(i*3)+","+str(i)+",gender,"+str(bool(final_prediction[i,0]))+"\n")
		fpredictions.write(str(i*3+1)+","+str(i)+",age,"+str(bool(final_prediction[i,1]))+"\n")
		fpredictions.write(str(i*3+2)+","+str(i)+",health,"+str(bool(final_prediction[i,2]))+"\n")
	fpredictions.close()
	
if __name__ == "__main__":
	main()
	
