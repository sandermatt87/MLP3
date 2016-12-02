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
	models=init_models.init(ntrain,ntest,max_cubes,nseg)

	#loop over all models, and calculate the features
	for imodel in range(0,nmodel): 
		path = "../data/"
		models[imodel].get_features(path)
	
	#train the models
	targets=parse.read_targets(ntrain,nclasses)
	for imodel in range(0,nmodel): 
		models[imodel].train(targets,kfold_splits,nclasses)
			
	final_prediction=np.copy(models[0].predictions)
	#write the final predictions to the csv file
	fpredictions = open("../predictions.csv", 'w')
	fpredictions.write("ID,Prediction\n")
	for i in range(0,ntest):
		fpredictions.write(str(i+1)+","+str(final_prediction[i])+"\n")
	fpredictions.close()
	
if __name__ == "__main__":
	main()
	