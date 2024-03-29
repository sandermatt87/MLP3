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

import tensorflow as tf


plt.ion()

def main():
    ntrain=278 # number of training points
    ntest=138 # number of test points
    nclasses=3 # number of classes
    max_cubes=2 # maximum number of cubes in each direction
    nseg=1   #number of segments of the brain
    nmodel=1 #number of models to fit
    kfold_splits=10 # number of splits in the kfold cross validation
	
    print("number of models: ", nmodel)
	
    #initialize all models
    models=init_models.init(ntrain,ntest,max_cubes,nseg)

    #loop over all models, and calculate the features
    path = "C:/phd/MLcourse/segms/"
    for imodel in range(0,nmodel):
        models[imodel].get_features(path)
	
    #train the models using a DNN
    targets=parse.read_targets(ntrain,nclasses)
    #	for imodel in range(0,nmodel): 
    #		models[imodel].train(targets,kfold_splits,nclasses)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=22010)]
 
    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10, 20, 10], n_classes=3, model_dir="C:/phd/MLcourse/MLP3_working_code_kct/src/tmp2/")      
  
    classifier.fit(x=models[0].train_features, y=targets[:,0],  steps=500)
    
    accuracy_score = classifier.evaluate(x=models[0].train_features[250:277,:], y=targets[250:277,0])["accuracy"]
    print('======================================================')
    print('======================================================')
    print('Accuracy: {0:f}'.format(accuracy_score))
    print('======================================================')
    print('======================================================')
    
    preds=classifier.predict(models[0].train_features[200:277,:])
    
    print('======================================================')
    print('======================================================')
    print([preds])
    print('======================================================')
    print('======================================================')
    print(targets[200:277,0])
    print('======================================================')
    print('======================================================')

  
			
#    final_prediction=np.copy(models[0].predictions)
#    #write the final predictions to the csv file
#    fpredictions = open("../predictions.csv", 'w')
#    fpredictions.write("ID,Sample,Label,Predicted\n")
#    for i in range(0,ntest):
#        fpredictions.write(str(i*3)+","+str(i)+",gender,"+str(bool(final_prediction[i,0]))+"\n")
#        fpredictions.write(str(i*3+1)+","+str(i)+",age,"+str(bool(final_prediction[i,1]))+"\n")
#        fpredictions.write(str(i*3+2)+","+str(i)+",health,"+str(bool(final_prediction[i,2]))+"\n")
#        fpredictions.close()
	
if __name__ == "__main__":
	main()
	
