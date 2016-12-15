#!/usr/bin/python

import matplotlib.pyplot as plt
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

import consts
import init_models
import parse
import model
import init_models

import auto_tune
import matplotlib.pyplot as plt

plt.ion()


def main():
    ntrain = 278  # number of training points
    ntest = 138  # number of test points
    nclasses = 3  # number of classes
    max_cubes = 3  # maximum number of cubes in each direction
    nseg = 1  # number of segments of the brain
    nmodel = 1  # number of models to fit
    kfold_splits = 10  # number of splits in the kfold cross validation

    print("number of models: ", nmodel)
    at=auto_tune.auto_tune(ntrain=100,     ntest=1,     seg=nseg, ncubes=max_cubes) #use only 100 training samples to make optimization faster
    atres = at.tune_parameters()

    # initialize all models
                          #(ntrain,  ntest,  max_cubes,  nseg,  slack_mod1,     slack_mod2,     smoothening )
    models=init_models.init(ntrain,  ntest,  max_cubes,  nseg,  atres[0][0][0], atres[1][0][0], atres[0][0][1],       nclasses)

    # loop over all models, and calculate the features
    if consts.modeKCT:
        path = "C:/phd/MLcourse/segms/"
    else:
        path = "../data/"
    for imodel in range(0, nmodel):
        models[imodel].clear_cache_features() # necessary, since
        models[imodel].clear_cache_predictions()  # necessary, since
        models[imodel].get_features(path)

    # train the models
    targets = parse.read_targets(ntrain, nclasses)
    for imodel in range(0, nmodel):
        models[imodel].train_multilabel(targets, kfold_splits, nclasses)

    final_prediction = np.copy(models[0].predictions)
    # write the final predictions to the csv file
    if consts.modeKCT:
        fpredictions = open("C:/Users/ktezcan/Desktop/unnecessary_stuff/predictions.csv", 'w')
    else:
        fpredictions = open("../predictions.csv", 'w')
    fpredictions.write("ID,Sample,Label,Predicted\n")
    for i in range(0, ntest):
        fpredictions.write(str(i * 3) + "," + str(i) + ",gender," + str(bool(final_prediction[i, 0])) + "\n")
        fpredictions.write(str(i * 3 + 1) + "," + str(i) + ",age," + str(bool(final_prediction[i, 1])) + "\n")
        fpredictions.write(str(i * 3 + 2) + "," + str(i) + ",health," + str(bool(final_prediction[i, 2])) + "\n")
    fpredictions.close()


if __name__ == "__main__":
    main()

