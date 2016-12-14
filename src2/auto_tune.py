#!/usr/bin/python

import scipy

import model
import voxel_model
import distance_segmentation
import parse

import consts

import time
import numpy as np


class auto_tune:
    ntrain = 100
    ntest = 1
    seg = 1
    if consts.modeKCT:
        path = "C:/phd/MLcourse/segms/"
    else:
        path = "../data/"
    kfold_splits = 10
    nclasses = 3
    ncubes = 3
    targets = parse.read_targets(ntrain, nclasses)

    range_ds = (slice(5.0, 5.4, 0.1), slice(2.0, 3.0, 1))
    range_vm = (slice(5.0, 5.2, 0.1), slice(2.0, 3.0, 1))



    def __init__(self,ntrain,ntest,seg):
        self.ntrain = ntrain
        self.ntest = ntest
        self.seg = seg




    def tune_parameters(self):

        def loss_vm(x):
            att = time.time()
            method = voxel_model.voxel_model(self.ntrain, self.ntest, self.seg, 0, x[0], self.ncubes, x[1],
                                             "seg" + str(self.seg) + "c" + str(self.ncubes) + "voxels")
            print("gamma scale: ", method.gamma_scale)
            print("slack: ", method.slack)
            print("smoothening: ", method.smoothening)
            method.clear_cache_features()
            method.get_features(self.path)
            method.clear_cache_predictions()
            method.train_multilabel(self.targets, self.kfold_splits, self.nclasses) # do multilabel classification
            print('Loss funct loop time: ' + str(time.time() - att))
            return method.cv_score

        def loss_ds(x):
            att = time.time()
            method = distance_segmentation.distance_segmentation(self.ntrain, self.ntest, self.seg, 0, x[0], self.ncubes, "seg" + str(self.seg) + "c" + str(self.ncubes) + "distance")
            print("gamma scale: ", method.gamma_scale)
            print("slack: ", method.slack)
            method.clear_cache_features()
            method.get_features(self.path)
            method.clear_cache_predictions()
            method.train_multilabel(self.targets, self.kfold_splits, self.nclasses) # do multilabel classification
            print('Loss funct loop time: ' + str(time.time() - att))
            return method.cv_score

        #optimize paramters by first doing a coarse grid search and then refining around the lowest point by fminsearch
        rr_ds = self.range_ds
        resbrute_ds = scipy.optimize.brute(loss_ds, rr_ds, full_output=True, finish=scipy.optimize.fmin, disp=True)

        rr_vm = self.range_vm
        resbrute_vm = scipy.optimize.brute(loss_vm, rr_vm, full_output=True, finish=scipy.optimize.fmin, disp=True)



        return [resbrute_vm, resbrute_ds]