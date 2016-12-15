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
    invert=False

    # use these to define the ranges for the grid search, each parameter has a slice, e.g. (range(slack) range(smoothing))
    range_ds = (slice(5.0, 5.1, 0.1), slice(2.0, 3.0, 1))
    range_vm = (slice(5.0, 5.1, 0.1), slice(2.0, 3.0, 1))

    def __init__(self,ntrain,ntest,seg, ncubes):
        self.ntrain = ntrain
        self.ntest = ntest
        self.seg = seg
        self.ncubes=ncubes


    def tune_parameters(self):

        def loss_vm(x):
            att = time.time()       #(self,  ntrain,      ntest,      seg,      gamma_scale,  slack,   ncubes,         smoothening,  cname,   nclasses)
            method = voxel_model.voxel_model(self.ntrain, self.ntest, self.seg, 0,            x[0],    self.ncubes,    x[1],         "seg" + str(self.seg) + "c" + str(self.ncubes) + "voxels", self.nclasses)
            print("gamma scale: ", method.gamma_scale)
            print("slack: ", method.slack)
            print("smoothening: ", method.smoothening)
            method.clear_cache_features()
            method.get_features(self.path)
            method.clear_cache_predictions()
            method.train_multilabel(self.targets, self.kfold_splits, self.nclasses) # do multilabel classification
            method.clear_cache_features()
            print('Loss funct loop time: ' + str(time.time() - att))
            return method.cv_score

        def loss_ds(x):
            att = time.time()                           #(self,  ntrain,      ntest,      seg,      gamma_scale,  slack,  ncubes,       cname,nclasses,                                              invert=False)
            method = distance_segmentation.distance_segmentation(self.ntrain, self.ntest, self.seg, 0,            x[0],   self.ncubes,  "seg" + str(self.seg) + "c" + str(self.ncubes) + "distance", self.invert)
            print("gamma scale: ", method.gamma_scale)
            print("slack: ", method.slack)
            method.clear_cache_features()
            method.get_features(self.path)
            method.clear_cache_predictions()
            method.train_multilabel(self.targets, self.kfold_splits, self.nclasses) # do multilabel classification
            method.clear_cache_features()
            print('Loss funct loop time: ' + str(time.time() - att))
            return method.cv_score

        #optimize paramters by first doing a coarse grid search and then refining around the lowest point by fminsearch
        rr_vm = self.range_vm
        resbrute_vm = scipy.optimize.brute(loss_vm, rr_vm, full_output=True, finish=None, disp=True) #scipy.optimize.fmin

        rr_ds = self.range_ds
        resbrute_ds = scipy.optimize.brute(loss_ds, rr_ds, full_output=True, finish=None, disp=True)


        return [resbrute_vm, resbrute_ds]