#!/usr/bin/python

import scipy

import model
import voxel_model
import distance_segmentation
import parse

ntrain=278
ntest=138
seg=1
x0=[50.0,50.0,50.0,1.0,1.0,1.0,2.2] #multiplicator of gamma, 
path = "../data/"
kfold_splits=10
nclasses=3
ncubes=1
targets=parse.read_targets(ntrain,nclasses)

def loss(x):
	print x
	method=voxel_model.voxel_model(ntrain,ntest,seg,[x[0],x[1],x[2]],[x[3],x[4],x[5]],ncubes,x[6],"seg"+str(seg)+"c"+str(ncubes)+"voxels")
	method.clear_cache_features()
	method.get_features(path)
	method.clear_cache_predictions()
	method.train(targets,kfold_splits,nclasses)
	return method.cv_score
	
scipy.optimize.minimize(loss,x0)

