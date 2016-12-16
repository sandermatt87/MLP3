import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold

import mixing

#the cube predictor allows for a model to make predictions fur subsets of its features. For the cube predictor to work the subsets need to be ordered sequentially in the feature vector. Each subset can be viewd at as a separate model. The cube predictor will internally mix the models ans expose one meta-model to the outside. The model tries to emulate the interface for a sklearn method such as svm.
class cube_predictor():
    ncubes=-1  # the number of subsets of the features
    
    #svm parameters
    gamma_scale=-1
    slack=-1
    
    
    predictors=[] # a list of the predictors for each subset, currently we use svm's
    cube_size=-1 # length of the feature vector for a subset/cube
    cv_opt=False #If true the models are mixed using cross validation optimization, of False they are mixed evenly
    cv_predictions=None #the prediction each submodel makes under crossvalidation
    nsplits=3 #splits in the kfold cross validation optimization
    weights=[]

    def __init__(self,ncubes,gamma_scale,slack,cv_opt=False):
        self.ncubes=ncubes**3
        self.slack=slack
        self.gamma_scale=gamma_scale
        self.cv_opt=cv_opt

    def fit(self,features,targets):
        cube_size=int(round(float(features.shape[1])/(self.ncubes)))
        for i in range(0,self.ncubes):
            self.predictors.append(SVC(C=self.slack,gamma=self.gamma_scale/features.shape[1],probability=True))
        if(not self.cv_opt):
            self.weights=np.zeros(self.ncubes)+1.0/self.ncubes
        else:
            self.cv_predictions=np.zeros((targets.shape[0],self.ncubes))
            kf = KFold(n_splits=self.nsplits)
            for cube in range(0,self.ncubes):
                cube_features=features[:,i*cube_size:(i+1)*cube_size]
                for train, test in kf.split(cube_features):
                    self.predictors[cube].fit(cube_features[train],targets[train])
                    self.cv_predictions[test,cube]=self.predictors[cube].predict_proba(cube_features[test])[:,1]
            self.weights=mixing.cv_optimization(self.cv_predictions,targets,self.ncubes)
            #print self.weights
        for i in range(0,self.ncubes):
            self.predictors[i].fit(features[:,i*cube_size:(i+1)*cube_size],targets)
    def predict(self,features):
        predictions=self.predict_proba(features)
        final_result=np.round(predictions)
        return final_result

    def predict_proba(self,features):
        cube_size=int(round(float(features.shape[1])/(self.ncubes)))
        results=[]
        for i in range(0,self.ncubes):
            results.append(self.predictors[i].predict_proba(features[:,i*cube_size:(i+1)*cube_size])[:,1])
        final_result=np.zeros(results[0].shape)
        for i in range(0,results[0].shape[0]):
            for j in range(0,self.ncubes):
                final_result[i]+=results[j][i]*self.weights[j]
            final_result[i]=final_result[i]
        return final_result

