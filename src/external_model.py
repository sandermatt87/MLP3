import model
import parse
import preprocess
import nibabel as nib
import numpy as np
import sklearn
from sklearn.model_selection  import KFold
from sklearn.svm import SVC

import preprocess

#reads the features from a npy matrix file (used for features calculated by a different code)
class canny(model.model):
    def __init__(self,ntrain,ntest,seg,ncubes,pos,cname, weight):
        model.model.__init__(self,ntrain,ntest,seg,ncubes,pos,cname, weight)
        self.predictor = SVC(probability=True,C=0.01,gamma=0.000000001)
        self.custom_svm=True

    def read_features(self,path):
        print("external model, please write te features directly to the cache")

class hog(model.model):
    def __init__(self,ntrain,ntest,seg,ncubes,pos,cname, weight):
        model.model.__init__(self,ntrain,ntest,seg,ncubes,pos,cname, weight)
        self.predictor = SVC(probability=True,C=1)

    def read_features(self,path):
        print("external model, please write te features directly to the cache")
