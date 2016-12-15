import nibabel as nib
import numpy as np
from sklearn.model_selection import KFold

import preprocess

import time
import cube_predictor
import model

import parse

#this model uses the voxels as input
class voxel_model(model.model):
    ncubes=1
    smoothening=-1

    def __init__(self,ntrain,ntest,seg,gamma_scale,slack,ncubes,smoothening,cname,nclasses):
        model.model.__init__(self,ntrain,ntest,seg,gamma_scale,slack,cname,nclasses)
        self.ncubes=ncubes
        self.smoothening=smoothening
        #if(ncubes>1):
        #    for i in range(0,nclasses):
        #        self.predictor[i]=cube_predictor.cube_predictor(ncubes,gamma_scale[i],slack[i],cv_opt=True)

    def read_features(self,path):
        tmp=[]
        train_path=path+"set_train/c"+str(self.seg)+"train_"
        print( "reading train images")
        at=time.time()
        for i in range(0,self.ntrain):
            filename=train_path+str(i+1)+".nii"
            array=parse.voxels_from_image(filename,smoothening_width=self.smoothening)
            tmp.append(preprocess.features1D(array,self.ncubes))
        self.train_features=np.zeros((self.ntrain,tmp[0].size))
        for i in range (0,self.ntrain):
            self.train_features[i,:]=tmp[i]
        print('Elapsed time: ' + str(time.time()-at))
        tmp=[]
        test_path=path+"set_test/c"+str(self.seg)+"test_"
        print( "reading test images")
        at=time.time()
        for i in range(0,self.ntest):
            filename=test_path+str(i+1)+".nii"
            array=parse.voxels_from_image(filename,self.ncubes,smoothening_width=self.smoothening)
            tmp.append(preprocess.features1D(array,self.ncubes))
        self.test_features=np.zeros((self.ntest,tmp[0].size))
        for i in range (0,self.ntest):
            self.test_features[i,:]=tmp[i]
        print('read test fetures: ' + str(time.time()-at))
        if(self.ncubes==1):
            #Compress the data. This can only be done without cubes, since otherwise the format gets messed up
            nonzeros=preprocess.get_nonzero_variance(self.train_features)
            self.train_features=preprocess.remove_zero_variance(self.train_features,nonzeros)
            self.test_features=preprocess.remove_zero_variance(self.test_features,nonzeros)

        
    def read_features_train(self,path):
        tmp=[]
        train_path=path+"set_train/c"+str(self.seg)+"train_"
        print( "reading train images")
        at=time.time()
        for i in range(0,self.ntrain):
            filename=train_path+str(i+1)+".nii"
            array=parse.voxels_from_image(filename,self.ncubes,smoothening_width=self.smoothening)
            tmp.append(np.reshape(array,array.size))
        self.train_features=np.zeros((self.ntrain,tmp[0].size))
        for i in range (0,self.ntrain):
            self.train_features[i,:]=tmp[i]
        print('reading images done!')
        print('Elapsed time: ' + str(time.time()-at))
       
    def read_features_test(self,path):
        tmp=[]
        test_path=path+"set_test/c"+str(self.seg)+"test_"
        print( "reading test images")
        at=time.time()
        for i in range(0,self.ntest):
            filename=test_path+str(i+1)+".nii"
            array=parse.voxels_from_image(filename,self.ncubes,smoothening_width=self.smoothening)
            tmp.append(np.reshape(array,array.size))
        self.test_features=np.zeros((self.ntest,tmp[0].size))
        for i in range (0,self.ntest):
            self.test_features[i,:]=tmp[i]
        print('read test features: ' + str(time.time()-at))
