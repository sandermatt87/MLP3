import numpy as np

import voxel_model
import distance_segmentation
import histogram_model
import fft_model

#initialize all models
def init(ntrain,ntest,max_cubes,nseg,nclasses):
	models=[]
	
	ncubes=1
	seg=1
	#ugly way of determining optimal slack variales, but until we have auto tuning we need to fix them ourselves
	gamma_voxel=np.zeros((4,nclasses))
	slack_voxel=np.zeros((4,nclasses))
	gamma_dist=np.zeros((4,nclasses))
	slack_dist=np.zeros((4,nclasses))
	gamma_fft=np.zeros((4,nclasses))
	slack_fft=np.zeros((6,nclasses))
	gamma_hist=np.zeros(nclasses)
	slack_hist=np.zeros(3)
	gamma_dist[0,:]=[0.000001,0.00001,0.00001]
	gamma_dist[1,:]=[0.00000018,0.000004,0.0000035]
	gamma_dist[2,:]=[0.0000001,0.000002,0.000002]
	slack_dist[:,:]=[1,1,1]
	gamma_voxel[0,:]=[0.0001,0.00001,0.00001]
	gamma_voxel[1,:]=[0.00003,0.000003,0.000004]
	gamma_voxel[2,:]=[0.00001,0.000001,0.000002]
	slack_voxel[:,:]=[1,1,1]
	gamma_fft[0,:]=[0.00000005,0.00000005,0.00000005] 0.388489208633 0.525179856115 0.241007194245
	gamma_fft[0,:]=[0.00000002,0.00000002,0.00000002] 0.302158273381 0.0611510791367 0.190647482014
	gamma_fft[0,:]=[0.00000001,0.00000001,0.00000001] 
	gamma_fft[1,:]=[0.000000006,0.000000006,0.000000006] 0.219424460432 0.0503597122302 0.172661870504
	gamma_fft[1,:]=[0.000000003,0.000000003,0.000000003] 0.219424460432 0.0539568345324 0.172661870504
	gamma_fft[1,:]=[0.000000002,0.000000008,0.000000001] 
	gamma_fft[2,:]=[0.000000003,0.000000003,0.000000003] 0.363309352518 0.0575539568345 0.172661870504
	gamma_fft[2,:]=[0.000000001,0.000000001,0.000000002] 0.36690647482 0.089928057554 0.172661870504
	gamma_fft[2,:]=[0.00000001, 0.000000005,0.00000001 ] 
	gamma_fft[3,:]=[0.000000001 ,0.000000001 ,0.000000002] 0.388489208633 0.0827338129496 0.169064748201
	gamma_fft[3,:]=[0.000000003 ,0.000000003 ,0.000000003] 0.388489208633 0.0827338129496 0.172661870504
	gamma_fft[3,:]=[0.0000000003,0.0000000003,0.000000001] 
	gamma_fft[4,:]=[0.000000003,0.000000003,0.000000003]
	gamma_fft[5,:]=[0.000000003,0.000000003,0.000000003] 
	slack_fft[:,:]=[1,1,1]
	gamma_hist=[0.000003,0.0000003,0.0000004]
	slack_hist=[1,1,1]
	
	for seg in range(1,nseg+1):
		for ncubes in range(1,max_cubes+1):
	#		models.append(distance_segmentation.distance_segmentation(ntrain,ntest,seg,gamma_dist[ncubes-1,:],slack_dist[ncubes-1,:],ncubes,"seg"+str(seg)+"c"+str(ncubes)+"distance",nclasses))
	#		models.append(voxel_model.voxel_model(ntrain,ntest,seg,gamma_voxel[ncubes-1,:],slack_voxel[ncubes-1,:],ncubes,2.2,"seg"+str(seg)+"c"+str(ncubes)+"voxels",nclasses))
			models.append(fft_model.fft_model(ntrain,ntest,seg,gamma_fft[ncubes-1,:],slack_fft[ncubes-1,:],ncubes,2.2,"seg"+str(seg)+"c"+str(ncubes)+"fft",nclasses))
	#for seg in range(1,nseg+1):
	#	for ncubes in range(1,8):
	#		models.append(histogram_model.histogram_model(ntrain,ntest,seg,gamma_hist,slack_hist,ncubes,False,"seg"+str(seg)+"c"+str(ncubes)+"histogram",nclasses))
	#		models.append(histogram_model.histogram_model(ntrain,ntest,seg,gamma_hist,slack_hist,ncubes,True,"seg"+str(seg)+"c"+str(ncubes)+"histogramSmooth",nclasses))
	return models
