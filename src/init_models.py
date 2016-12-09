import voxel_model
import distance_segmentation

#initialize all models
def init(ntrain,ntest,max_cubes,nseg,nclasses):
	models=[]
	
	ncubes=1
	seg=1
	for seg in range(1,2):
		for ncubes in range(1,3):
			#models.append(distance_segmentation.distance_segmentation(ntrain,ntest,seg,[0.1,0.1,0.1],[1.0,1.0,1.0],ncubes,"seg"+str(seg)+"c"+str(ncubes)+"distance",nclasses))
			models.append(voxel_model.voxel_model(ntrain,ntest,seg,[50.0,50.0,50.0],[1.0,1.0,1.0],ncubes,2.2,"seg"+str(seg)+"c"+str(ncubes)+"voxels",nclasses))
	
	return models
