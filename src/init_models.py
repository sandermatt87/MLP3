import voxel_model
import distance_segmentation

#initialize all models
def init(ntrain,ntest,max_cubes,nseg):
	models=[]
	
	ncubes=1
	seg=1
	models.append(voxel_model.voxel_model(ntrain,ntest,1,1,"seg"+str(seg)+"c"+str(ncubes)+"voxels"))

	return models
