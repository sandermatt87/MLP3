import voxel_model
import distance_segmentation

#initialize all models
def init(ntrain,ntest,max_cubes,nseg,nclasses):
	models=[]
	
	ncubes=1
	seg=1
	gamma_distance_seg=0.0000003
	slack_distance_seg=80
	gamma_voxel=0.00001
	slack_voxel=80
	for seg in range(1,2):
		for ncubes in range(1,max_cubes+1):
			models.append(distance_segmentation.distance_segmentation(ntrain,ntest,seg,[gamma_distance_seg,gamma_distance_seg,gamma_distance_seg],[slack_distance_seg,slack_distance_seg,slack_distance_seg],ncubes,"seg"+str(seg)+"c"+str(ncubes)+"distance",nclasses))
			models.append(voxel_model.voxel_model(ntrain,ntest,seg,[gamma_voxel,gamma_voxel,gamma_voxel],[slack_voxel,slack_voxel,slack_voxel],ncubes,2.2,"seg"+str(seg)+"c"+str(ncubes)+"voxels",nclasses))
	
	return models
