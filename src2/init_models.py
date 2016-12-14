import voxel_model
import distance_segmentation

#initialize all models
def init(ntrain,ntest,max_cubes,nseg, slack_mod1, slack_mod2, smoothening ):
    models=[]

    ncubes=4
    seg=1
    for seg in range(1,2):
         for ncubes in range(0,1):
             models.append(voxel_model.voxel_model(ntrain,ntest,seg,0,slack_mod1,ncubes,smoothening,"seg"+str(seg)+"c"+str(ncubes)+"voxels"))
             models.append(distance_segmentation.distance_segmentation(ntrain,ntest,seg,0,slack_mod2,ncubes,"seg"+str(seg)+"c"+str(ncubes)+"distance"))

    return models
