import voxel_model
import distance_segmentation

#initialize all models
def init(ntrain,ntest,max_cubes,nseg, slack_mod1, slack_mod2, smoothening, nclasses ):
    models=[]

    for seg in range(1,2):
         for ncubes in range(1,max_cubes+1):
             #                             (self,  ntrain,  ntest,  seg,  gamma_scale,   slack,       ncubes,   smoothening,  cname,                                    nclasses)
             models.append(voxel_model.voxel_model(ntrain,  ntest,  seg,  0,             slack_mod1,  ncubes,   smoothening,  "seg"+str(nseg)+"c"+str(ncubes)+"voxels", nclasses))
             models.append(distance_segmentation.distance_segmentation(ntrain,ntest,nseg,0,slack_mod2,ncubes,"seg"+str(nseg)+"c"+str(ncubes)+"distance", nclasses))

    return models
