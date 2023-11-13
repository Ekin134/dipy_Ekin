import dipy.direction.peaks
import dipy.data
import numpy as np
from dipy.data import get_sphere
from dipy.reconst import shm
from dipy.core.sphere import HemiSphere

# List of spheres symmetric724
#sphere = dipy.data.get_sphere("repulsion724")
sphere = dipy.data.get_sphere("symmetric724")
#sphere_724 = get_sphere("repulsion724")             #average min angle: 7.76deg
sphere_724 = get_sphere("symmetric724") 
#sphere_2890 = sphere_724.subdivide(1) 
sphere_2890 = sphere_724.subdivide(1) 
sphere_error= HemiSphere.from_sphere(sphere_2890)

matrix = np.dot(sphere_error.vertices, sphere_error.vertices.T)
matrix[matrix>1]=1
matrix[matrix<-1]=-1

# matrix representing all the angles between each pair of vertexes on the sphere (in degreees, shape: 724x724)  
ang_matrix = np.rad2deg(np.arccos(matrix)) 

#sphere must be the same as for ang_matrix, sh_order should match the input SH
def get_angular_error(sh_GT, sh_prediction, sh_order):
    
    # compute the spherical signal of the ODF from the spherical harmonic coefficients
    sf_GT = shm.sh_to_sf(sh_GT, sphere=sphere_error, sh_order=sh_order, basis_type='tournier07')
    sf_prediction = shm.sh_to_sf(sh_prediction, sphere=sphere_error, sh_order=sh_order, basis_type='tournier07')
    
    #return the vertex indices corresponding to local maxima of both ODF
    _, _, peaks_GT_ind = dipy.direction.peaks.peak_directions(sf_GT, sphere_error, relative_peak_threshold=.2, min_separation_angle=15)
    _, _, peaks_prediction_ind = dipy.direction.peaks.peak_directions(sf_prediction, sphere_error, relative_peak_threshold=.2, min_separation_angle=15)
    
    # find the minimum angle between the T and the prediction
    errors = []
    
    for i in peaks_GT_ind:
        err = []
        
        for j in peaks_prediction_ind:
            err.append(ang_matrix[i,j])
            err.append(180-ang_matrix[i,j])
        
        if err != []:
            errors.append(np.min(err))
            #print("errors",errors)
        
    return np.mean(errors), np.size(peaks_GT_ind) , np.size(peaks_prediction_ind)
