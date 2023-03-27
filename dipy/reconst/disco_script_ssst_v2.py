"SSST"
#import necessary packages
import numpy as np
import nibabel as nib
from dipy.denoise.localpca import mppca
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.viz import window, actor
from dipy.reconst.csdeconv import (response_from_mask_ssst)
from dipy.sims.voxel import single_tensor_odf
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.data import get_sphere
from dipy.sims.phantom import add_noise
from skimage.restoration import (calibrate_denoiser,
                                 denoise_tv_chambolle,
                                 denoise_bilateral,
                                 denoise_wavelet,
                                 estimate_sigma,
                                 denoise_nl_means)

def average(sh_coeffs):
    "Averaging local neighbourhood around a central voxel in a cubic patch."
    x, y, z, v = np.shape(sh_coeffs)[0:4]
    K = np.zeros([x+2*1, y+2*1, z+2*1, v])
    K[1:x+1, 1:y+1, 1:z+1, :] = sh_coeffs
    average_sh_coeff = np.zeros([x,y,z,v])

    for a in range(x):
        for b in range(y):
            for c in range(z):
                neighbourhood = np.array([K[a+1,b,c,:],K[a+1,b+1,c,:],K[a+1, b+2,c,:],K[a+1,b,c+1,:]\
                    ,K[a+1,b+1,c+1,:],K[a+1,b+2,c+1,:],K[a+1,b,c+2,:],K[a+1,b+1,c+2,:],K[a+1,b+2,c+2,:]\
                        ,K[a,b+1,c,:],K[a+2,b+1,c,:],K[a,b+1,c+1,:],K[a+2,b+1,c+1,:],K[a,b+1,c+2,:]\
                            ,K[a+2,b+1,c+2,:],K[a+2,b+2,c+1,:],K[a+2,b,c+1,:],K[a,b+2,c+1,:],K[a,b,c+1,:]\
                                ,K[a+2,b+2,c+2,:],K[a+2,b+2,c,:],K[a,b+2,c+2,:],K[a,b+2,c,:],K[a+2,b,c+2,:]\
                                    ,K[a+2,b,c,:],K[a,b,c+2,:],K[a,b,c,:]])
                
                average_sh_coeff[a,b,c,:] = np.sum(neighbourhood, 0)/ np.count_nonzero(neighbourhood[:,0])
                
    return average_sh_coeff


sphere = get_sphere('symmetric724')

#load data, bvals, bvecs and mask. Create gradient table
data, affine = load_nifti(r"/home/ekin/Documents&code/Noisy Disco Data/snr15/data_snr15.nii.gz")
bvals, bvecs = read_bvals_bvecs(r"/home/ekin/Documents&code/datasets/disco1_ekin/1_shell/DiSCo_1_shell.bvals", r"/home/ekin/Documents&code/datasets/disco1_ekin/1_shell/DiSCo_1_shell.bvecs")
mask, affine = load_nifti(r"/home/ekin/Documents&code/datasets/disco1_ekin/DiSCo1_mask.nii.gz")

gtab = gradient_table(bvals, bvecs)

# infer response from mask
response, ratio = response_from_mask_ssst(gtab, data, mask)
scene = window.Scene()
evals = response[0]
evecs = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T

rho = 1
#fODF reconstruction
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
csd_fit = csd_model.fit(data)
print("Checkpoint-1")
sh_coeff = csd_fit.shm_coeff

#convert estimated odfs to Tournier basis
from basis_conversion import convert_to_mrtrix
order_sh = 8
conversion_matrix = convert_to_mrtrix(order_sh)
tournier_sh_coeff = np.dot(sh_coeff, conversion_matrix.T)

from shm import mean_square_error, angular_correlation
ground_sh, _ = load_nifti(r"/home/ekin/Documents&code/datasets/disco1_ekin/DiSCo1_Strand_ODFs.nii.gz")
ground_sh = ground_sh[:,:,:,0:sh_coeff.shape[3]]

mse_conventional = mean_square_error(tournier_sh_coeff, ground_sh, mask)
acc_conventional = angular_correlation(tournier_sh_coeff, ground_sh, mask)

#SH coefficients denoising
#sh_coeff = average(sh_coeff)
for voxelt in range(sh_coeff.shape[3]):
    data_vol  = np.squeeze(sh_coeff[:,:,:,voxelt])
    sigma_est = np.mean(estimate_sigma(data_vol, channel_axis=None))
    #print("Sigma Estimated: ", sigma_est)
    sh_coeff[:,:,:,voxelt] = denoise_tv_chambolle(data_vol, weight=1.0*sigma_est, eps=0.0002, max_num_iter=200, channel_axis=None)

fodf_amp = np.zeros([sh_coeff.shape[0],sh_coeff.shape[1],sh_coeff.shape[2],csd_model.B_reg.shape[0]])
for x in range(sh_coeff.shape[0]):
    for y in range(sh_coeff.shape[1]):
        for z in range(sh_coeff.shape[2]):
            fodf_amp[x,y,z,:] = np.dot(csd_model.B_reg, sh_coeff[x,y,z,:])
            
fodf_amp[fodf_amp < 0] = 0

thr_sh_coeff = np.zeros(sh_coeff.shape)
for x in range(sh_coeff.shape[0]):
    for y in range(sh_coeff.shape[1]):
        for z in range(sh_coeff.shape[2]):
            thr_sh_coeff[x,y,z,:] = np.dot((np.linalg.inv(np.transpose(csd_model.B_reg) @ csd_model.B_reg) @ np.transpose(csd_model.B_reg)), fodf_amp[x,y,z,:])



csd_model2 = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
csd_model2.sh_coeff = thr_sh_coeff
csd_model2.rho = rho

csd_fit2 = csd_model2.fit_qp(data)
print("Checkpoint-2")
new_sh_coeff = csd_fit2.shm_coeff

#convert estimated odfs to Tournier basis
conversion_matrix = convert_to_mrtrix(order_sh)
tournier_new_sh_coeff = np.dot(new_sh_coeff, conversion_matrix.T)

fods_img = nib.Nifti1Image(tournier_new_sh_coeff, affine)
nib.save(fods_img, "tv_reg_rho1.nii.gz")

mse_modified = mean_square_error(tournier_new_sh_coeff, ground_sh, mask)
acc_modified = angular_correlation(tournier_new_sh_coeff, ground_sh, mask)

""" csd_odf = csd_fit.odf(sphere)
fodf_spheres = actor.odf_slicer(csd_odf[:,:,16:17,:], sphere=sphere, scale=1,
                                norm=False, colormap='plasma')

scene.add(fodf_spheres)
window.show(scene) """


print("MSE: " , mse_conventional)
print("Regularized MSE: " , mse_modified)

print("ACC: " , acc_conventional)
print("Regularized ACC: " , acc_modified)



