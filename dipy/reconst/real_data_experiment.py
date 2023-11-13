#import necessary packages
import numpy as np
import nibabel as nib
from dipy.denoise.localpca import mppca
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.viz import window, actor
from dipy.reconst.csdeconv import (response_from_mask_ssst)
from dipy.sims.voxel import single_tensor_odf
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.data import get_sphere, get_fnames
from skimage.restoration import (calibrate_denoiser,
                                 denoise_tv_chambolle,                                                             
                                 estimate_sigma)
from skimage.restoration.j_invariant import _invariant_denoise
from dipy.reconst.csdeconv import (auto_response_ssst,
                                   mask_for_response_ssst,
                                   response_from_mask_ssst)
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

def denoise_tv_chambolle_K(data_3d, K):
    sigma_est = np.mean(estimate_sigma(data_3d, channel_axis=None))
    scaled_sigma = K*sigma_est
    func = denoise_tv_chambolle(data_3d, weight=scaled_sigma, eps=0.0002, max_num_iter=200, channel_axis=None)
    return func

def denoise_tv_chambolle_K_4d(data_4d, K):
    denoised_sh = np.zeros(np.shape(data_4d))
    print("NOW K is ", K)
    for voxelt in range(data_4d.shape[3]):
        data_vol  = np.squeeze(data_4d[:,:,:,voxelt])
        brain_mask, affine = load_nifti(r"/home/ekin/Desktop/penthera/brain_mask.nii.gz")
        brain_mask = brain_mask[:,:,:,0,0]
        masked_data_vol = data_vol * brain_mask
        nonzero_masked = masked_data_vol[masked_data_vol != 0]
        sigma_est = np.mean(estimate_sigma(nonzero_masked, channel_axis=None))
        print(sigma_est)
        scaled_sigma = K*sigma_est
        denoised_sh[:,:,:,voxelt] = denoise_tv_chambolle(data_vol, weight=scaled_sigma, eps=0.0002, max_num_iter=200, channel_axis=None)
    return denoised_sh

sphere = get_sphere('symmetric724')

#fraw, fbval, fbvec = get_fnames("sherbrooke_3shell")
#data_noisy, affine = load_nifti(fraw)
#bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
data_noisy, affine = load_nifti(r"/home/ekin/Desktop/penthera/sub-PT012.ses-2_3.nii.gz")
bvals, bvecs = read_bvals_bvecs(r"/home/ekin/Desktop/penthera/sub-PT012.ses-2_3.bvals", r"/home/ekin/Desktop/penthera/sub-PT012.ses-2_3.bvecs")
#data_noisy, affine = load_nifti(r"/home/ekin/Downloads/DWIS_hardi-scheme_snr-20-set5.nii.gz")
#bvals, bvecs = read_bvals_bvecs(r"/home/ekin/Downloads/hardi-scheme.bval", r"/home/ekin/Downloads/hardi-scheme.bvec")
print("bvals are ", np.unique(bvals))
#brain mask
#from dipy.segment.mask import median_otsu
#b_sel = np.where(bvals==0)
#brain_mask_data = data_noisy[..., b_sel]
#print(np.shape(brain_mask_data))
#b0_mask, mask = median_otsu(brain_mask_data, median_radius=2, numpass=1)
#maskkk = np.uint8(mask)
#save_nifti(r"/home/ekin/Desktop/hardi_doublecheck/hardi_test/brain_mask2.nii.gz", maskkk, affine)

#bvecs[:,[1,2]] = bvecs[:,[2,1]]
#bvecs[:,1] *= -1
#bvecs[:,2] *= -1
bvecs[:,0] *= -1 #what I use

gtab = gradient_table(bvals, bvecs)

bvals = gtab.bvals
bvecs = gtab.bvecs
sel_b = np.logical_or(bvals == 0, bvals == 2000)
data_noisy = data_noisy[..., sel_b]
gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

# MPPCA OR NO MPPCA
#data = mppca(data_noisy, patch_radius=2)
data = data_noisy

mask = mask_for_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
print("voxel number in mask is ",np.sum(mask))
#mask_save = np.uint8(mask)
#save_nifti(r"/home/ekin/Desktop/penthera/mppca_mask_for_response.nii.gz", mask_save, affine)

# infer response from mask
response, ratio = response_from_mask_ssst(gtab, data, mask)
print(response)
print(ratio)
#scene = window.Scene()
#evals = response[0]
#evecs = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T

#response_odf = single_tensor_odf(sphere.vertices, evals, evecs)
# transform our data from 1D to 4D
#response_odf = response_odf[None, None, None, :]
#response_actor = actor.odf_slicer(response_odf, sphere=sphere,
#                                  colormap='plasma')
#scene.add(response_actor)
#window.show(scene)
#scene.rm(response_actor)

#fODF reconstruction
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=12)
csd_fit = csd_model.fit(data)
print("Checkpoint-1")
sh_coeff = csd_fit.shm_coeff

fods_img = nib.Nifti1Image(sh_coeff, affine)
nib.save(fods_img, r"/home/ekin/Desktop/penthera/nomppca_csd_12_descoteaux.nii.gz")

#convert estimated odfs to Tournier basis
from basis_conversion import convert_to_mrtrix
order_sh = 12
conversion_matrix = convert_to_mrtrix(order_sh)
#tournier_sh_coeff = np.dot(sh_coeff, conversion_matrix.T)

#fods_img = nib.Nifti1Image(tournier_sh_coeff, affine)
#nib.save(fods_img,  r"/home/ekin/Desktop/penthera/corrected/mppca_csd_8.nii.gz")
#sh_coeff, affine = load_nifti(r"/home/ekin/Desktop/hardi_doublecheck/hardi_test/csd_12_descoteaux.nii.gz")

#K_range = {'K': np.arange(0.1, 10, 0.1)}
#K_range = {'K': [1]}

#_, (parameters_tested_tv, losses_tv) = calibrate_denoiser(
#                                        sh_coeff,
#                                        denoise_tv_chambolle_K_4d,
#                                        denoise_parameters=K_range,
#                                        extra_output=True)
#best_parameters_tv = parameters_tested_tv[np.argmin(losses_tv)]
#print(best_parameters_tv)
#sh_coeff_new = denoise_tv_chambolle_K_4d(sh_coeff, best_parameters_tv['K'])
sh_coeff_new = denoise_tv_chambolle_K_4d(sh_coeff, 0.5)
sh_coeff = sh_coeff_new

# PROJECT ONTO THE CONSTRAINT SET
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


#fods_img = nib.Nifti1Image(thr_sh_coeff, affine)
#nib.save(fods_img,  r"/home/ekin/Desktop/hardi_doublecheck/snr20-set5/redo_optimal_K/prior_descoteaux.nii.gz")
tournier_thr_sh_coeff = np.dot(thr_sh_coeff, conversion_matrix.T)
fods_img2 = nib.Nifti1Image(tournier_thr_sh_coeff, affine)
nib.save(fods_img2,  r"/home/ekin/Desktop/penthera/K0.5_nomppca_prior_12.nii.gz")

rho = 2

csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=12)
csd_model.sh_coeff = thr_sh_coeff
csd_model.rho = rho

csd_fit = csd_model.fit_qp(data)
new_sh_coeff = csd_fit.shm_coeff

tournier_new_sh_coeff = np.dot(new_sh_coeff, conversion_matrix.T)

fods_img = nib.Nifti1Image(tournier_new_sh_coeff, affine)
nib.save(fods_img, r"/home/ekin/Desktop/penthera/K0.5_nomppca_sr2csd.nii.gz") 
print("eof")


