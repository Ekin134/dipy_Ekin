"SSST"
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
from dipy.data import get_sphere
from dipy.sims.phantom import add_noise
from skimage.restoration import (calibrate_denoiser,
                                 denoise_tv_chambolle,
                                 denoise_bilateral,
                                 denoise_wavelet,
                                 estimate_sigma,
                                 denoise_nl_means)
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
    for voxelt in range(data_4d.shape[3]):
        data_vol  = np.squeeze(data_4d[:,:,:,voxelt])
        mask, affine = load_nifti(r"/home/ekin/Desktop/hardi_doublecheck/hardi_test/brain_mask.nii.gz")
        maskk = mask[:,:,:,0,0]
        masked_data_vol = maskk * data_vol
        nonzero_masked = masked_data_vol[masked_data_vol != 0]
        sigma_est = np.mean(estimate_sigma(nonzero_masked, channel_axis=None))
        scaled_sigma = K*sigma_est
        denoised_sh[:,:,:,voxelt] = denoise_tv_chambolle(data_vol, weight=scaled_sigma, eps=0.0002, max_num_iter=200, channel_axis=None)
    return denoised_sh

sphere = get_sphere('symmetric724')

#load data, bvals, bvecs and mask. Create gradient table
data_noisy, affine = load_nifti(r"/home/ekin/Downloads/DWIS_hardi-scheme_snr-20-set5.nii.gz")
bvals, bvecs = read_bvals_bvecs(r"/home/ekin/Downloads/hardi-scheme.bval", r"/home/ekin/Downloads/hardi-scheme.bvec")
#data_noisy, affine = load_nifti(r"/home/ekin/Desktop/MSMT/data/DiSCo1_4_shells.nii.gz")
#bvals, bvecs = read_bvals_bvecs(r"/home/ekin/Desktop/MSMT/data/DiSCo_4_shells.bvals", r"/home/ekin/Desktop/MSMT/data/DiSCo_4_shells.bvecs")

#wm_mask, affine = load_nifti(r"/home/ekin/Downloads/DiSCo1_mask.nii.gz")
#data = mppca(data_noisy, mask=wm_mask, patch_radius=2)
#mppca_mask, affine = load_nifti(r"/home/ekin/Downloads/mask_compute_local_metrics.nii.gz")
#data = mppca(data_noisy, mask=mppca_mask, patch_radius=2)
data = data_noisy

gtab = gradient_table(bvals, bvecs)
#mask = mask_for_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
#mask_save = np.uint8(mask)
#save_nifti(r"C:\Users\ekint\OneDrive\Masaüstü\hardi_ssst\computed_mask.nii.gz", mask_save, affine)
mask, affine = load_nifti(r"/home/ekin/Downloads/mask_for_response.nii.gz")

# infer response from mask
response, ratio = response_from_mask_ssst(gtab, data, mask)
print(response)
print(ratio)
scene = window.Scene()
evals = response[0]
evecs = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T

response_odf = single_tensor_odf(sphere.vertices, evals, evecs)
# transform our data from 1D to 4D
response_odf = response_odf[None, None, None, :]
response_actor = actor.odf_slicer(response_odf, sphere=sphere,
                                  colormap='plasma')
scene.add(response_actor)
window.show(scene)
scene.rm(response_actor)

rho = 2
#fODF reconstruction
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=12)
csd_fit = csd_model.fit(data)
print("Checkpoint-1")
sh_coeff = csd_fit.shm_coeff

fods_img = nib.Nifti1Image(sh_coeff, affine)
nib.save(fods_img, r"/home/ekin/Desktop/hardi_doublecheck/hardi_test/csd_12_descoteaux.nii.gz")

#convert estimated odfs to Tournier basis
from basis_conversion import convert_to_mrtrix
order_sh = 12
conversion_matrix = convert_to_mrtrix(order_sh)
tournier_sh_coeff = np.dot(sh_coeff, conversion_matrix.T)

fods_img = nib.Nifti1Image(tournier_sh_coeff, affine)
nib.save(fods_img,  r"/home/ekin/Desktop/hardi_doublecheck/hardi_test/csd_12.nii.gz")
#nib.save(fods_img,  r"C:\Users\ekint\OneDrive\Masaüstü\mppca_J_invariant_results\snr30_set5\disco_csd_12.nii.gz")
#sh_coeff, affine = load_nifti(r"C:\Users\ekint\OneDrive\Masaüstü\J_invariant_results\snr20_set1\hardi_csd_12_descoteaux.nii.gz")
#sigma_est = np.zeros((sh_coeff.shape[3]))
iteration_number = 1
for i in range(iteration_number):
    #sh_coeff = average(sh_coeff)
    
    # FOURTH METHOD 
    #K_range = {'K': [5]}
    
    #K_range = {'K': [2.6]}
   
    K_range = {'K': np.arange(0.1, 10, 0.1)}
 
    _, (parameters_tested_tv, losses_tv) = calibrate_denoiser(
                                            sh_coeff,
                                            denoise_tv_chambolle_K_4d,
                                            denoise_parameters=K_range,
                                            extra_output=True)
    best_parameters_tv = parameters_tested_tv[np.argmin(losses_tv)]
    print(best_parameters_tv)
    sh_coeff_new = denoise_tv_chambolle_K_4d(sh_coeff, best_parameters_tv['K'])
    sh_coeff = sh_coeff_new
    
    """
    for voxelt in range(sh_coeff.shape[3]):
        
        data_vol  = np.squeeze(sh_coeff[:,:,:,voxelt])
        
        # FIRST METHOD
        sigma_est[voxelt] = np.mean(estimate_sigma(data_vol, channel_axis=None))
        #sigma_est = 0.025935670194014392
        print("Sigma Estimated: ", sigma_est)
        sh_coeff[:,:,:,voxelt] = denoise_tv_chambolle(data_vol, weight=1.0*sigma_est[voxelt], eps=0.0002, max_num_iter=200, channel_axis=None)
        """

    """
        # SECOND METHOD
        parameter_ranges_tv = {'weight': np.arange(0.001, 0.2, 0.001)}
        _, (parameters_tested_tv, losses_tv) = calibrate_denoiser(
                                            data_vol,
                                            denoise_tv_chambolle,
                                            denoise_parameters=parameter_ranges_tv,
                                            extra_output=True)
        best_parameters_tv = parameters_tested_tv[np.argmin(losses_tv)]
        print(best_parameters_tv)
        sh_coeff[:,:,:,voxelt] = _invariant_denoise(data_vol, denoise_tv_chambolle,
                                            denoiser_kwargs=best_parameters_tv)
        """
    """
        # THIRD METHOD
        K_range = {'K': np.arange(0.1, 2, 0.1)}
        _, (parameters_tested_tv, losses_tv) = calibrate_denoiser(
                                                data_vol,
                                                denoise_tv_chambolle_K,
                                                denoise_parameters=K_range,
                                                extra_output=True)
        best_parameters_tv = parameters_tested_tv[np.argmin(losses_tv)]
        print(best_parameters_tv)
        sh_coeff[:,:,:,voxelt] = denoise_tv_chambolle_K(data_vol, best_parameters_tv['K'])
        """
        


    
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
    
    
    fods_img = nib.Nifti1Image(thr_sh_coeff, affine)
    nib.save(fods_img,  r"/home/ekin/Desktop/hardi_doublecheck/hardi_test/prior_descoteaux.nii.gz")
    #nib.save(fods_img,  r"C:\Users\ekint\OneDrive\Masaüstü\mppca_J_invariant_results\hardi\snr10_set2\hardi_prior_descoteaux.nii.gz")
    tournier_thr_sh_coeff = np.dot(thr_sh_coeff, conversion_matrix.T)
    fods_img2 = nib.Nifti1Image(tournier_thr_sh_coeff, affine)
    nib.save(fods_img2,  r"/home/ekin/Desktop/hardi_doublecheck/hardi_test/prior.nii.gz")
    #nib.save(fods_img2,  r"C:\Users\ekint\OneDrive\Masaüstü\mppca_J_invariant_results\hardi\snr10_set2\hardi_prior.nii.gz")
    
    #thr_sh_coeff, affine = load_nifti(r"/home/ekin/Desktop/no_mppca/hardi_snr10_set2/hardi_prior_descoteaux.nii.gz")

    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=12)
    csd_model.sh_coeff = thr_sh_coeff
    csd_model.rho = rho

    csd_fit = csd_model.fit_qp(data)
    new_sh_coeff = csd_fit.shm_coeff

    #convert estimated odfs to Tournier basis
    conversion_matrix = convert_to_mrtrix(order_sh)
    tournier_new_sh_coeff = np.dot(new_sh_coeff, conversion_matrix.T)

    fods_img = nib.Nifti1Image(tournier_new_sh_coeff, affine)
    nib.save(fods_img, r"/home/ekin/Desktop/hardi_doublecheck/hardi_test/sr2csd.nii.gz") 
    sh_coeff = new_sh_coeff

