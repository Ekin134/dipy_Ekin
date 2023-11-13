import numpy as np
import dipy.reconst.shm as shm
import dipy.direction.peaks as dp
import dipy.reconst.dti as dti
import matplotlib.pyplot as plt

from dipy.denoise.localpca import mppca
from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.reconst.mcsd import (auto_response_msmt,
                               mask_for_response_msmt,
                               response_from_mask_msmt)
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response
from dipy.viz import window, actor
from skimage.restoration import (calibrate_denoiser,
                                 denoise_tv_chambolle,
                                 denoise_bilateral,
                                 denoise_wavelet,
                                 estimate_sigma,
                                 denoise_nl_means)
from dipy.direction.peaks import peaks_from_model

from dipy.data import get_sphere, get_fnames
import nibabel as nib

def denoise_tv_chambolle_K(data_3d, K):
    sigma_est = np.mean(estimate_sigma(data_3d, channel_axis=None))
    scaled_sigma = K*sigma_est
    func = denoise_tv_chambolle(data_3d, weight=scaled_sigma, eps=0.0002, max_num_iter=200, channel_axis=None)
    return func

def denoise_tv_chambolle_K_4d(data_4d, K):
    denoised_sh = np.zeros(np.shape(data_4d))
    for voxelt in range(data_4d.shape[3]):
        data_vol  = np.squeeze(data_4d[:,:,:,voxelt])
        sigma_est = np.mean(estimate_sigma(data_vol, channel_axis=None))
        scaled_sigma = K*sigma_est
        print("K is ", K)
        print("sigma is ", sigma_est)
        denoised_sh[:,:,:,voxelt] = denoise_tv_chambolle(data_vol, weight=scaled_sigma, eps=0.0002, max_num_iter=200, channel_axis=None)
    return denoised_sh

sphere = get_sphere('symmetric724')


#data_noisy, affine = load_nifti(r"/home/ekin/Desktop/MSMT/data/DiSCo1_4_shells_snr10_set5.nii.gz")
bvals, bvecs = read_bvals_bvecs(r"/home/ekin/Desktop/MSMT/data/DiSCo_4_shells.bvals", r"/home/ekin/Desktop/MSMT/data/DiSCo_4_shells.bvecs")
gtab = gradient_table(bvals, bvecs)

bvals = gtab.bvals
bvecs = gtab.bvecs


#data = data_noisy # no denoising
#wm_mask, affine = load_nifti(r"/home/ekin/Downloads/DiSCo1_mask.nii.gz")
#data = mppca(data_noisy, mask=wm_mask, patch_radius=2)
#save_nifti(r"/home/ekin/Desktop/MSMT/data/mppca_denoised_DiSCo1_4_shells_snr10_set5.nii.gz", data, affine)
data, affine = load_nifti(r"/home/ekin/Desktop/MSMT/data/DiSCo1_4_shells_snr20_set4.nii.gz")


mask_wm, mask_gm, mask_csf = mask_for_response_msmt(gtab, data, roi_radii=10,
                                                    wm_fa_thr=0.7,
                                                    gm_fa_thr=0.3,
                                                    csf_fa_thr=0.15,
                                                    gm_md_thr=0.001,
                                                    csf_md_thr=0.0032)



"""
The masks can also be used to calculate the number of voxels for each tissue.
"""

nvoxels_wm = np.sum(mask_wm)
nvoxels_gm = np.sum(mask_gm)
nvoxels_csf = np.sum(mask_csf)

print(nvoxels_wm)
print(nvoxels_gm)
print(nvoxels_csf)

"""
Then, the ``response_from_mask`` function will return the msmt response
functions using precalculated tissue masks.
"""

response_wm, response_gm, response_csf = response_from_mask_msmt(gtab, data,
                                                                 mask_wm,
                                                                 mask_gm,
                                                                 mask_csf)


ubvals = unique_bvals_tolerance(gtab.bvals)
response_mcsd = multi_shell_fiber_response(sh_order=8,
                                           bvals=ubvals,
                                           wm_rf=response_wm,
                                           gm_rf=response_gm,
                                           csf_rf=response_csf)


mcsd_model = MultiShellDeconvModel(gtab, response_mcsd, sh_order=8)

mcsd_fit = mcsd_model.fit(data)

sh_coeff = mcsd_fit.all_shm_coeff
csf_sh_coeff = sh_coeff[..., 0]
gm_sh_coeff = sh_coeff[..., 1]
wm_sh_coeff = mcsd_fit.shm_coeff

#fods_img = nib.Nifti1Image(csf_sh_coeff, affine)
#nib.save(fods_img,  r"/home/ekin/Desktop/MSMT/snr10/lmax12/mppca/set4/csf_sh_descoteaux.nii.gz")
#fods_img3 = nib.Nifti1Image(gm_sh_coeff, affine)
#nib.save(fods_img3,  r"/home/ekin/Desktop/MSMT/snr10/lmax12/mppca/set4/gm_sh_descoteaux.nii.gz")

from basis_conversion import convert_to_mrtrix
order_sh = 8
coeff = (order_sh+1)*(order_sh+2)/2
conversion_matrix = convert_to_mrtrix(order_sh)
tournier_sh_coeff = np.dot(wm_sh_coeff, conversion_matrix.T)
#fods_img = nib.Nifti1Image(tournier_sh_coeff, affine)
#nib.save(fods_img,  r"/home/ekin/Desktop/MSMT/snr10/lmax8/mppca/set4/csd.nii.gz")

iteration_number = 1
for i in range(iteration_number):
    
    K_range = {'K': np.arange(0.1, 3, 0.1)}
 
    _, (parameters_tested_tv, losses_tv) = calibrate_denoiser(
                                            wm_sh_coeff,
                                            denoise_tv_chambolle_K_4d,
                                            denoise_parameters=K_range,
                                            extra_output=True)
    best_parameters_tv = parameters_tested_tv[np.argmin(losses_tv)]
    print(best_parameters_tv)
    wm_sh_coeff_new = denoise_tv_chambolle_K_4d(wm_sh_coeff, best_parameters_tv['K'])
    wm_sh_coeff = wm_sh_coeff_new

    fodf_amp = np.zeros([wm_sh_coeff.shape[0],wm_sh_coeff.shape[1],wm_sh_coeff.shape[2],mcsd_model.B_reg.shape[0]-2])
    for x in range(wm_sh_coeff.shape[0]):
        for y in range(wm_sh_coeff.shape[1]):
            for z in range(wm_sh_coeff.shape[2]):
                fodf_amp[x,y,z,:] = np.dot(mcsd_model.B_reg[2:,2:], wm_sh_coeff[x,y,z,:])
               
    fodf_amp[fodf_amp < 0] = 0
    thr_sh_coeff = np.zeros(wm_sh_coeff.shape)
    for x in range(wm_sh_coeff.shape[0]):
        for y in range(wm_sh_coeff.shape[1]):
            for z in range(wm_sh_coeff.shape[2]):
                thr_sh_coeff[x,y,z,:] = np.dot((np.linalg.inv(np.transpose(mcsd_model.B_reg[2:,2:]) @ mcsd_model.B_reg[2:,2:]) @ np.transpose(mcsd_model.B_reg[2:,2:])), fodf_amp[x,y,z,:])
    
    
    fods_img = nib.Nifti1Image(thr_sh_coeff, affine)
    nib.save(fods_img,  r"/home/ekin/Desktop/MSMT/snr10/lmax12/mppca/set4/prior_descoteaux.nii.gz")
    tournier_thr_sh_coeff = np.dot(thr_sh_coeff, conversion_matrix.T)
    fods_img2 = nib.Nifti1Image(tournier_thr_sh_coeff, affine)
    nib.save(fods_img2,  r"/home/ekin/Desktop/MSMT/snr10/lmax12/mppca/set4/prior.nii.gz")
    
    thr_sh_coeff, affine = load_nifti(r"/home/ekin/Desktop/MSMT/lmax12/no_mppca/prior_descoteaux.nii.gz") #wm
    gm_sh_coeff, affine = load_nifti(r"/home/ekin/Desktop/MSMT/lmax12/no_mppca/gm_sh_descoteaux.nii.gz")
    csf_sh_coeff, affine = load_nifti(r"/home/ekin/Desktop/MSMT/lmax12/no_mppca/csf_sh_descoteaux.nii.gz")
    rho = 1.2
    mcsd_model = MultiShellDeconvModel(gtab, response_mcsd, sh_order=12)
    mcsd_model.wm_sh_coeff = thr_sh_coeff
    mcsd_model.gm_sh_coeff = gm_sh_coeff
    mcsd_model.csf_sh_coeff = csf_sh_coeff
    mcsd_model.rho = rho

    mcsd_fit = mcsd_model.fit_qp(data)
    sh_coeff = mcsd_fit.all_shm_coeff
    csf_sh_coeff = sh_coeff[..., 0]
    gm_sh_coeff = sh_coeff[..., 1]
    reg_wm_sh_coeff = mcsd_fit.shm_coeff

    tournier_thr_sh_coeff = np.dot(reg_wm_sh_coeff, conversion_matrix.T)
    fods_img2 = nib.Nifti1Image(tournier_thr_sh_coeff, affine)
    nib.save(fods_img2,  r"/home/ekin/Desktop/MSMT/lmax12/no_mppca/reg_csd_rho1.2.nii.gz")
    print("done")


    




