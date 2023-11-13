"SSST"
#import necessary packages
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.viz import window, actor
from dipy.reconst.csdeconv import (response_from_mask_ssst)
from dipy.sims.voxel import single_tensor_odf
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.data import get_sphere
from skimage.restoration import (denoise_tv_chambolle,
                                 estimate_sigma)
from dipy.reconst.csdeconv import (auto_response_ssst,
                                   mask_for_response_ssst,
                                   response_from_mask_ssst)
from scipy.linalg import pinv, eigh
def csdeconv_PGD_voxelwise_ssst(fconv, S, B_reg, bvalues, where_dwi, tau=0.1, Niters=600):
    max_fconv  = np.max(fconv)
    fconv_norm = fconv/max_fconv

    S = S[where_dwi]
    max_S  = np.max(S)
    S_norm = S/max_S

    lmax_t = 4
    ncoeff = int((lmax_t + 1)*(lmax_t + 2)/2)
    F_SH   = np.linalg.lstsq(fconv_norm[:, :ncoeff], S_norm, rcond=None)[0]
    ncoeff = np.shape(fconv)[1]
    F_SH   = np.concatenate((F_SH, np.zeros((ncoeff-F_SH.shape[0]))))

    mean_fODF = np.mean(np.dot(B_reg, F_SH))
    # tau = 0.1
    threshold = tau*mean_fODF

    #ncoeff     = HR_scheme['sh'].shape[1]-1
    #fconv_norm = np.concatenate((fconv_norm, np.zeros((fconv_norm.shape[0], ncoeff-fconv_norm.shape[1]))), axis=1)

    FtF = np.dot(fconv_norm.T, fconv_norm)
    FtS = np.dot(fconv_norm.T, S_norm)
    HR_scheme_shinv = pinv(B_reg)

    F_SH_xold = np.zeros((F_SH.shape[0], 1))
    F_SH_xold[0] = F_SH[0]

    l1 = eigh(FtF, eigvals_only=True, check_finite=False)[-1]

    # Niters = 600
    F_SH_yold = F_SH_xold

    corr_factor = np.exp(-np.mean(bvalues)*1e-3)
    corr_factor = np.clip(corr_factor, 0.01, 0.5)
    alpha = corr_factor/l1

    ls_old = 1

    for num_it in range(Niters):
        gradient  = np.transpose([FtS]) - np.matmul(FtF, F_SH_xold)
        F_SH_ynew = F_SH_xold + alpha*gradient

        fODF = np.dot(B_reg, F_SH_ynew)
        #ind = fODF < threshold
        ind = np.where(fODF < threshold)
        fODF[ind] = 0
        F_SH_ynew = np.dot(HR_scheme_shinv, fODF)

        ls = (1 + np.sqrt(1 + 4*ls_old**2))/2
        if num_it == Niters-1:
            ls = (1 + np.sqrt(1 + 8*ls_old**2))/2
        F_SH_xnew = F_SH_ynew + (ls_old - 1)*(1/ls)*(F_SH_ynew - F_SH_yold) + (ls_old/ls)*(F_SH_ynew - F_SH_xold)

        F_SH_xold = F_SH_xnew
        F_SH_yold = F_SH_ynew
        ls_old = ls
    

    F_SH = F_SH_xnew*(max_S/max_fconv)
    S_est = np.dot(fconv, F_SH)

    return F_SH, S_est

sphere = get_sphere('symmetric724')

#load data, bvals, bvecs and mask. Create gradient table
data, affine = load_nifti(r"/home/ekin/Downloads/testing-data_DWIS_hardi-scheme_SNR-20.nii.gz")
bvals, bvecs = read_bvals_bvecs(r"/home/ekin/Downloads/hardi-scheme.bval", r"/home/ekin/Downloads/hardi-scheme.bvec")

gtab = gradient_table(bvals, bvecs)
#mask = mask_for_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
#mask_save = np.uint8(mask)
#save_nifti(r"C:\Users\ekint\OneDrive\Masaüstü\hardi_ssst\computed_mask.nii.gz", mask_save, affine)
mask, affine = load_nifti(r"/home/ekin/Downloads/mask_for_response.nii.gz")
# infer response from mask
response, ratio = response_from_mask_ssst(gtab, data, mask)
print(response)
print(ratio)

#convert estimated odfs to Tournier basis

csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)

sh = np.zeros([data.shape[0], data.shape[1], data.shape[2], 45,1])

for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        for z in range(data.shape[2]):
            sh[x, y, z, :], _  = csdeconv_PGD_voxelwise_ssst(csd_model._X, data[x,y,z,:], csd_model.B_reg, np.unique(bvals), csd_model._where_dwi)
            
from basis_conversion import convert_to_mrtrix
order_sh = 8
conversion_matrix = convert_to_mrtrix(order_sh)
tournier_sh_coeff = np.dot(sh, conversion_matrix.T)

fods_img = nib.Nifti1Image(tournier_sh_coeff, affine)
nib.save(fods_img,  r"/home/ekin/Downloads/hardi_PGD_v2.nii.gz")

print("finito")

