from shm import mean_square_error, angular_correlation
from dipy.io.image import load_nifti


est_sh, affine = load_nifti(r"/home/ekin/Desktop/MSMT/snr10/lmax12/mppca/set4/reg_csd_rho2.5.nii.gz")
ground_sh, affine = load_nifti(r"/home/ekin/Documents&code/datasets/disco1_ekin/DiSCo1_Strand_ODFs.nii.gz")
#ground_sh = ground_sh[:,:,:,0:45]
mask, affine = load_nifti(r"/home/ekin/Downloads/DiSCo1_mask.nii.gz")
#ground_sh = ground_sh[:,:,:,0:est_sh.shape[3]]

mse = mean_square_error(est_sh, ground_sh, mask)
acc = angular_correlation(est_sh, ground_sh, mask)

print("MSE:", mse)
print("ACC:", acc)
print("the end")