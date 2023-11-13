from shm import mean_square_error, angular_correlation
from dipy.io.image import load_nifti
from angular_error import get_angular_error
import numpy as np

#est_sh, affine = load_nifti(r"C:\Users\ekint\OneDrive\Masaüstü\J_invariant_results\snr20_set2\disco_rho1.2_reg_csd_12.nii.gz")
ground_sh, affine = load_nifti(r"/home/ekin/Documents&code/datasets/disco1_ekin/DiSCo1_Strand_ODFs.nii.gz")
ground_sh = ground_sh[:,:,:,0:45]
mask, affine = load_nifti(r"/home/ekin/Downloads/DiSCo1_mask.nii.gz")

def compute(est_sh, mask, ground_sh):
    order_sh = 8
    ang_err = np.zeros([est_sh.shape[0], est_sh.shape[1], est_sh.shape[2]])
    gt_peak = np.zeros([est_sh.shape[0], est_sh.shape[1], est_sh.shape[2]])
    est_peak = np.zeros([est_sh.shape[0], est_sh.shape[1], est_sh.shape[2]])
    for x in range(est_sh.shape[0]):
        for y in range(est_sh.shape[1]):
            for z in range(est_sh.shape[2]):
            
                if mask[x,y,z] == 0:
                    continue
                else:
                    ang_err[x,y,z], gt_peak[x,y,z], est_peak[x,y,z] = get_angular_error(ground_sh[x,y,z,:], est_sh[x,y,z,:], order_sh)
    return ang_err, gt_peak, est_peak


est_sh, affine = load_nifti(r"/home/ekin/Desktop/MSMT/snr30/lmax8/no_mppca/set3/csd_8.nii.gz")
ang_err233, gt_peak, est_peak233 = compute(est_sh, mask, ground_sh)
ang_err233[np.isnan(ang_err233)] = 0
mean_ang_err233 = np.sum(ang_err233)/np.sum(mask)
print("angular error:", mean_ang_err233)
print("end")

est_sh, affine = load_nifti(r"/home/ekin/Desktop/MSMT/snr30/lmax8/no_mppca/set4/csd_8.nii.gz")
ang_err24, gt_peak, est_peak24 = compute(est_sh, mask, ground_sh)
ang_err24[np.isnan(ang_err24)] = 0
mean_ang_err24 = np.sum(ang_err24)/np.sum(mask)
print("angular error:", mean_ang_err24)
print("end")

est_sh, affine = load_nifti(r"/home/ekin/Desktop/MSMT/snr30/lmax8/mppca/set3/csd_8.nii.gz")
ang_err32, gt_peak32, est_peak32 = compute(est_sh, mask, ground_sh)
ang_err32[np.isnan(ang_err32)] = 0
mean_ang_err32 = np.sum(ang_err32)/np.sum(mask)
print("angular error:", mean_ang_err32)
print("end")


est_sh, affine = load_nifti(r"/home/ekin/Desktop/MSMT/snr30/lmax8/mppca/set4/csd_8.nii.gz")
ang_err322, gt_peak322, est_peak322 = compute(est_sh, mask, ground_sh)
ang_err322[np.isnan(ang_err322)] = 0
mean_ang_err322 = np.sum(ang_err322)/np.sum(mask)
print("angular error:", mean_ang_err322)
print("end")

est_sh, affine = load_nifti(r"/home/ekin/Desktop/MSMT/snr10/lmax8/no_mppca/set5/csd.nii.gz")
ang_err42, gt_peak42, est_peak42 = compute(est_sh, mask, ground_sh)
ang_err42[np.isnan(ang_err42)] = 0
mean_ang_err42 = np.sum(ang_err42)/np.sum(mask)
print("angular error:", mean_ang_err42)
print("end")

est_sh, affine = load_nifti(r"/home/ekin/Desktop/MSMT/snr10/lmax12/mppca/set4/reg_csd_rho0.75.nii.gz")
ang_err52, gt_peak52, est_peak52 = compute(est_sh, mask, ground_sh)
ang_err52[np.isnan(ang_err52)] = 0
mean_ang_err52 = np.sum(ang_err52)/np.sum(mask)
print("angular error:", mean_ang_err52)
print("end")

est_sh, affine = load_nifti(r"/home/ekin/Desktop/MSMT/snr10/lmax12/mppca/set4/reg_csd_rho1.nii.gz")
ang_err62, gt_peak62, est_peak62 = compute(est_sh, mask, ground_sh)
ang_err62[np.isnan(ang_err62)] = 0
mean_ang_err62 = np.sum(ang_err62)/np.sum(mask)
print("angular error:", mean_ang_err62)
print("end")

est_sh, affine = load_nifti(r"/home/ekin/Desktop/MSMT/snr10/lmax12/mppca/set4/reg_csd_rho1.2.nii.gz")
ang_err72, gt_peak72, est_peak72 = compute(est_sh, mask, ground_sh)
ang_err72[np.isnan(ang_err72)] = 0
mean_ang_err72 = np.sum(ang_err72)/np.sum(mask)
print("angular error:", mean_ang_err72)
print("end")

est_sh, affine = load_nifti(r"/home/ekin/Desktop/MSMT/snr10/lmax12/mppca/set4/reg_csd_rho1.5.nii.gz")
ang_err82, gt_peak82, est_peak82 = compute(est_sh, mask, ground_sh)
ang_err82[np.isnan(ang_err82)] = 0
mean_ang_err82 = np.sum(ang_err82)/np.sum(mask)
print("angular error:", mean_ang_err82)
print("end")


# SET 3
est_sh, affine = load_nifti(r"/home/ekin/Desktop/MSMT/snr10/lmax12/mppca/set4/reg_csd_rho1.75.nii.gz")
ang_err13, gt_peak, est_peak13 = compute(est_sh, mask, ground_sh)
ang_err13[np.isnan(ang_err13)] = 0
mean_ang_err13 = np.sum(ang_err13)/np.sum(mask)
print("angular error:", mean_ang_err13)
print("end")

est_sh, affine = load_nifti(r"/home/ekin/Desktop/MSMT/snr10/lmax12/mppca/set4/reg_csd_rho2.nii.gz")
ang_err23, gt_peak, est_peak23 = compute(est_sh, mask, ground_sh)
ang_err23[np.isnan(ang_err23)] = 0
mean_ang_err23 = np.sum(ang_err23)/np.sum(mask)
print("angular error:", mean_ang_err23)
print("end")

est_sh, affine = load_nifti(r"/home/ekin/Desktop/MSMT/snr10/lmax12/mppca/set4/reg_csd_rho2.25.nii.gz")
ang_err33, gt_peak, est_peak33 = compute(est_sh, mask, ground_sh)
ang_err33[np.isnan(ang_err33)] = 0
mean_ang_err33 = np.sum(ang_err33)/np.sum(mask)
print("angular error:", mean_ang_err33)
print("end")

est_sh, affine = load_nifti(r"/home/ekin/Desktop/MSMT/snr10/lmax12/mppca/set4/reg_csd_rho2.5.nii.gz")
ang_err43, gt_peak, est_peak43 = compute(est_sh, mask, ground_sh)
ang_err43[np.isnan(ang_err43)] = 0
mean_ang_err43 = np.sum(ang_err43)/np.sum(mask)
print("angular error:", mean_ang_err43)
print("end")
