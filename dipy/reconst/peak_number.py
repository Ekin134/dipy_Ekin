from dipy.io.image import load_nifti
import numpy as np
from dipy.reconst import shm
import dipy.direction.peaks
import dipy.data
from dipy.data import get_sphere
from dipy.core.sphere import HemiSphere

ground_sh, affine = load_nifti(r"/home/ekin/Documents&code/datasets/disco1_ekin/DiSCo1_Strand_ODFs.nii.gz")
mask, affine = load_nifti(r"/home/ekin/Documents&code/datasets/disco1_ekin/DiSCo1_mask.nii.gz")
ground_sh = ground_sh[:,:,:,0:45]

# List of spheres 

sphere = dipy.data.get_sphere("repulsion724")
sphere_724 = get_sphere("repulsion724") 
sphere_2890 = sphere_724.subdivide(1) 
sphere_error= HemiSphere.from_sphere(sphere_2890)

def number_peak(est_sh, mask):
    peak_nb = np.zeros([est_sh.shape[0], est_sh.shape[1], est_sh.shape[2]])
    for x in range(est_sh.shape[0]):
        for y in range(est_sh.shape[1]):
            for z in range(est_sh.shape[2]):
                if mask[x,y,z] == 0:
                    continue
                else:
                    sf_prediction = shm.sh_to_sf(est_sh[x,y,z,:], sphere=sphere_error, sh_order=8, basis_type='tournier07')
                    _, _, peaks_prediction_ind = dipy.direction.peaks.peak_directions(sf_prediction, sphere_error, relative_peak_threshold=.3, min_separation_angle=20)
                    peak_nb[x,y,z] = np.size(peaks_prediction_ind)
    return peak_nb

gt_peak = number_peak(ground_sh, mask)

est_sh, affine = load_nifti(r"/home/ekin/Downloads/csd.nii.gz")
est_peak = number_peak(est_sh, mask)
print("Checkpoint-0")

est_sh, affine = load_nifti(r"C:\Users\ekint\OneDrive\Masaüstü\SSST-Disco Tests\TV_Filter_Out_sh\rho0.01_reg_csd.nii.gz")
est_peak001 = number_peak(est_sh, mask)
print("Checkpoint-1")

est_sh, affine = load_nifti(r"C:\Users\ekint\OneDrive\Masaüstü\SSST-Disco Tests\TV_Filter_Out_sh\rho0.1_reg_csd.nii.gz")
est_peak01 = number_peak(est_sh, mask)
print("Checkpoint-2")

est_sh, affine = load_nifti(r"C:\Users\ekint\OneDrive\Masaüstü\SSST-Disco Tests\TV_Filter_Out_sh\rho0.25_reg_csd.nii.gz")
est_peak025 = number_peak(est_sh, mask)
print("Checkpoint-3")

est_sh, affine = load_nifti(r"C:\Users\ekint\OneDrive\Masaüstü\SSST-Disco Tests\TV_Filter_Out_sh\rho0.5_reg_csd.nii.gz")
est_peak05 = number_peak(est_sh, mask)
print("Checkpoint-4")

est_sh, affine = load_nifti(r"C:\Users\ekint\OneDrive\Masaüstü\SSST-Disco Tests\TV_Filter_Out_sh\rho0.75_reg_csd.nii.gz")
est_peak075 = number_peak(est_sh, mask)
print("Checkpoint-5")

est_sh, affine = load_nifti(r"C:\Users\ekint\OneDrive\Masaüstü\SSST-Disco Tests\TV_Filter_Out_sh\rho1_reg_csd.nii.gz")
est_peak1 = number_peak(est_sh, mask)
print("Checkpoint-6")

est_sh, affine = load_nifti(r"C:\Users\ekint\OneDrive\Masaüstü\SSST-Disco Tests\TV_Filter_Out_sh\rho1.2_reg_csd.nii.gz")
est_peak12 = number_peak(est_sh, mask)
print("Checkpoint-7") 
