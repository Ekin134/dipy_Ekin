from dipy.io.image import load_nifti, save_nifti
from dipy.sims.phantom import add_noise
import nibabel as nib
import os
data, affine = load_nifti(r"/home/ekin/Desktop/MSMT/data/DiSCo1_4_shells.nii.gz")
noisy_data = add_noise(data, snr=10, noise_type='rician')
save_nifti(r"/home/ekin/Desktop/MSMT/data/DiSCo1_4_shells_snr10_set5.nii.gz", noisy_data, affine)