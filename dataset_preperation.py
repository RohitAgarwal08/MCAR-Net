import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
# from tifffile import imsave

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


t2_list = sorted(glob.glob('/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
t1ce_list = sorted(glob.glob('/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
flair_list = sorted(glob.glob('/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
t1_list = sorted(glob.glob('/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1.nii'))

mask_list = sorted(glob.glob('/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*seg.nii'))

import os
os.mkdir("Train")
os.mkdir("Train/Images")
os.mkdir("Train/Mask")

for img in range(0, len(t2_list)):
    if(img):
        print("Now preparing image and masks number: ", img)
    
        temp_image_t2=nib.load(t2_list[img]).get_fdata()
        temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
    
        temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
        temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
    
        temp_image_flair=nib.load(flair_list[img]).get_fdata()
        temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
    
        temp_image_t1=nib.load(t1_list[img]).get_fdata()
        temp_image_t1=scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(temp_image_t1.shape)
    
        temp_mask=nib.load(mask_list[img]).get_fdata()
        temp_mask=temp_mask.astype(np.uint8)
        temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
           
    
        temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2, temp_image_t1], axis=3)
    
        #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
        #cropping x, y, and z
        temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
        temp_mask = temp_mask[56:184, 56:184, 13:141]
         
        #val, counts = np.unique(temp_mask, return_counts=True)
    
        #if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
        # print("Save Me")
        # print(np.unique(temp_mask))
        temp_mask= to_categorical(temp_mask, num_classes=4)
        # print(np.unique(temp_mask))
        output_path = '/kaggle/working/Train/Images/image_'+str(img)+'.nii'
        output_path1 = '/kaggle/working/Train/Mask/mask_'+str(img)+'.nii'
        
        nifti_img = nib.Nifti1Image(temp_combined_images, affine=np.eye(4))
        nib.save(nifti_img, output_path)

        nifti_img = nib.Nifti1Image(temp_mask, affine=np.eye(4))
        nib.save(nifti_img, output_path1)


        


