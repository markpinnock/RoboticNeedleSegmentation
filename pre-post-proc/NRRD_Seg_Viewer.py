import matplotlib.pyplot as plt
import numpy as np
import os
import nrrd


FILE_PATH = "Z:/Robot_Seg_Raw_Data/"
subject_id = "UCLH_08470270/"
# img_path = FILE_PATH + "SparseVolImgs/" + subject_id
# mask_path = FILE_PATH + "SparseVolMasks/" + subject_id
img_path = FILE_PATH + "TestVolImages/" + subject_id
mask_path = FILE_PATH + "TestVolMasks/" + subject_id

# img_list = os.listdir(img_path)
mask_list = os.listdir(mask_path)
# img_list.sort()
mask_list.sort()

for mask in mask_list:
    img = mask[:-10] + "L.nrrd"
    mask_vol, _ = nrrd.read(mask_path + mask)
    mask_vol = mask_vol[0, ...]

    try:
        img_vol, _ = nrrd.read(img_path + img)
    except:
        print(f"NO CORRESPONDING MASK FOUND: {mask} {img}")
    else:
        for i in range(mask_vol.shape[2]):
            print(mask)
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(img_vol[:, :, i].T, cmap='gray', origin='lower')
            axs[1].imshow(mask_vol[:, :, i].T, cmap='gray', origin='lower')
            axs[2].imshow(img_vol[:, :, i].T * mask_vol[:, :, i].T, cmap='gray', origin='lower')
            plt.show()
