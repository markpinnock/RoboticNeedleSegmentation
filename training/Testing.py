import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append('..')

from Networks import UNetGen
from utils.DataLoader import imgLoader
from utils.TrainFuncs import diceLoss


""" Tests U-Net on test dataset """
# Set up hyperparameters to generate model name, set input size
MB_SIZE = 4
NC = 4
EPOCHS = 500
ETA = 0.001
LO_VOL_SIZE = (512, 512, 3, 1, )

# Set up file paths and dataset
FILE_PATH = "C:/Users/roybo/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/001_CNN_RNS/"
DATA_PATH = "Z:/Robot_Data/Test/"
EXPT_NAME = f"nc{NC}_ep{EPOCHS}_eta{ETA}"
MODEL_SAVE_PATH = f"{FILE_PATH}models/{EXPT_NAME}/"
img_path = f"{DATA_PATH}Img/"
seg_path = f"{DATA_PATH}Seg/"
imgs = os.listdir(img_path)
segs = os.listdir(seg_path)
imgs.sort()
segs.sort()

N = len(imgs)
assert N == len(segs), "HI/LO IMG PAIRS UNEVEN LENGTHS"

# Create test dataset
test_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[img_path, seg_path, imgs, segs, False], output_types=(tf.float32, tf.float32))

# Initialise U-Net and load weights
UNet = UNetGen(input_shape=LO_VOL_SIZE, starting_channels=NC)
UNet.load_weights(f"{MODEL_SAVE_PATH}{EXPT_NAME}.ckpt")

# Initialise test result array and rgb array for seg difference
test_metric = 0
rgb_pred = np.zeros((MB_SIZE, 512, 512, 3, 3), dtype=np.float32)

# Iterate through imgs in dataset, display results
for img, seg in test_ds.batch(MB_SIZE):
    fig, axs = plt.subplots(3, MB_SIZE)
    pred = UNet(img)
    temp_metric = diceLoss(pred, seg)
    test_metric += temp_metric
    print(f"Batch Dice score: {1 - temp_metric / MB_SIZE}")

    rgb_pred[:, :, :, :, 0] = seg[:, :, :, :, 0]
    rgb_pred[:, :, :, :, 1] = pred[:, :, :, :, 0].numpy()
    rgb_pred[:, :, :, :, 2] = pred[:, :, :, :, 0].numpy()

    for j in range(img.shape[0]):
        axs[0, j].imshow(np.fliplr(img[j, :, :, 1, 0].numpy().T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
        axs[0, j].axis('off')
        axs[1, j].imshow(np.fliplr(img[j, :, :, 1, 0].numpy().T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
        axs[1, j].axis('off')
        axs[1, j].imshow(np.fliplr(np.ma.masked_where(pred[j, :, :, 1, 0].numpy().T == False, pred[j, :, :, 1, 0].numpy().T)), cmap='Set1', origin='lower')
        axs[1, j].axis('off')
        r_pred = np.fliplr(rgb_pred[j, :, :, 1, 0].T)
        g_pred = np.fliplr(rgb_pred[j, :, :, 1, 1].T)
        b_pred = np.fliplr(rgb_pred[j, :, :, 1, 2].T)
        axs[2, j].imshow(np.concatenate([r_pred[:, :, np.newaxis], g_pred[:, :, np.newaxis], b_pred[:, :, np.newaxis]], axis=2), origin='lower')
        axs[2, j].axis('off')

    plt.show()

# Print final dice score
print(f"Final Dice score: {test_metric / N}")