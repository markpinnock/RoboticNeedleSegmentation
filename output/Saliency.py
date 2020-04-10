import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras

from NetworkLayers import UNetGen

sys.path.append('..')

LO_VOL_SIZE = (512, 512, 3, 1, )
NC = 4

UNet = UNetGen(input_shape=LO_VOL_SIZE, starting_channels=NC)
print(UNet.summary())
UNet.load_weights("C:/Users/roybo/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/001_CNN_Robotic_Needle_Seg/models/nc4_mb4_ep100_eta0.001/nc4_mb4_ep100_eta0.001.ckpt")
print([var.shape for var in UNet.trainable_variables])
input_image, _ = nrrd.read('5 4.0 Cryo CTF  CE.nrrd')

input_image = ((input_image - input_image.min()) / (input_image.max() - input_image.min())).astype(np.float32)
input_tensor = tf.convert_to_tensor(input_image[np.newaxis, :, :, :, np.newaxis])
layers = UNet(input_tensor)
print([layer.numpy().shape for layer in layers])
# 256, 128, 64, 32, 32, 64, 128, 256, 512, 512
# Needle tip F (276, 230)???, S (306, 130)

with tf.GradientTape() as tape:
    tape.watch(input_tensor)
    pred = UNet(input_tensor)[9][0, 256:356, 80:180, 0, 0]
    gradients = tape.gradient(pred, input_tensor)

gradients_np = np.squeeze(gradients.numpy())
# gradients_np = (gradients_np - gradients_np.min()) / (gradients_np.max() - gradients_np.min())

fig, axs = plt.subplots(1, 2)
axs[0].imshow(np.fliplr(input_image[:, :, 0].T), cmap='gray', origin='lower')
axs[0].axis('off')
axs[1].imshow(np.fliplr(gradients_np[:, :, 0].T), cmap='plasma', origin='lower')
axs[1].axis('off')
# axs[2].imshow(np.fliplr(pred[0, :, :, 0, 0].numpy().T), cmap='gray', origin='lower')
# axs[2].axis('off')
plt.show()