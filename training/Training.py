from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import tensorflow.keras as keras
import tensorflow as tf
import time

sys.path.append('..')
sys.path.append('/home/mpinnock/CNN_Robotic_Needle_Seg/scripts/')

from Networks import UNetGen
from utils.DataLoader import imgLoader
from utils.TrainFuncs import trainStep, valStep


parser = ArgumentParser()
parser.add_argument('--file_path', '-fp', help="File path", type=str)
parser.add_argument('--data_path', '-dp', help="Data path", type=str)
# parser.add_argument('--data_aug', '-da', help="Data augmentation", action='store_true')
parser.add_argument('--minibatch_size', '-mb', help="Minibatch size", type=int, nargs='?', const=4, default=4)
parser.add_argument('--num_chans', '-nc', help="Starting number of channels", type=int, nargs='?', const=32, default=32)
parser.add_argument('--epochs', '-ep', help="Number of epochs", type=int, nargs='?', const=5, default=5)
parser.add_argument('--folds', '-f', help="Number of cross-validation folds", type=int, nargs='?', const=0, default=0)
parser.add_argument('--crossval', '-c', help="Fold number", type=int, nargs='?', const=0, default=0)
parser.add_argument('--gpu', '-g', help="GPU number", type=int, nargs='?', const=0, default=0)
parser.add_argument('--eta', '-e', help="Learning rate", type=float, nargs='?', const=0.001, default=0.001)
arguments = parser.parse_args()

if arguments.file_path == None:
    FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/001_CNN_Robotic_Needle_Seg/"
else:
    FILE_PATH = arguments.file_path

if arguments.data_path == None:
    DATA_PATH = "Z:/Robot_Data/"
else:
    DATA_PATH = arguments.data_path

MB_SIZE = arguments.minibatch_size
NC = arguments.num_chans
EPOCHS = arguments.epochs
NUM_FOLDS = arguments.folds
FOLD = arguments.crossval
ETA = arguments.eta
NUM_EX = 4

if FOLD >= NUM_FOLDS and NUM_FOLDS != 0:
   raise ValueError("Fold number cannot be greater or equal to number of folds")

GPU = arguments.gpu

EXPT_NAME = f"nc{NC}_mb{MB_SIZE}_ep{EPOCHS}_eta{ETA}"

if NUM_FOLDS > 0:
    EXPT_NAME += f"_cv{FOLD}"

MODEL_SAVE_PATH = f"{FILE_PATH}models/{EXPT_NAME}/"

if not os.path.exists(MODEL_SAVE_PATH) and NUM_FOLDS == 0:
    os.mkdir(MODEL_SAVE_PATH)

IMAGE_SAVE_PATH = f"{FILE_PATH}images/{EXPT_NAME}/"

if not os.path.exists(IMAGE_SAVE_PATH) and NUM_FOLDS == 0:
    os.mkdir(IMAGE_SAVE_PATH)

if arguments.file_path == None:
    LOG_SAVE_PATH = f"{FILE_PATH}/{EXPT_NAME}.txt"
else:
    LOG_SAVE_PATH = f"{FILE_PATH}reports/{EXPT_NAME}.txt"

img_path = f"{DATA_PATH}Img/"
seg_path = f"{DATA_PATH}Seg/"
imgs = os.listdir(img_path)
segs = os.listdir(seg_path)
imgs.sort()
segs.sort()

N = len(imgs)
assert N == len(segs), "HI/LO IMG PAIRS UNEVEN LENGTHS"

LO_VOL_SIZE = (512, 512, 3, 1, )

random.seed(10)
temp_list = list(zip(imgs, segs))
random.shuffle(temp_list)
imgs, segs = zip(*temp_list)

if NUM_FOLDS == 0:
    img_train = imgs
    seg_train = segs
    ex_indices = np.random.choice(len(img_train), NUM_EX)
    img_examples = np.array(img_train)[ex_indices]
    seg_examples = np.array(seg_train)[ex_indices]
    img_examples = [s.encode("utf-8") for s in img_examples]
    seg_examples = [s.encode("utf-8") for s in seg_examples]
else:
    num_in_fold = int(N / NUM_FOLDS)
    img_val = imgs[FOLD * num_in_fold:(FOLD + 1) * num_in_fold]
    seg_val = segs[FOLD * num_in_fold:(FOLD + 1) * num_in_fold]
    img_train = imgs[0:FOLD * num_in_fold] + imgs[(FOLD + 1) * num_in_fold:]
    seg_train = segs[0:FOLD * num_in_fold] + segs[(FOLD + 1) * num_in_fold:]
    ex_indices = np.random.choice(len(img_val), NUM_EX)
    img_examples = np.array(img_val)[ex_indices]
    seg_examples = np.array(seg_val)[ex_indices]
    img_examples = [s.encode("utf-8") for s in img_examples]
    seg_examples = [s.encode("utf-8") for s in seg_examples]

train_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[img_path, seg_path, img_train, seg_train, True], output_types=(tf.float32, tf.float32))

if NUM_FOLDS > 0:
    val_ds = tf.data.Dataset.from_generator(
        imgLoader, args=[img_path, seg_path, img_val, seg_val, False], output_types=(tf.float32, tf.float32))

UNet = UNetGen(input_shape=LO_VOL_SIZE, starting_channels=NC)

if arguments.file_path == None:
    print(UNet.summary())

train_metric = 0
val_metric = 0
train_count = 0
val_count = 0
Optimiser = keras.optimizers.Adam(ETA)

for epoch in range(EPOCHS):
    for img, seg in train_ds.batch(MB_SIZE):
        train_metric += trainStep(img, seg, UNet, Optimiser)
        train_count += 1

    if NUM_FOLDS > 0:
        for img, seg in val_ds.batch(MB_SIZE):
            val_metric += valStep(img, seg, UNet)
            val_count += 1
    else:
        val_count += 1e-6

    print(f"Epoch: {epoch + 1}, Train Loss: {train_metric / (train_count)}, Val Loss: {val_metric / (val_count)}")
    train_metric = 0
    val_metric = 0

    fig, axs = plt.subplots(4, NUM_EX)

    for i in range(4):
        for j in range(NUM_EX):
            for data in imgLoader(img_path.encode("utf-8"), seg_path.encode("utf-8"), [img_examples[j]], [seg_examples[j]], False):
                img = data[0]
                seg = data[1]

            pred = UNet(img[np.newaxis, ...])

            axs[0, j].imshow(np.fliplr(img[:, :, 1, 0].T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
            axs[0, j].axis('off')
            axs[1, j].imshow(np.fliplr(pred[0, :, :, 1, 0].numpy().T), cmap='hot', origin='lower')
            axs[1, j].axis('off')
            axs[2, j].imshow(np.fliplr(seg[:, :, 1, 0].T), cmap='gray', origin='lower')
            axs[2, j].axis('off')
            axs[3, j].imshow(np.fliplr(img[:, :, 1, 0].T * pred[0, :, :, 1, 0].numpy().T), cmap='gray', origin='lower')
            axs[3, j].axis('off')

    plt.tight_layout()
    plt.savefig(f"{IMAGE_SAVE_PATH}/Epoch_{epoch + 1}.png", dpi=250)
    plt.close()