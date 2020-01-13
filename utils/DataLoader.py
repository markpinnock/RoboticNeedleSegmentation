import numpy as np
import os
import random
import tensorflow as tf


def imgLoader(img_path, seg_path, img_list, seg_list, shuffle_flag):
    img_path = img_path.decode("utf-8")
    seg_path = seg_path.decode("utf-8")

    if shuffle_flag == True:
        temp_list = list(zip(img_list, seg_list))
        np.random.shuffle(temp_list)
        img_list, seg_list = zip(*temp_list)

    N = len(img_list)
    i = 0 

    while i < N:
        try:
            img_name = img_list[i].decode("utf-8")
            img_vol = np.load(img_path + img_name).astype(np.float32)

            seg_name = img_name[:-5] + "M.npy"
            seg_vol = np.load(seg_path  + seg_name).astype(np.float32)

        except Exception as e:
            print(f"IMAGE OR MASK LOAD FAILURE: {img_name} ({e})")

        else:
            yield (img_vol[::1, ::1, :, np.newaxis], seg_vol[::1, ::1, :, np.newaxis])
        finally:
            i += 1

# Data aug

if __name__ == "__main__":

    FILE_PATH = "Z:/Robot_Data/"
    img_path = f"{FILE_PATH}Img/"
    seg_path = f"{FILE_PATH}Seg/"
    imgs = os.listdir(img_path)
    segs = os.listdir(seg_path)
    imgs.sort()
    segs.sort()

    N = len(imgs)
    NUM_FOLDS = 5
    FOLD = 0
    MB_SIZE = 8
    random.seed(10)

    for i in range(N):
        print(imgs[i], segs[i])
        assert imgs[i][:-5] == segs[i][:-5], "HI/LO PAIRS DON'T MATCH"

    temp_list = list(zip(imgs, segs))
    random.shuffle(temp_list)
    imgs, segs = zip(*temp_list)

    for i in range(N):
        # print(imgs[i], segs[i])
        assert imgs[i][:-5] == segs[i][:-5], "HI/LO PAIRS DON'T MATCH"

    num_in_fold = int(N / NUM_FOLDS)
    img_val = imgs[FOLD * num_in_fold:(FOLD + 1) * num_in_fold]
    seg_val = segs[FOLD * num_in_fold:(FOLD + 1) * num_in_fold]
    img_train = imgs[0:FOLD * num_in_fold] + imgs[(FOLD + 1) * num_in_fold:]
    seg_train = segs[0:FOLD * num_in_fold] + segs[(FOLD + 1) * num_in_fold:]

    for i in range(len(img_val)):
        # print(img_val[i], seg_val[i])
        assert img_val[i][:-5] == seg_val[i][:-5], "HI/LO PAIRS DON'T MATCH"
    
    for i in range(len(img_train)):
        # print(img_train[i], seg_train[i])
        assert img_train[i][:-5] == seg_train[i][:-5], "HI/LO PAIRS DON'T MATCH"

    print(f"N: {N}, val: {len(img_val)}, train: {len(img_train)}, val + train: {len(img_val) + len(img_train)}")
    
    train_ds = tf.data.Dataset.from_generator(
        imgLoader, args=[img_path, seg_path, img_train, seg_train, True], output_types=(tf.float32, tf.float32))

    val_ds = tf.data.Dataset.from_generator(
        imgLoader, args=[img_path, seg_path, img_val, seg_val, False], output_types=(tf.float32, tf.float32))
    
    for data in train_ds.batch(MB_SIZE):
        print(data[0].shape, data[1].shape)
        # pass
    
    for data in val_ds.batch(MB_SIZE):
        print(data[0].shape, data[1].shape)
        # pass
