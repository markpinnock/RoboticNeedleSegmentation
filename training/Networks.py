import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def dnResNetBlock(nc, inputlayer, drop_rate, dropout_flag):
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(inputlayer)
    BN1 = tf.nn.relu(keras.layers.BatchNormalization()(conv1))
    drop1 = keras.layers.SpatialDropout3D(drop_rate)(BN1, training=dropout_flag)
    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(drop1)
    BN2 = tf.nn.relu(keras.layers.BatchNormalization()(conv2))
    drop2 = keras.layers.SpatialDropout3D(drop_rate)(BN2, training=dropout_flag)
    pool = keras.layers.MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(drop2)
    return BN2, pool


def upResNetBlock(nc, inputlayer, skip, tconv_strides, drop_rate, dropout_flag):
    tconv = keras.layers.Conv3DTranspose(nc, (3, 3, 3), strides=tconv_strides, padding='same')(inputlayer)
    BN1 = tf.nn.relu(keras.layers.BatchNormalization()(tconv))
    drop1 = keras.layers.SpatialDropout3D(drop_rate)(BN1, training=dropout_flag)
    concat = keras.layers.concatenate([drop1, skip], axis=4)
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(concat)
    BN2 = tf.nn.relu(keras.layers.BatchNormalization()(conv1))
    drop2 = keras.layers.SpatialDropout3D(drop_rate)(BN2, training=dropout_flag)
    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(drop2)
    BN3 = tf.nn.relu(keras.layers.BatchNormalization()(conv2))
    drop3 = keras.layers.SpatialDropout3D(drop_rate)(BN3, training=dropout_flag)
    return drop3


def UNetGen(input_shape, starting_channels, drop_rate=0.0, dropout_flag=True):
    inputlayer = keras.layers.Input(shape=input_shape)
    nc = starting_channels

    skip1, dnres1 = dnResNetBlock(nc, inputlayer, drop_rate, dropout_flag)
    skip2, dnres2 = dnResNetBlock(nc * 2, dnres1, drop_rate, dropout_flag)
    skip3, dnres3 = dnResNetBlock(nc * 4, dnres2, drop_rate, dropout_flag)
    skip4, dnres4 = dnResNetBlock(nc * 8, dnres3, drop_rate, dropout_flag)
    dn5 = keras.layers.Conv3D(nc * 16, (3, 3, 3), strides=(1, 1, 1), padding='same')(dnres4)
    BN = tf.nn.relu(keras.layers.BatchNormalization()(dn5))
    drop = keras.layers.SpatialDropout3D(drop_rate)(BN, training=dropout_flag)

    upres4 = upResNetBlock(nc * 8, BN, skip4, (2, 2, 1), drop_rate, dropout_flag)
    upres3 = upResNetBlock(nc * 4, upres4, skip3, (2, 2, 1), drop_rate, dropout_flag)
    upres2 = upResNetBlock(nc * 2, upres3, skip2, (2, 2, 1), drop_rate, dropout_flag)
    upres1 = upResNetBlock(nc, upres2, skip1, (2, 2, 1), drop_rate, dropout_flag)

    outputlayer = keras.layers.Conv3D(1, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='sigmoid')(upres1)
    
    return keras.Model(inputs=inputlayer, outputs=outputlayer)


def ResNet(input_shape, starting_channels):
    inputlayer = keras.layers.Input(shape=input_shape)
    nc = starting_channels

    convblock1 = dnResNetBlock(nc, inputlayer)
    convblock2 = dnResNetBlock(nc * 2, convblock1)
    convblock3 = dnResNetBlock(nc * 4, convblock2)
    convblock4 = dnResNetBlock(nc * 8, convblock3)
    convblock5 = dnResNetBlock(nc * 16, convblock4)
    flat_layer = keras.layers.Flatten()(convblock5)
    dense = keras.layers.Dense(nc * 16, activation='relu')(flat_layer)
    dropout = keras.layers.Dropout(0.4)(dense)
    outputlayer = keras.layers.Dense(2, activation='relu')(dropout)

    return keras.Model(inputs=inputlayer, outputs=outputlayer)
