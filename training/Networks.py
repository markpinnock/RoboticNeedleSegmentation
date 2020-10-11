import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def down_block(nc, inputlayer):

    """ Implements downsampling block for 3D UNet
        - nc: number of channels in block
        - inputlayer: input of type keras.layers.Input

        Returns: output before (skip) and after pooling """

    # TODO: REMOVE BATCHNORM GIVEN SMALL MB SIZES AND TRY E.G. INSTANCE/GROUP NORM
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(inputlayer)
    BN1 = tf.nn.relu(keras.layers.BatchNormalization()(conv1))
    
    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(BN1)
    BN2 = tf.nn.relu(keras.layers.BatchNormalization()(conv2))
    
    # TODO: CONSIDER CHANGING TO CONV STRIDE 2
    pool = keras.layers.MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(conv2)
    return BN2, pool


def up_block(nc, inputlayer, skip, tconv_strides):

    """ Implements upsampling block for 3D UNet
        - nc: number of channels in block
        - inputlayer: input of type keras.layers.Input
        - skip: skip_layer from encoder
        - tconv_strides: number of strides for transpose convolution (a, b, c) """

    tconv = keras.layers.Conv3DTranspose(nc, (3, 3, 3), strides=tconv_strides, padding='same')(inputlayer)
    BN1 = tf.nn.relu(keras.layers.BatchNormalization()(tconv))
    
    concat = keras.layers.concatenate([BN1, skip], axis=4)
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(concat)
    BN2 = tf.nn.relu(keras.layers.BatchNormalization()(conv1))
    
    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same')(BN2)
    BN3 = tf.nn.relu(keras.layers.BatchNormalization()(conv2))
    return BN3


def UNetGen(input_shape, nc): 
   
    """ Implements UNet
        - input_shape: size of input volumes
        - nc: number of channels in first block """

    # TODO: SUBCLASS keras.Model
    inputlayer = keras.layers.Input(shape=input_shape)

    # Encoder
    skip1, dnres1 = down_block(nc, inputlayer)
    skip2, dnres2 = down_block(nc * 2, dnres1)
    skip3, dnres3 = down_block(nc * 4, dnres2)
    skip4, dnres4 = down_block(nc * 8, dnres3)
    dn5 = keras.layers.Conv3D(nc * 16, (3, 3, 3), strides=(1, 1, 1), padding='same')(dnres4)
    BN = tf.nn.relu(keras.layers.BatchNormalization()(dn5))

    # Decoder
    upres4 = up_block(nc * 8, BN, skip4, (2, 2, 1))
    upres3 = up_block(nc * 4, upres4, skip3, (2, 2, 1))
    upres2 = up_block(nc * 2, upres3, skip2, (2, 2, 1))
    upres1 = up_block(nc, upres2, skip1, (2, 2, 1))

    outputlayer = keras.layers.Conv3D(1, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='sigmoid')(upres1)

    return keras.Model(inputs=inputlayer, outputs=outputlayer)


def ResNet(input_shape, nc):
    
    """ Alternative model for needle tip prediction
        - input_shape: size of input volumes
        - nc: number of channels in first block
        
        Returns coordinates of needle tip - not yet tested """

    inputlayer = keras.layers.Input(shape=input_shape)
    # TODO: RESIDUAL CONNECTIONS
    convblock1, _ = down_block(nc, inputlayer)
    convblock2, _ = down_block(nc * 2, convblock1)
    convblock3, _ = down_block(nc * 4, convblock2)
    convblock4, _ = down_block(nc * 8, convblock3)
    convblock5, _ = down_block(nc * 16, convblock4)

    flat_layer = keras.layers.Flatten()(convblock5)
    dense = keras.layers.Dense(nc * 16, activation='relu')(flat_layer)
    dropout = keras.layers.Dropout(0.4)(dense)
    outputlayer = keras.layers.Dense(2, activation='relu')(dropout)

    return keras.Model(inputs=inputlayer, outputs=outputlayer)
