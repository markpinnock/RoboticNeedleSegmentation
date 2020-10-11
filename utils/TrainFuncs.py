import tensorflow as tf
import tensorflow.keras as keras


@tf.function
def dice_loss(pred, mask):

    """ Implements Dice loss
        - pred: predicted segmentation
        - mask: ground truth label """

    numer = tf.reduce_sum(pred * mask, axis=[1, 2, 3, 4]) * 2
    denom = tf.reduce_sum(pred, axis=[1, 2, 3, 4]) + tf.reduce_sum(mask, axis=[1, 2, 3, 4]) + 1e-6
    dice = numer / denom

    return 1 - tf.reduce_mean(dice)


@tf.function
def train_step(imgs, segs, Model, ModelOptimiser):

    """ Implements training step
        - imgs: input images
        - segs: segmentation labels
        - Model: model to be trained (keras.Model)
        - ModelOptimiser: e.g. keras.optimizers.Adam() """

    # TODO: subclass UNet and convert train_step to class method
    with tf.GradientTape() as tape:
        prediction = Model(imgs, training=True)
        loss = diceLoss(prediction, segs)
        gradients = tape.gradient(loss, Model.trainable_variables)
        ModelOptimiser.apply_gradients(zip(gradients, Model.trainable_variables))

        return loss


@tf.function
def valStep(imgs, labels, Model):
    """ Implements validation step """

    # TODO: subclass UNet and convert val_step to class method
    prediction = Model(imgs, training=False)
    loss = diceLoss(prediction, labels)
    
    return loss
