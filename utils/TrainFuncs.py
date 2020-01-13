import tensorflow as tf
import tensorflow.keras as keras


@tf.function
def diceLoss(pred, mask):
    numer = tf.reduce_sum(pred * mask, axis=[1, 2, 3, 4]) * 2
    denom = tf.reduce_sum(pred, axis=[1, 2, 3, 4]) + tf.reduce_sum(mask, axis=[1, 2, 3, 4]) + 1e-6
    dice = numer / denom

    return 1 - tf.reduce_mean(dice)


@tf.function
def trainStep(imgs, segs, Model, ModelOptimiser):
    with tf.GradientTape() as tape:
        prediction = Model(imgs, training=True)
        loss = diceLoss(prediction, segs)
        gradients = tape.gradient(loss, Model.trainable_variables)
        ModelOptimiser.apply_gradients(zip(gradients, Model.trainable_variables))

        return loss


@tf.function
def valStep(imgs, labels, Model):
    prediction = Model(imgs, training=False)
    loss = diceLoss(prediction, labels)
    
    return loss
