import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import layers
import pdb


def buildModel(x, y, drop_rate, args, training_flag):
    #batch_shape = x.get_shape().as_list()
    #x = (tf.reshape(x, [-1, batch_shape[1], batch_shape[2], args["z_dimension"], 1]))
    neurons = [4,4,4,4,4,  2,2,2,2,2,  1,1,1,1,1]
    conv1 = layers.batch_normalization(slim.convolution(x, neurons[0], [3, 3, 3], stride=[1,1,1], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv2 = layers.batch_normalization(slim.convolution(conv1, neurons[1], [3, 3, 3], stride=[1,1,1], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv3 = layers.batch_normalization(slim.convolution(conv2, neurons[2], [3, 3, 3], stride=[1,1,1], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv4 = layers.batch_normalization(slim.convolution(conv3, neurons[3], [3, 3, 3], stride=[1,1,1], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv5 = layers.batch_normalization(slim.convolution(conv4, neurons[4], [3, 3, 3], stride=[1,1,1], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    # receptive field after conv5: 11x11x11 (volume: X x Y x Z)

    conv6 = layers.batch_normalization(slim.convolution(conv5, neurons[5], [3, 3, 3], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv7 = layers.batch_normalization(slim.convolution(conv6, neurons[6], [3, 3, 3], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv8 = layers.batch_normalization(slim.convolution(conv7, neurons[7], [3, 3, 3], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv9 = layers.batch_normalization(slim.convolution(conv8, neurons[8], [3, 3, 3], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv10 = layers.batch_normalization(slim.convolution(conv9, neurons[9], [3, 3, 3], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    # receptive field after conv10: 21x21x48 (volume: X x Y x 1)

    conv11 = layers.batch_normalization(slim.convolution(conv10, neurons[10], [3, 3, 1], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv12 = layers.batch_normalization(slim.convolution(conv11, neurons[11], [3, 3, 1], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv13 = layers.batch_normalization(slim.convolution(conv12, neurons[12], [3, 3, 1], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv14 = layers.batch_normalization(slim.convolution(conv13, neurons[13], [3, 3, 1], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv15 = layers.batch_normalization(slim.convolution(conv14, neurons[14], [3, 3, 1], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    # receptive field after conv10: 31x31x48 (volume: X x Y x 1)

    print("Conv15 shape: {}".format(conv15.get_shape().as_list()))
    pred = tf.squeeze(conv15, axis=[3,4])
    print("Pred shape: {}".format(pred.get_shape().as_list()))

    x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)
    mean_x_entropy = tf.reduce_mean(x_entropy)

    return tf.nn.sigmoid(pred), mean_x_entropy
