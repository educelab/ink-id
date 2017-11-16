import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import layers
import pdb


def buildModel(x, y, keep_prob, args):
    x = (tf.reshape(x, [-1, args["x_dimension"], args["y_dimension"], args["z_dimension"], 1]))
    conv1 = layers.batch_normalization(slim.convolution(x, args["neurons"][0], [3, 3, 3], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv2 = layers.batch_normalization(slim.convolution(conv1, args["neurons"][1], [3, 3, 3], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv3 = layers.batch_normalization(slim.convolution(conv2, args["neurons"][2], [3, 3, 3], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv4 = layers.batch_normalization(slim.convolution(conv3, args["neurons"][3], [3, 3, 3], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv5 = layers.batch_normalization(slim.convolution(conv4, args["neurons"][4], [3, 3, 3], stride=[1,1,2], padding='same'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])

    net = layers.dropout(slim.fully_connected(slim.flatten(conv5), args["n_classes"], activation_fn=None), rate=drop_rate)#, training=training_flag)

    zsquashed = tf.reshape(conv6, [-1, args["x_Dimension"], args["y_Dimension"], 32])
    print("SHAPE: {}".format(conv6.shape))
    # now, input is 64x64x1

    conv2d1 = slim.batch_norm(slim.convolution(zsquashed, 16, [3, 3], stride=1, padding='same'))
    conv2d2 = slim.batch_norm(slim.convolution(conv2d1, 8, [3, 3], stride=1, padding='same'))
    conv2d3 = slim.batch_norm(slim.convolution(conv2d2, 4, [3, 3], stride=1, padding='same'))
    conv2d4 = slim.batch_norm(slim.convolution(conv2d3, 2, [3, 3], stride=1, padding='same'))

    convt6 = layers.dropout(slim.batch_norm(layers.conv2d(conv2d4, 2, [1, 1], strides=1, padding='same')), keep_prob)

    pred = tf.reshape(convt6, [-1, args["x_Dimension"]*args["y_Dimension"], 2])
    truth = tf.reshape(y, [-1, args["x_Dimension"]*args["y_Dimension"]])

    x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=truth)
    mean_x_entropy = tf.reduce_mean(x_entropy)

    return tf.nn.softmax(convt6), mean_x_entropy
