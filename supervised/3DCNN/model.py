import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb


def buildModel(x, y, args):
    x = tf.reshape(x, [-1, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], 1])
    conv1 = slim.batch_norm(slim.convolution(x, 8, [3,3,3], stride=2))
    conv2 = slim.batch_norm(slim.convolution(conv1, 16, [3,3,3], stride=2))
    conv3 = slim.batch_norm(slim.convolution(conv2, 32, [3,3,3], stride=2))
    conv4 = slim.batch_norm(slim.convolution(conv3, 64, [3,3,3], stride=2))
    conv5 = slim.batch_norm(slim.convolution(conv3, 128, [3,3,3], stride=2))

    pred = tf.nn.dropout(slim.fully_connected(slim.flatten(conv5), args["n_Classes"], activation_fn=None), args["dropout"])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    return pred, loss
