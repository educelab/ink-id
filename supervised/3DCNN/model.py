import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb


def buildModel(x, y, args):
    x = tf.reshape(x, [-1, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], 1])
    conv1 = slim.batch_norm(slim.convolution(x, 2, args["receptiveField"], stride=2))
    conv2 = slim.batch_norm(slim.convolution(conv1, 8, args["receptiveField"], stride=2))
    conv3 = slim.batch_norm(slim.convolution(conv2, 16, args["receptiveField"], stride=2))
    conv4 = slim.batch_norm(slim.convolution(conv3, 32, args["receptiveField"], stride=1))
    conv5 = slim.batch_norm(slim.convolution(conv4, 64, args["receptiveField"], stride=2))
    conv6 = slim.batch_norm(slim.convolution(conv5, 128, args["receptiveField"], stride=2))
    conv7 = slim.batch_norm(slim.convolution(conv6, 256, args["receptiveField"], stride=2))
    conv8 = slim.batch_norm(slim.convolution(conv7, 512, args["receptiveField"], stride=1))

    fc1 = tf.nn.dropout(slim.fully_connected(slim.flatten(conv8), 1024), args["dropout"])
    net = tf.nn.dropout(slim.fully_connected(fc1, args["n_Classes"], activation_fn=None), args["dropout"])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))

    return tf.nn.softmax(net), loss
