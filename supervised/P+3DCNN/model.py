import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb


def buildModel(x, y, args):
    x = tf.reshape(x, [-1, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], args["numChannels"]])
    conv1 = slim.batch_norm(slim.convolution(x, 50, [5,5,5], stride=[1,1,1]))
    conv2 = slim.batch_norm(slim.convolution(conv1, 50, [5,5,5], stride=[2,2,1]))
    conv3 = slim.batch_norm(slim.convolution(conv2, 50, [5,5,5], stride=[1,1,1]))
    conv4 = slim.batch_norm(slim.convolution(conv3, 50, [5,5,5], stride=[2,2,1]))
    # conv5 = slim.batch_norm(slim.convolution(conv3, 128, [3,3,3], stride=[2,2,1]))

    pred = tf.nn.dropout(slim.fully_connected(slim.flatten(conv4), args["n_Classes"], activation_fn=None), args["dropout"])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

    return tf.nn.softmax(pred), loss
