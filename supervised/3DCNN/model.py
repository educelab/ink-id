import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb


def buildModel(x, y, args):
    x = tf.reshape(x, [-1, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], 1])
    conv1 = slim.batch_norm(slim.convolution(x, args["neurons"][0], args["receptiveField"], stride=[2,2,2]))
    conv2 = slim.batch_norm(slim.convolution(conv1, args["neurons"][1], args["receptiveField"], stride=[1,1,1]))
    conv3 = slim.batch_norm(slim.convolution(conv2, args["neurons"][2], args["receptiveField"], stride=[2,2,2]))
    conv4 = slim.batch_norm(slim.convolution(conv3, args["neurons"][3], args["receptiveField"], stride=[2,2,2]))

    net = tf.nn.dropout(slim.fully_connected(slim.flatten(conv4), args["n_Classes"], activation_fn=None), args["dropout"])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))

    return tf.nn.softmax(net), loss
