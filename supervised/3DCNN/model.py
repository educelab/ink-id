import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb


def buildModel(x, y, args):
    x = tf.reshape(x, [-1, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], 1])
    conv1 = slim.batch_norm(slim.convolution(x, args["layer1_neurons"], args["receptiveField"], stride=[2,2,2]))
    conv2 = slim.batch_norm(slim.convolution(conv1, 2*args["layer1_neurons"], args["receptiveField"], stride=[1,1,1]))
    conv3 = slim.batch_norm(slim.convolution(conv2, 4*args["layer1_neurons"], args["receptiveField"], stride=[2,2,2]))
    conv4 = slim.batch_norm(slim.convolution(conv3, 8*args["layer1_neurons"], args["receptiveField"], stride=[2,2,2]))
    #conv5 = slim.batch_norm(slim.convolution(conv3, 16*args["layer1_neurons"], args["receptiveField"], stride=[2,2,2]))

    #conv7 = slim.batch_norm(slim.convolution(conv6, 256, args["receptiveField"], stride=2))
    #conv8 = slim.batch_norm(slim.convolution(conv7, 512, args["receptiveField"], stride=2))

    #fc1 = tf.nn.dropout(slim.fully_connected(slim.flatten(conv3), 8*args["layer1_neurons"]), args["dropout"])
    net = tf.nn.dropout(slim.fully_connected(slim.flatten(conv4), args["n_Classes"], activation_fn=None), args["dropout"])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=y))
    # a positive error is weighted to prioritize precision over recall
    #loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=net, targets=y, pos_weight=4.0))

    return tf.nn.softmax(net), loss
