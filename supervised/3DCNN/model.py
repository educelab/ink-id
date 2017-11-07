import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer as xavier
from tensorflow import layers
import pdb


def buildModel(x, y, drop_rate, args, training_flag):
    x = (tf.reshape(x, [-1, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], 1]))
    conv1 = layers.batch_normalization(slim.convolution(x, args["neurons"][0], [3, 3, 3], stride=[2,2,2], padding='valid'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv2 = layers.batch_normalization(slim.convolution(conv1, args["neurons"][1], [3, 3, 3], stride=[2,2,2], padding='valid'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv3 = layers.batch_normalization(slim.convolution(conv2, args["neurons"][2], [3, 3, 3], stride=[2,2,2], padding='valid'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])
    conv4 = layers.batch_normalization(slim.convolution(conv3, args["neurons"][3], [3, 3, 3], stride=[2,2,2], padding='valid'), training=training_flag, scale=False, axis=4, momentum=args["batch_norm_momentum"])

    net = layers.dropout(slim.fully_connected(slim.flatten(conv4), args["n_classes"], activation_fn=None), rate=drop_rate)#, training=training_flag)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=net))
    #targets_1d = y[:,1]
    #logits_1d = net[:,1]
    #loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits_1d, targets=targets_1d, pos_weight=args["pos_weight"]))

    return tf.nn.softmax(net), loss



def buildMultitaskModel(x, y, drop_rate, args):
    x = tf.reshape(x, [-1, args["x_dimension"], args["y_dimension"], args["z_dimension"], 1])
    conv1 = slim.batch_norm(slim.convolution(x, args["neurons"][0], [3, 3, 3], stride=[2,2,2], padding='valid'))
    conv2 = slim.batch_norm(slim.convolution(conv1, args["neurons"][1], [3, 3, 3], stride=[2,2,2], padding='valid'))
    conv3 = slim.batch_norm(slim.convolution(conv2, args["neurons"][2], [3, 3, 3], stride=[2,2,2], padding='valid'))
    conv4 = slim.batch_norm(slim.convolution(conv3, args["neurons"][3], [3, 3, 3], stride=[1,1,1], padding='valid'))
    conv5 = slim.batch_norm(slim.convolution(conv4, args["neurons"][3], [3, 3, 3], stride=[1,1,1], padding='valid'))

    shallow_net = layers.dropout(slim.fully_connected(slim.flatten(conv4), args["n_classes"], activation_fn=None), rate=drop_rate)
    net = layers.dropout(slim.fully_connected(slim.flatten(conv5), args["n_classes"], activation_fn=None), rate=drop_rate)

    shallow_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=shallow_net))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=net))

    return tf.nn.softmax(net), shallow_loss, loss
