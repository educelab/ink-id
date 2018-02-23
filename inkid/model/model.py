"""
Functions for building the tf model.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import layers


def build_model(inputs, labels, drop_rate, args, training_flag):
    """Build a model."""
    (coordinates, subvolumes) = inputs
    subvolumes_internal = (tf.reshape(subvolumes,
                            [-1, args["subvolume_dimension_x"], args["subvolume_dimension_y"], args["subvolume_dimension_z"], 1]))
    conv1 = layers.batch_normalization(slim.convolution(subvolumes_internal, args["neurons"][0], [3, 3, 3],
                                                        stride=[2, 2, 2], padding='valid'),
                                       training=training_flag,
                                       scale=False,
                                       axis=4,
                                       momentum=args["batch_norm_momentum"])
    conv2 = layers.batch_normalization(slim.convolution(conv1, args["neurons"][1], [3, 3, 3],
                                                        stride=[2, 2, 2], padding='valid'),
                                       training=training_flag,
                                       scale=False,
                                       axis=4,
                                       momentum=args["batch_norm_momentum"])
    conv3 = layers.batch_normalization(slim.convolution(conv2, args["neurons"][2], [3, 3, 3],
                                                        stride=[2, 2, 2], padding='valid'),
                                       training=training_flag,
                                       scale=False,
                                       axis=4,
                                       momentum=args["batch_norm_momentum"])
    conv4 = layers.batch_normalization(slim.convolution(conv3, args["neurons"][3], [3, 3, 3],
                                                        stride=[2, 2, 2], padding='valid'),
                                       training=training_flag,
                                       scale=False,
                                       axis=4,
                                       momentum=args["batch_norm_momentum"])

    net = layers.dropout(slim.fully_connected(slim.flatten(conv4),
                                              args["n_classes"],
                                              activation_fn=None),
                         rate=drop_rate)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=net))

    return tf.nn.softmax(net), loss, subvolumes, coordinates



def build_multitask_model(subvolume, labels, drop_rate, args):
    """Build a multitask model."""
    subvolume = tf.reshape(subvolume,
                           [-1, args["subvolume_dimension_x"], args["subvolume_dimension_y"], args["subvolume_dimension_z"], 1])
    conv1 = slim.batch_norm(slim.convolution(subvolume, args["neurons"][0], [3, 3, 3],
                                             stride=[2, 2, 2], padding='valid'))
    conv2 = slim.batch_norm(slim.convolution(conv1, args["neurons"][1], [3, 3, 3],
                                             stride=[2, 2, 2], padding='valid'))
    conv3 = slim.batch_norm(slim.convolution(conv2, args["neurons"][2], [3, 3, 3],
                                             stride=[2, 2, 2], padding='valid'))
    conv4 = slim.batch_norm(slim.convolution(conv3, args["neurons"][3], [3, 3, 3],
                                             stride=[1, 1, 1], padding='valid'))
    conv5 = slim.batch_norm(slim.convolution(conv4, args["neurons"][3], [3, 3, 3],
                                             stride=[1, 1, 1], padding='valid'))

    shallow_net = layers.dropout(slim.fully_connected(slim.flatten(conv4), args["n_classes"],
                                                      activation_fn=None),
                                 rate=drop_rate)
    net = layers.dropout(slim.fully_connected(slim.flatten(conv5), args["n_classes"],
                                              activation_fn=None),
                         rate=drop_rate)

    shallow_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                                             logits=shallow_net))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                                     logits=net))

    return tf.nn.softmax(net), shallow_loss, loss
