'''
ops.py
    - various wrapper functions
'''

__author__ = "Kendall Weihe"
__email__ = "kendall.weihe@uky.edu"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def conv3d(x, weight, bias, padding="SAME", strides=[2,2,2]):
    if padding == "SAME":
        conv = tf.nn.convolution(x, weight, padding="SAME", dilation_rate=[3,3,3])
    else:
        conv = tf.nn.convolution(x, weight, padding="VALID", dilation_rate=[3,3,3])

    conv = tf.nn.bias_add(conv, bias)
    # conv = lrelu(conv)
    convForward = tf.nn.relu(conv)
    convForward = tf.nn.pool(convForward, [3,3,3], pooling_type="AVG", padding="SAME", strides=strides)
    convForward = slim.batch_norm(convForward)
    return conv, convForward


def conv3d_transpose(x, weight, bias, outputShape, strides=2, padding="SAME", activation_fn="relu"):
    if padding == "SAME":
        deconv = tf.nn.conv3d_transpose(x, weight, output_shape=outputShape, strides=[1,strides,strides,strides,1], padding="SAME")
    else:
        deconv = tf.nn.conv3d_transpose(x, weight, output_shape=outputShape, strides=[1,strides,strides,strides,1], padding="VALID")
    deconv = tf.nn.bias_add(deconv, bias)

    if activation_fn == None:
        return deconv, deconv

    deconvForward = tf.nn.relu(deconv)
    deconvForward = slim.batch_norm(deconvForward)
    return deconv, deconvForward
