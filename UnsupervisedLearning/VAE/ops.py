import tensorflow as tf


def conv3d(x, weight, bias, strides=2, padding=0):
    if padding == 0:
        conv = tf.nn.conv3d(x, weight, strides=[1, strides, strides, strides, 1], padding='SAME')
    else:
        conv = tf.nn.conv3d(x, W, strides=[1, strides, strides, strides, 1], padding='VALID')
    conv = tf.nn.bias_add(x, bias)
    return tf.nn.relu(conv)


def conv3d_transpose(x, weight, bias, outputShape, strides=2, padding=0, activation_fn=None):
    if padding == 0:
        deconv = tf.nn.conv3d_transpose(x, weight, output_shape=outputShape, strides=strides, padding="SAME")
    else:
        deconv = tf.nn.conv3d_transpose(x, weight, output_shape=outputShape, strides=strides, padding="VALID")
    deconv = tf.nn.bias_add(deconv, bias)
    if activation_fn == "sigmoid":
        return tf.nn.sigmoid(deconv)
    deconv = tf.nn.relu(deconv)
    return deconv
