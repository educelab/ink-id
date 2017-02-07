import tensorflow as tf
import pdb

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def conv3d(x, weight, bias, padding=0):
    if padding == 0:
        conv = tf.nn.convolution(x, weight, padding="SAME", dilation_rate=[3,3,3])
    else:
        conv = tf.nn.convolution(x, weight, padding="VALID", dilation_rate=[3,3,3])

    conv = tf.nn.bias_add(conv, bias)
    # conv = lrelu(conv)
    convForward = tf.nn.relu(conv)

    mean, variance = tf.nn.moments(convForward, [0,1,2,3])
    convForward = tf.nn.batch_normalization(convForward, mean, variance, None, None, 1e-5)

    convForward = tf.nn.pool(convForward, [3,3,3], pooling_type="AVG", padding="SAME", strides=[2,2,2])
    return conv, convForward


def conv3d_transpose(x, weight, bias, outputShape, strides=2, padding=0, activation_fn=None):
    if padding == 0:
        deconv = tf.nn.conv3d_transpose(x, weight, output_shape=outputShape, strides=[1,strides,strides,strides,1], padding="SAME")
    else:
        deconv = tf.nn.conv3d_transpose(x, weight, output_shape=outputShape, strides=[1,strides,strides,strides,1], padding="VALID")
    deconv = tf.nn.bias_add(deconv, bias)

    if activation_fn == "sigmoid":
        return deconv, tf.nn.sigmoid(deconv)

    deconvForward = tf.nn.relu(deconv)

    mean, variance = tf.nn.moments(deconvForward, [0,1,2,3])
    deconvForward = tf.nn.batch_normalization(deconvForward, mean, variance, None, None, 1e-5)

    return deconv, deconvForward
