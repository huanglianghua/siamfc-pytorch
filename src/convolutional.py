import tensorflow as tf


def set_convolutional(X, W, b, stride, bn_beta, bn_gamma, bn_mm, bn_mv, filtergroup=False, batchnorm=True,
                      activation=True, scope=None, reuse=False):
    # use the input scope or default to "conv"
    with tf.variable_scope(scope or 'conv', reuse=reuse):
        # sanity check    
        W = tf.get_variable("W", W.shape, trainable=False, initializer=tf.constant_initializer(W))
        b = tf.get_variable("b", b.shape, trainable=False, initializer=tf.constant_initializer(b))

        if filtergroup:
            X0, X1 = tf.split(X, 2, 3)
            W0, W1 = tf.split(W, 2, 3)
            h0 = tf.nn.conv2d(X0, W0, strides=[1, stride, stride, 1], padding='VALID')
            h1 = tf.nn.conv2d(X1, W1, strides=[1, stride, stride, 1], padding='VALID')
            h = tf.concat([h0, h1], 3) + b
        else:
            h = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='VALID') + b

        if batchnorm:
            h = tf.layers.batch_normalization(h, beta_initializer=tf.constant_initializer(bn_beta),
                                              gamma_initializer=tf.constant_initializer(bn_gamma),
                                              moving_mean_initializer=tf.constant_initializer(bn_mm),
                                              moving_variance_initializer=tf.constant_initializer(bn_mv),
                                              training=False, trainable=False)

        if activation:
            h = tf.nn.relu(h)

        return h
