import tensorflow as tf

from inception_v3 import inception_v3, inception_v3_base
slim = tf.contrib.slim

sess = tf.InteractiveSession()

weights_regularizer = None
trainable = False
is_inception_model_training = False
batch_norm_params = {
    "is_training": is_inception_model_training,
    "trainable": trainable,
    # Decay for the moving averages.
    "decay": 0.9997,
    # Epsilon to prevent 0s in variance.
    "epsilon": 0.001,
    # Collection containing the moving mean and moving variance.
    "variables_collections": {
        "beta": None,
        "gamma": None,
        "moving_mean": ["moving_vars"],
        "moving_variance": ["moving_vars"],
    }
}
stddev = 0.01
dropout_keep_prob = 0.8

images = tf.placeholder(tf.float32, [None, 299, 299, 3])


with tf.variable_scope("InceptionV3") as scope:
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer,
            trainable=trainable):
        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params):
            net, end_points = inception_v3_base(images, scope=scope)
            with tf.variable_scope("logits"):
                shape = net.get_shape()
                net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                net = slim.dropout(
                    net,
                    keep_prob=dropout_keep_prob,
                    is_training=is_inception_model_training,
                    scope="dropout")
                net = slim.flatten(net, scope="flatten")

writer = tf.train.SummaryWriter('./log', sess.graph)
