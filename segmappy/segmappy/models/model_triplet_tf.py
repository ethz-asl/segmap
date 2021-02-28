import tensorflow as tf

# define the cnn model
def init_model(input_shape, margin=0.1):
    with tf.name_scope("InputScope") as scope:
        cnn_input = tf.placeholder(
            #dtype=tf.float32, shape=(None,) + input_shape + (3 + 35,), name="input"
            dtype=tf.float32, shape=(None,) + input_shape + (1,), name="input")

    positive_input = tf.placeholder(
        dtype=tf.float32, shape=(None,) + input_shape + (1,), name="positive")
    negative_input = tf.placeholder(
        dtype=tf.float32, shape=(None,) + input_shape + (1,), name="negative")

    cnn_scales = tf.placeholder(dtype=tf.float32, shape=(None, 3), name="scales")
    positive_scales = tf.placeholder(
        dtype=tf.float32, shape=(None, 3), name="positive_scales")
    negative_scales = tf.placeholder(
        dtype=tf.float32, shape=(None, 3), name="negative_scales")


    training = tf.placeholder_with_default(
        tf.constant(False, dtype=tf.bool), shape=(), name="training"
    )

    inputs = [cnn_input, positive_input, negative_input]
    scales = [cnn_scales, positive_scales, negative_scales]

    # base convolutional layers
    outputs = []
    with tf.variable_scope("network_weights", reuse=tf.AUTO_REUSE):
        for i, input in enumerate(inputs):
            conv1 = tf.layers.conv3d(
                inputs=input,
                filters=32,
                kernel_size=(3, 3, 3),
                padding="same",
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="conv1",
            )

            conv1_1 = tf.layers.conv3d(
                inputs=conv1,
                filters=32,
                kernel_size=(3, 3, 3),
                padding="same",
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="conv1_1",
            )

            pool1 = tf.layers.max_pooling3d(
                inputs=conv1, pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool1"
            )

            conv2 = tf.layers.conv3d(
                inputs=pool1,
                filters=64,
                kernel_size=(3, 3, 3),
                padding="same",
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="conv2",
            )

            conv2_2 = tf.layers.conv3d(
                inputs=conv2,
                filters=64,
                kernel_size=(3, 3, 3),
                padding="same",
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="conv2_2",
            )

            pool2 = tf.layers.max_pooling3d(
                inputs=conv2_2, pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool2"
            )

            conv3 = tf.layers.conv3d(
                inputs=pool2,
                filters=64,
                kernel_size=(3, 3, 3),
                padding="same",
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="conv3",
            )

            flatten = tf.contrib.layers.flatten(inputs=conv3)

            dense_scales = tf.layers.dense(
                inputs=scales[i],
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                use_bias=True,
                name="dense_scales",
            )

            flatten = tf.concat([flatten, dense_scales], axis=1, name="flatten")

            bn_flatten = tf.layers.batch_normalization(
                flatten, training=training, name="bn_flatten"
            )

            dropout_flatten = tf.layers.dropout(
                bn_flatten, rate=0.2, training=training, name="dropout_flatten"
            )

            # classification network
            dense1 = tf.layers.dense(
                inputs=dropout_flatten,
                units=512,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                use_bias=True,
                name="dense1",
            )

            bn_dense1 = tf.layers.batch_normalization(
                dense1, training=training, name="bn_dense1"
            )

            dropout_dense1 = tf.layers.dropout(
                bn_dense1, rate=0.2, training=training, name="dropout_dense1"
            )

            descriptor = tf.layers.dense(
                inputs=dropout_dense1,
                units=64,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                activation=tf.nn.relu,
                use_bias=True,
                name="descriptor",
            )

            outputs.append(descriptor)

    output_cnn, output_positive, output_negative = outputs

    with tf.name_scope("OutputScope") as scope:
        tf.identity(output_cnn, name="descriptor_read")

    # training
    dist_positive = tf.sqrt(tf.reduce_sum(tf.square(output_cnn - output_positive), 1) + 1e-16)
    dist_negative = tf.sqrt(tf.reduce_sum(tf.square(output_cnn - output_negative), 1) + 1e-16)

    loss = tf.reduce_mean(tf.maximum(margin + dist_positive - dist_negative,
        0.), name='loss')

    global_step = tf.Variable(0, trainable=False, name="global_step")
    update_step = tf.assign(
        global_step, tf.add(global_step, tf.constant(1)), name="update_step"
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

    # add batch normalization updates to the training operation
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, name="train_op")

    # statistics
    means = []
    top75 = []
    for i in range(10):
        means.append(tf.placeholder(
            dtype=tf.float32, shape=(), name="means_" + str(i)))
        top75.append(tf.placeholder(
            dtype=tf.float32, shape=(), name="top75_" + str(i)))

    with tf.name_scope("summary"):
        tf.summary.scalar("loss", loss, collections=["summary_batch"])

    with tf.name_scope("summary_test"):
        for i in range(10):
            tf.summary.scalar(
                "means_" + str(i + 1), means[i], collections=["summary_epoch"])
            tf.summary.scalar(
                "top75_" + str(i + 1), top75[i], collections=["summary_epoch"])
