import tensorflow as tf

# define the cnn model
def init_model(input_shape, n_classes):
    with tf.name_scope("InputScope") as scope:
        cnn_input = tf.placeholder(
            dtype=tf.float32, shape=(None,) + input_shape + (1,), name="input"
        )

    # base convolutional layers
    y_true = tf.placeholder(dtype=tf.float32, shape=(None, n_classes), name="y_true")

    scales = tf.placeholder(dtype=tf.float32, shape=(None, 3), name="scales")

    training = tf.placeholder_with_default(
        tf.constant(False, dtype=tf.bool), shape=(), name="training"
    )

    conv1 = tf.layers.conv3d(
        inputs=cnn_input,
        filters=32,
        kernel_size=(3, 3, 3),
        padding="same",
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name="conv1",
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
        name="conv3",
    )

    pool2 = tf.layers.max_pooling3d(
        inputs=conv2, pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool2"
    )

    conv3 = tf.layers.conv3d(
        inputs=pool2,
        filters=64,
        kernel_size=(3, 3, 3),
        padding="same",
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name="conv5",
    )

    flatten = tf.contrib.layers.flatten(inputs=conv3)
    flatten = tf.concat([flatten, scales], axis=1, name="flatten")

    # classification network
    dense1 = tf.layers.dense(
        inputs=flatten,
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
        bn_dense1, rate=0.5, training=training, name="dropout_dense1"
    )

    descriptor = tf.layers.dense(
        inputs=dropout_dense1,
        units=64,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        activation=tf.nn.relu,
        use_bias=True,
        name="descriptor",
    )

    bn_descriptor = tf.layers.batch_normalization(
        descriptor, training=training, name="bn_descriptor"
    )

    with tf.name_scope("OutputScope") as scope:
        tf.add(bn_descriptor, 0, name="descriptor_bn_read")
        tf.add(descriptor, 0, name="descriptor_read")

    dropout_descriptor = tf.layers.dropout(
        bn_descriptor, rate=0.35, training=training, name="dropout_descriptor"
    )

    y_pred = tf.layers.dense(
        inputs=dropout_descriptor,
        units=n_classes,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        activation=None,
        use_bias=True,
        name="classes",
    )

    loss_c = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true),
        name="loss_c",
    )

    # reconstruction network
    dec_dense1 = tf.layers.dense(
        inputs=descriptor,
        units=8192,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        activation=tf.nn.relu,
        use_bias=True,
        name="dec_dense1",
    )

    reshape = tf.reshape(dec_dense1, (tf.shape(cnn_input)[0], 8, 8, 4, 32))

    dec_conv1 = tf.layers.conv3d_transpose(
        inputs=reshape,
        filters=32,
        kernel_size=(3, 3, 3),
        strides=(2, 2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        activation=tf.nn.relu,
        name="dec_conv1",
    )

    dec_conv2 = tf.layers.conv3d_transpose(
        inputs=dec_conv1,
        filters=32,
        kernel_size=(3, 3, 3),
        strides=(2, 2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        activation=tf.nn.relu,
        name="dec_conv2",
    )

    dec_reshape = tf.layers.conv3d_transpose(
        inputs=dec_conv2,
        filters=1,
        kernel_size=(3, 3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        activation=tf.nn.sigmoid,
        name="dec_reshape",
    )

    reconstruction = dec_reshape
    with tf.name_scope("ReconstructionScopeAE") as scope:
        tf.add(reconstruction, 0, name="ae_reconstruction_read")

    FN_TO_FP_WEIGHT = 0.9
    loss_r = -tf.reduce_mean(
        FN_TO_FP_WEIGHT * cnn_input * tf.log(reconstruction + 1e-10)
        + (1 - FN_TO_FP_WEIGHT) * (1 - cnn_input) * tf.log(1 - reconstruction + 1e-10)
    )
    tf.identity(loss_r, "loss_r")

    # training
    LOSS_R_WEIGHT = 200
    LOSS_C_WEIGHT = 1
    loss = tf.add(LOSS_C_WEIGHT * loss_c, LOSS_R_WEIGHT * loss_r, name="loss")

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
    y_prob = tf.nn.softmax(y_pred, name="y_prob")

    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    roc_auc = tf.placeholder(dtype=tf.float32, shape=(), name="roc_auc")

    with tf.name_scope("summary"):
        tf.summary.scalar("loss", loss, collections=["summary_batch"])
        tf.summary.scalar("loss_c", loss_c, collections=["summary_batch"])
        tf.summary.scalar("loss_r", loss_r, collections=["summary_batch"])
        tf.summary.scalar("accuracy", accuracy, collections=["summary_batch"])

        tf.summary.scalar("roc_auc", roc_auc, collections=["summary_epoch"])
