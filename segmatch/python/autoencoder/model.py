import numpy as np
import tensorflow as tf

class ModelParams:
  def __init__(self):
    self.INPUT_SHAPE = [24,24,24,1]
    self.CONVOLUTION_LAYERS = [{'type': 'conv3d', 'filter': [5, 5, 5,  1,  10], 'downsampling': {'type': 'max_pool3d', 'k': 2}},
                               {'type': 'conv3d', 'filter': [5, 5, 5, 10, 100], 'downsampling': {'type': 'max_pool3d', 'k': 2}}]
    self.HIDDEN_LAYERS = [{'shape': [1000]}, {'shape': [600]}, {'shape': [400]}]
    self.LATENT_SHAPE = [15]
    self.COERCED_LATENT_DIMS = 1
    self.LEARNING_RATE = 0.0001
    self.CLIP_GRADIENTS = 0
    self.DROPOUT = 0.8 # Keep-prob
    self.FLOAT_TYPE = tf.float32
    self.DISABLE_SUMMARY = False
  def __str__(self):
    return str(self.__dict__)
  def __eq__(self, other): 
    return self.__dict__ == other.__dict__
  def __ne__(self, other):
    return not self.__eq__(other)

def n_dimensional_weightmul(L, W, L_shape, Lout_shape, first_dim_of_l_is_batch=True):
  """ Equivalent to matmul(W,L)
      but works for L with larger shapes than 1
      L_shape and Lout_shape are excluding the batch dimension (0)"""
  if not first_dim_of_l_is_batch:
    raise NotImplementedError
  if len(L_shape) == 1 and len(Lout_shape) == 1:
    return tf.matmul(L, W)
  # L    : ?xN1xN2xN3x...
  # Lout : ?xM1xM2xM3x...
  # W    : N1xN2x...xM1xM2x...
  # Einstein notation: letter b (denotes batch dimension)
  # Lout_blmn... = L_bijk... * Wijk...lmn...
  letters = list('ijklmnopqrst')
  l_subscripts = ''.join([letters.pop(0) for _ in range(len(L_shape))])
  lout_subscripts   = ''.join([letters.pop(0) for _ in range(len(Lout_shape))])
  einsum_string = 'b'+l_subscripts+','+l_subscripts+lout_subscripts+'->'+'b'+lout_subscripts
  return tf.einsum(einsum_string,L,W)

def gaussian_log_likelihood(self, sample, mean, log_sigma_squared):
    stddev = tf.exp(tf.sqrt(log_sigma_squared))
    epsilon = (sample - mean) / (stddev + 1e-10)
    return tf.reduce_sum(- 0.5 * np.log(2 * np.pi) - tf.log(stddev + 1e-10) - 0.5 * tf.square(epsilon), reduction_indices=1)

class Autoencoder(object):
  def __init__(self, model_params):
    self.MP = model_params

    tf.reset_default_graph()
    preset_batch_size = None
    self.variables = []
    # Graph input
    with tf.name_scope('Placeholders') as scope:
      self.input_placeholder = tf.placeholder(self.MP.FLOAT_TYPE,
                                              shape=[preset_batch_size] + self.MP.INPUT_SHAPE,
                                              name="input")
      if self.MP.DROPOUT is not None:
        default_dropout = tf.constant(1, dtype=self.MP.FLOAT_TYPE)
        self.dropout_placeholder = tf.placeholder_with_default(default_dropout, (), name="dropout_prob")
      self.twin_placeholder = None
    # Encoder
    previous_layer = self.input_placeholder
    previous_layer_shape = self.MP.INPUT_SHAPE # Excludes batch dim (which should be at pos 0)
    # Convolutional Layers
    self.conv_layers_input_shapes = []
    for i, LAYER in enumerate(self.MP.CONVOLUTION_LAYERS):
      self.conv_layers_input_shapes.append(previous_layer_shape)
      with tf.name_scope('ConvLayer'+str(i)) as scope:
        filter_shape = LAYER['filter']
        stride = LAYER['stride'] if 'stride' in LAYER else 1
        strides = [1, stride, stride, stride, 1]
        padding = LAYER['padding'] if 'padding' in LAYER else "SAME"
        with tf.variable_scope('ConvLayer'+str(i)+'Weights') as varscope:
          weights = tf.get_variable("weights_conv_"+str(i), dtype=self.MP.FLOAT_TYPE,
                                    shape=filter_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_conv_"+str(i) , dtype=self.MP.FLOAT_TYPE,
                                    shape=[filter_shape[-1]],
                                    initializer=tf.constant_initializer(0))
        self.variables.append(weights)
        self.variables.append(biases)
        layer_output = tf.nn.conv3d(previous_layer, weights, strides, padding)
        layer_output = tf.nn.bias_add(layer_output, biases)
        layer_shape = previous_layer_shape[:]
        for i, (prev_dim, filt_dim) in enumerate(zip(previous_layer_shape[:3], filter_shape[:3])):
          pad = np.floor(filt_dim/2) if padding == "SAME" else 0
          layer_shape[i] = int(((prev_dim + 2*pad - filt_dim)/stride)+1)
        layer_shape[-1] = filter_shape[-1]
        # Downsampling
        if 'downsampling' in LAYER:
          DOWNSAMPLING = LAYER['downsampling']
          if DOWNSAMPLING['type'] != 'max_pool3d': raise NotImplementedError
          if self.MP.FLOAT_TYPE != tf.float32: raise TypeError('max_pool3d only supports float32')
          k = DOWNSAMPLING['k'] if 'k' in DOWNSAMPLING else 2
          ksize   = [1, k, k, k, 1]
          strides = [1, k, k, k, 1]
          padding = DOWNSAMPLING['padding'] if 'padding' in DOWNSAMPLING else "VALID"
          layer_output = tf.nn.max_pool3d(layer_output, ksize, strides, padding)
          pad = np.floor(k/2) if padding == "SAME" else 0
          layer_shape = [int(((dim + 2*pad - k)/k)+1) for dim in layer_shape[:3]] + layer_shape[3:]
        # set up next loop
        previous_layer = layer_output
        previous_layer_shape = layer_shape
    # Flatten output
    self.shape_before_flattening = previous_layer_shape
    previous_layer_shape = [np.prod(previous_layer_shape)]
    previous_layer = tf.reshape(previous_layer, shape=[-1]+previous_layer_shape, name="flatten")
    # Fully connected Layers
    for i, LAYER in enumerate(self.MP.HIDDEN_LAYERS):
      with tf.name_scope('EncoderLayer'+str(i)) as scope:
        layer_shape = LAYER['shape']
        with tf.variable_scope('EncoderLayer'+str(i)+'Weights') as varscope:
          weights = tf.get_variable("weights_encoder_"+str(i), dtype=self.MP.FLOAT_TYPE,
                                    shape=previous_layer_shape + layer_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_encoder_"+str(i) , dtype=self.MP.FLOAT_TYPE,
                                    shape=layer_shape,
                                    initializer=tf.constant_initializer(0))
        self.variables.append(weights)
        self.variables.append(biases)
        self.variable_summaries(weights)
        layer_output = tf.nn.softplus(tf.add(n_dimensional_weightmul(previous_layer,
                                                                     weights,
                                                                     previous_layer_shape,
                                                                     layer_shape),
                                             biases),
                                      name='softplus')
        if self.MP.DROPOUT is not None:
          layer_output = tf.nn.dropout(layer_output, self.dropout_placeholder)
        # set up next loop
        previous_layer = layer_output
        previous_layer_shape = layer_shape
    # Latent space
    with tf.name_scope('ZMeanLayer') as scope:
        layer_shape = self.MP.LATENT_SHAPE
        with tf.variable_scope('LatentLayerWeights') as varscope:
          weights = tf.get_variable("weights_z_mean", dtype=self.MP.FLOAT_TYPE, 
                                    shape=previous_layer_shape + layer_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_z_mean" , dtype=self.MP.FLOAT_TYPE,
                                    shape=layer_shape,
                                    initializer=tf.constant_initializer(0))
        self.variables.append(weights)
        self.variables.append(biases)
        self.z_mean = tf.add(n_dimensional_weightmul(previous_layer,
                                                     weights,
                                                     previous_layer_shape,
                                                     layer_shape),
                             biases, name='softplus') # name should be 'add'
    with tf.name_scope('ZLogSigmaSquaredLayer') as scope:
        layer_shape = self.MP.LATENT_SHAPE
        with tf.variable_scope('LatentLayerWeights') as varscope:
          weights = tf.get_variable("weights_z_log_sig2", dtype=self.MP.FLOAT_TYPE, 
                                    shape=previous_layer_shape + layer_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_z_log_sig2" , dtype=self.MP.FLOAT_TYPE,
                                    shape=layer_shape,
                                    initializer=tf.constant_initializer(0))
        self.variables.append(weights)
        self.variables.append(biases)
        self.z_log_sigma_squared = tf.add(n_dimensional_weightmul(previous_layer,
                                                                  weights,
                                                                  previous_layer_shape,
                                                                  layer_shape),
                                          biases, name='softplus') # name should be 'add'
    # Sample Z values from Latent-space Estimate
    with tf.name_scope('SampleZValues') as scope:
        # sample = mean + sigma*epsilon
        epsilon = tf.random_normal(tf.shape(self.z_mean), 0, 1,
                                   dtype=self.MP.FLOAT_TYPE, name='randomnormal')
        self.z_sample = tf.add(self.z_mean,
                               tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_squared)), epsilon),
                               name='z_sample')
    # Decoder
    # Fully connected layers
    previous_layer = self.z_sample
    previous_layer_shape = self.MP.LATENT_SHAPE
    for i, LAYER in enumerate(self.MP.HIDDEN_LAYERS[::-1]):
      with tf.name_scope('DecoderLayer'+str(i)) as scope:
        layer_shape = LAYER['shape']
        with tf.variable_scope('DecoderLayer'+str(i)+'Weights') as varscope:
          weights = tf.get_variable("weights_decoder_"+str(i), dtype=self.MP.FLOAT_TYPE,
                                    shape=previous_layer_shape + layer_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_decoder_"+str(i) , dtype=self.MP.FLOAT_TYPE,
                                    shape=layer_shape,
                                    initializer=tf.constant_initializer(0))
        self.variables.append(weights)
        self.variables.append(biases)
        layer_output = tf.nn.softplus(tf.add(n_dimensional_weightmul(previous_layer,
                                                                     weights,
                                                                     previous_layer_shape,
                                                                     layer_shape),
                                             biases),
                                      name='softplus')
        self.variable_summaries(weights)
        if self.MP.DROPOUT is not None:
          layer_output = tf.nn.dropout(layer_output, self.dropout_placeholder)
        # set up next loop
        previous_layer = layer_output
        previous_layer_shape = layer_shape
    # Post fully-connected layer
    with tf.name_scope('Reconstruction') as scope:
      layer_shape = [np.prod(self.shape_before_flattening)]
      with tf.variable_scope('ReconstructionLayerWeights') as varscope:
        weights = tf.get_variable("weights_reconstruction", dtype=self.MP.FLOAT_TYPE, 
                                  shape=previous_layer_shape + layer_shape,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases  = tf.get_variable("biases_reconstruction" , dtype=self.MP.FLOAT_TYPE,
                                  shape=layer_shape,
                                  initializer=tf.constant_initializer(0))
      self.variables.append(weights)
      self.variables.append(biases)
      previous_layer = tf.nn.sigmoid(tf.add(n_dimensional_weightmul(previous_layer,
                                                                         weights,
                                                                         previous_layer_shape,
                                                                         layer_shape),
                                                 biases),
                                          name='sigmoid')
    # Unflatten output
    previous_layer = tf.reshape(previous_layer, shape=[-1]+self.shape_before_flattening, name="unflatten")
    previous_layer_shape = self.shape_before_flattening
    # Deconvolutional Layers
    deconv_layers_output_shapes = self.conv_layers_input_shapes[::-1]
    dynamic_batch_size = tf.shape(previous_layer)[0]
    for i, LAYER in enumerate(self.MP.CONVOLUTION_LAYERS[::-1]):
      with tf.name_scope('DeConvLayer'+str(i)) as scope:
        filter_shape = LAYER['filter'][:]
        stride = LAYER['downsampling']['k'] if 'downsampling' in LAYER else 1
        strides = [1, stride, stride, stride, 1]
        padding = LAYER['padding'] if 'padding' in LAYER else "SAME"
        with tf.variable_scope('DeConvLayer'+str(i)+'Weights') as varscope:
          weights = tf.get_variable("weights_deconv_"+str(i), dtype=self.MP.FLOAT_TYPE,
                                    shape=filter_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_deconv_"+str(i) , dtype=self.MP.FLOAT_TYPE,
                                    shape=[filter_shape[-2]],
                                    initializer=tf.constant_initializer(0))
        self.variables.append(weights)
        self.variables.append(biases)
        output_shape=[dynamic_batch_size]+deconv_layers_output_shapes[i]
        layer_output = tf.nn.conv3d_transpose(previous_layer, weights, output_shape=output_shape, strides=strides, padding=padding)
        layer_output = tf.nn.bias_add(layer_output, biases)
        layer_output = tf.nn.relu(layer_output)
        layer_shape = output_shape[1:]
        # set up next loop
        previous_layer = layer_output
        previous_layer_shape = layer_shape
    # Output (as probability of output being 1)
    self.output = tf.minimum(previous_layer, 1)
    # Loss
    with tf.name_scope('Loss') as scope:
      with tf.name_scope('ReconstructionLoss') as sub_scope:
        # Cross entropy loss of output probabilities vs. input certainties.
        reconstruction_loss = \
            -tf.reduce_sum(self.input_placeholder * tf.log(1e-10 + self.output, name="log1")
                           + (1-self.input_placeholder) * tf.log(1e-10 + (1 - self.output), name="log2"),
                           list(range(1,len(self.MP.INPUT_SHAPE)+1)))
      with tf.name_scope('LatentLoss') as sub_scope:
        # Kullback Leibler divergence between latent normal distribution and ideal.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_squared
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_squared),
                                           list(range(1,len(self.MP.LATENT_SHAPE)+1)))
      # Average sum of costs over batch.
      self.cost = tf.reduce_mean(reconstruction_loss + latent_loss, name="cost")
    # Optimizer (ADAM)
    with tf.name_scope('Optimizer') as scope:
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.MP.LEARNING_RATE).minimize(self.cost)
      if self.MP.CLIP_GRADIENTS > 0:
          raise NotImplementedError
    # Initialize session
    self.catch_nans = tf.add_check_numerics_ops()
    self.sess = tf.Session()
    self.merged = tf.summary.merge_all() if not self.MP.DISABLE_SUMMARY else tf.constant(0)
    tf.initialize_all_variables().run(session=self.sess)
    # Saver
    self.saver = tf.train.Saver(self.variables)

  def build_info_graph(self):
    preset_batch_size = None
    # Encoder
    previous_layer = self.output
    previous_layer_shape = self.MP.INPUT_SHAPE # Excludes batch dim (which should be at pos 0)
    # Convolutional Layers
    for i, LAYER in enumerate(self.MP.CONVOLUTION_LAYERS):
      with tf.name_scope('Q_ConvLayer'+str(i)) as scope:
        filter_shape = LAYER['filter']
        stride = LAYER['stride'] if 'stride' in LAYER else 1
        strides = [1, stride, stride, stride, 1]
        padding = LAYER['padding'] if 'padding' in LAYER else "SAME"
        with tf.variable_scope('ConvLayer'+str(i)+'Weights', reuse=True) as varscope:
          weights = tf.get_variable("weights_conv_"+str(i), dtype=self.MP.FLOAT_TYPE,
                                    shape=filter_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_conv_"+str(i) , dtype=self.MP.FLOAT_TYPE,
                                    shape=[filter_shape[-1]],
                                    initializer=tf.constant_initializer(0))
        layer_output = tf.nn.conv3d(previous_layer, weights, strides, padding)
        layer_output = tf.nn.bias_add(layer_output, biases)
        layer_shape = previous_layer_shape[:]
        for i, (prev_dim, filt_dim) in enumerate(zip(previous_layer_shape[:3], filter_shape[:3])):
          pad = np.floor(filt_dim/2) if padding == "SAME" else 0
          layer_shape[i] = int(((prev_dim + 2*pad - filt_dim)/stride)+1)
        layer_shape[-1] = filter_shape[-1]
        # Downsampling
        if 'downsampling' in LAYER:
          DOWNSAMPLING = LAYER['downsampling']
          if DOWNSAMPLING['type'] != 'max_pool3d': raise NotImplementedError
          if self.MP.FLOAT_TYPE != tf.float32: raise TypeError('max_pool3d only supports float32')
          k = DOWNSAMPLING['k'] if 'k' in DOWNSAMPLING else 2
          ksize   = [1, k, k, k, 1]
          strides = [1, k, k, k, 1]
          padding = DOWNSAMPLING['padding'] if 'padding' in DOWNSAMPLING else "VALID"
          layer_output = tf.nn.max_pool3d(layer_output, ksize, strides, padding)
          pad = np.floor(k/2) if padding == "SAME" else 0
          layer_shape = [int(((dim + 2*pad - k)/k)+1) for dim in layer_shape[:3]] + layer_shape[3:]
        # set up next loop
        previous_layer = layer_output
        previous_layer_shape = layer_shape
    # Flatten output
    previous_layer_shape = [np.prod(previous_layer_shape)]
    previous_layer = tf.reshape(previous_layer, shape=[-1]+previous_layer_shape, name="flatten")
    # Fully connected Layers
    for i, LAYER in enumerate(self.MP.HIDDEN_LAYERS):
      with tf.name_scope('Q_EncoderLayer'+str(i)) as scope:
        layer_shape = LAYER['shape']
        with tf.variable_scope('EncoderLayer'+str(i)+'Weights', reuse=True) as varscope:
          weights = tf.get_variable("weights_encoder_"+str(i), dtype=self.MP.FLOAT_TYPE,
                                    shape=previous_layer_shape + layer_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_encoder_"+str(i) , dtype=self.MP.FLOAT_TYPE,
                                    shape=layer_shape,
                                    initializer=tf.constant_initializer(0))
        layer_output = tf.nn.softplus(tf.add(n_dimensional_weightmul(previous_layer,
                                                                     weights,
                                                                     previous_layer_shape,
                                                                     layer_shape),
                                             biases),
                                      name='softplus')
        if self.MP.DROPOUT is not None:
          layer_output = tf.nn.dropout(layer_output, self.dropout_placeholder)
        # set up next loop
        previous_layer = layer_output
        previous_layer_shape = layer_shape
    # Latent space
    with tf.name_scope('Q_ZMeanLayer') as scope:
        layer_shape = self.MP.LATENT_SHAPE
        with tf.variable_scope('LatentLayerWeights', reuse=True) as varscope:
          weights = tf.get_variable("weights_z_mean", dtype=self.MP.FLOAT_TYPE, 
                                    shape=previous_layer_shape + layer_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_z_mean" , dtype=self.MP.FLOAT_TYPE,
                                    shape=layer_shape,
                                    initializer=tf.constant_initializer(0))
        self.q_z_mean = tf.add(n_dimensional_weightmul(previous_layer,
                                                       weights,
                                                       previous_layer_shape,
                                                       layer_shape),
                               biases)[:,:self.MP.COERCED_LATENT_DIMS]
    with tf.name_scope('Q_ZLogSigmaSquaredLayer') as scope:
        layer_shape = self.MP.LATENT_SHAPE
        with tf.variable_scope('LatentLayerWeights', reuse=True) as varscope:
          weights = tf.get_variable("weights_z_log_sig2", dtype=self.MP.FLOAT_TYPE, 
                                    shape=previous_layer_shape + layer_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_z_log_sig2" , dtype=self.MP.FLOAT_TYPE,
                                    shape=layer_shape,
                                    initializer=tf.constant_initializer(0))
        self.q_z_log_sigma_squared = tf.add(n_dimensional_weightmul(previous_layer,
                                                                    weights,
                                                                    previous_layer_shape,
                                                                    layer_shape),
                                            biases)[:,:self.MP.COERCED_LATENT_DIMS]
    # Loss
    with tf.name_scope('MutualInfo_Loss') as scope:
      c_sample = self.z_sample[:,self.MP.COERCED_LATENT_DIMS]
      c_mean_prior = self.z_mean[:,self.MP.COERCED_LATENT_DIMS]
      c_log_sigma_squared_prior = self.z_log_sigma_squared[:,self.MP.COERCED_LATENT_DIMS]
      self.mutual_information_loss = tf.reduce_mean(gaussian_log_likelihood(c_sample, self.q_z_mean, self.q_z_log_sigma_squared) -
                                                    gaussian_log_likelihood(c_sample, self.c_mean_prior, self.c_log_sigma_squared_prior))
      self.discriminator_loss -= self.info_reg_coeff * cont_mi_est
      self.generator_loss -= self.info_reg_coeff * cont_mi_est
      if not self.MP.DISABLE_SUMMARY:
          tf.summary.scalar('generator_loss_with_MI', self.generator_loss)
          tf.summary.scalar('discriminator_loss_with_MI', self.discriminator_loss)
          tf.summary.scalar('MI_loss', self.mutual_information_loss)
          self.merged = tf.summary.merge_all()
    with tf.name_scope('Adversarial_Optimizers_With_MI') as scope:
      self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.MP.LEARNING_RATE).minimize(self.generator_loss)
      self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.MP.LEARNING_RATE*0.1).minimize(self.discriminator_loss)
    tf.initialize_all_variables().run(session=self.sess)

  def build_discriminator_graph(self):
    # TODO: apply discriminator to both, loss = 0.5 Dlossreal + 0.5 Dlossfake
    preset_batch_size = None
    # Discriminator
    # y is a float of value either 0. or 1. representing wether the input is real (1. if real, 0. if fake)
    y = tf.cast(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32), MP.FLOAT_TYPE)
    previous_layer = self.input_placeholder * y + self.output * (1-y)
    previous_layer_shape = self.MP.INPUT_SHAPE # Excludes batch dim (which should be at pos 0)
    # Convolutional Layers
    for i, LAYER in enumerate(self.MP.CONVOLUTION_LAYERS):
      with tf.name_scope('Discriminator_ConvLayer'+str(i)) as scope:
        filter_shape = LAYER['filter']
        stride = LAYER['stride'] if 'stride' in LAYER else 1
        strides = [1, stride, stride, stride, 1]
        padding = LAYER['padding'] if 'padding' in LAYER else "SAME"
        with tf.variable_scope('ConvLayer'+str(i)+'Weights', reuse=True) as varscope:
          weights = tf.get_variable("weights_conv_"+str(i), dtype=self.MP.FLOAT_TYPE,
                                    shape=filter_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_conv_"+str(i) , dtype=self.MP.FLOAT_TYPE,
                                    shape=[filter_shape[-1]],
                                    initializer=tf.constant_initializer(0))
        layer_output = tf.nn.conv3d(previous_layer, weights, strides, padding)
        layer_output = tf.nn.bias_add(layer_output, biases)
        layer_shape = previous_layer_shape[:]
        for i, (prev_dim, filt_dim) in enumerate(zip(previous_layer_shape[:3], filter_shape[:3])):
          pad = np.floor(filt_dim/2) if padding == "SAME" else 0
          layer_shape[i] = int(((prev_dim + 2*pad - filt_dim)/stride)+1)
        layer_shape[-1] = filter_shape[-1]
        # Downsampling
        if 'downsampling' in LAYER:
          DOWNSAMPLING = LAYER['downsampling']
          if DOWNSAMPLING['type'] != 'max_pool3d': raise NotImplementedError
          if self.MP.FLOAT_TYPE != tf.float32: raise TypeError('max_pool3d only supports float32')
          k = DOWNSAMPLING['k'] if 'k' in DOWNSAMPLING else 2
          ksize   = [1, k, k, k, 1]
          strides = [1, k, k, k, 1]
          padding = DOWNSAMPLING['padding'] if 'padding' in DOWNSAMPLING else "VALID"
          layer_output = tf.nn.max_pool3d(layer_output, ksize, strides, padding)
          pad = np.floor(k/2) if padding == "SAME" else 0
          layer_shape = [int(((dim + 2*pad - k)/k)+1) for dim in layer_shape[:3]] + layer_shape[3:]
        # set up next loop
        previous_layer = layer_output
        previous_layer_shape = layer_shape
    # Flatten output
    previous_layer_shape = [np.prod(previous_layer_shape)]
    previous_layer = tf.reshape(previous_layer, shape=[-1]+previous_layer_shape, name="flatten")
    # Fully connected Layers
    for i, LAYER in enumerate(self.MP.HIDDEN_LAYERS):
      with tf.name_scope('Discriminator_EncoderLayer'+str(i)) as scope:
        layer_shape = LAYER['shape']
        with tf.variable_scope('DiscriminatorFCLayer'+str(i)+'Weights', reuse=False) as varscope:
          weights = tf.get_variable("weights_discriminator_fc_"+str(i), dtype=self.MP.FLOAT_TYPE,
                                    shape=previous_layer_shape + layer_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_discriminator_fc_"+str(i) , dtype=self.MP.FLOAT_TYPE,
                                    shape=layer_shape,
                                    initializer=tf.constant_initializer(0))
        layer_output = tf.nn.softplus(tf.add(n_dimensional_weightmul(previous_layer,
                                                                     weights,
                                                                     previous_layer_shape,
                                                                     layer_shape),
                                             biases),
                                      name='softplus')
        if self.MP.DROPOUT is not None:
          layer_output = tf.nn.dropout(layer_output, self.dropout_placeholder)
        # set up next loop
        previous_layer = layer_output
        previous_layer_shape = layer_shape
    # Latent space
    with tf.name_scope('Discriminator_output') as scope:
        layer_shape = [1]
        with tf.variable_scope('DiscriminatorOutputWeights', reuse=False) as varscope:
          weights = tf.get_variable("weights_discriminator_output", dtype=self.MP.FLOAT_TYPE, 
                                    shape=previous_layer_shape + layer_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_discriminator_output" , dtype=self.MP.FLOAT_TYPE,
                                    shape=layer_shape,
                                    initializer=tf.constant_initializer(0))
        self.discriminator_output = tf.add(n_dimensional_weightmul(previous_layer,
                                                                   weights,
                                                                   previous_layer_shape,
                                                                   layer_shape),
                                           biases)
        self.discriminator_output = tf.nn.softplus(self.discriminator_output)
    # Loss
    with tf.name_scope('Adversarial_Loss') as scope:
      self.discriminator_loss = tf.reduce_mean(y * -tf.log(self.discriminator_output + 1e-10) +
                                               (1.0 - y) * -tf.log(1.0 - self.discriminator_output + 1e-10))
      self.generator_loss = tf.reduce_mean((1.0 - y) * -tf.log(self.discriminator_output + 1e-10))
      if not self.MP.DISABLE_SUMMARY:
          tf.summary.scalar('generator_loss', self.generator_loss)
          tf.summary.scalar('discriminator_loss', self.discriminator_loss)
          self.merged = tf.summary.merge_all()
    # Optimizer (ADAM)
    with tf.name_scope('Adversarial_Optimizers') as scope:
      self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.MP.LEARNING_RATE).minimize(self.generator_loss)
      self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.MP.LEARNING_RATE*0.1).minimize(self.discriminator_loss)
    tf.initialize_all_variables().run(session=self.sess)

  def variable_summaries(self, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    if not DISABLE_SUMMARY:
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

  ## Example functions for different ways to call the autoencoder graph.
  def encode(self, batch_input):
    return self.sess.run((self.z_mean, self.z_log_sigma_squared),
                         feed_dict={self.input_placeholder: batch_input})
  def decode(self, batch_z):
    return self.sess.run(self.output,
                         feed_dict={self.z_sample: batch_z})
  def encode_decode(self, batch_input):
    return self.sess.run(self.output,
                         feed_dict={self.input_placeholder: batch_input})
  def train_on_single_batch(self, batch_input, adversarial=False, cost_only=False, dropout=None, summary_writer=None):
    # feed placeholders
    dict_ = {self.input_placeholder: batch_input}
    if self.MP.DROPOUT is not None:
      dict_[self.dropout_placeholder] = self.MP.DROPOUT if dropout is None else dropout
    else:
      if dropout is not None: raise ValueError('This model does not implement dropout yet a value was specified')
    # Graph nodes to target
    cost = [self.cost]
    if adversarial: cost = cost + [self.generator_loss, self.discriminator_loss]
    opt = [self.optimizer]
    if adversarial: opt = opt + [self.generator_optimizer, self.discriminator_optimizer]
    # compute
    if cost_only:
      cost = self.sess.run(cost,
                           feed_dict=dict_)
    else:
      cost, _, _, summary = self.sess.run((cost, opt, self.catch_nans, self.merged),
                                          feed_dict=dict_)
      if summary_writer is not None: summary_writer.add_summary(summary)
    return sum(cost)
  def cost_on_single_batch(self, batch_input, adversarial=False):
    return self.train_on_single_batch(batch_input, adversarial=adversarial, cost_only=True, dropout=1.0)

  def batch_encode(self, batch_input, batch_size=200, verbose=True):
    return batch_generic_func(self.encode, batch_input, batch_size, verbose)
  def batch_decode(self, batch_input, batch_size=200, verbose=True):
    return batch_generic_func(self.decode, batch_input, batch_size, verbose)
  def batch_encode_decode(self, batch_input, batch_size=100, verbose=True):
    return batch_generic_func(self.encode_decode, batch_input, batch_size, verbose)

def concat(a, b):
  if isinstance(a, list):
    return a+b
  elif isinstance(b, np.ndarray):
    return np.concatenate((a,b), axis=0)
  else:
    raise TypeError('Inputs are of unsupported type')

def batch_add(A, B):
  if isinstance(A, tuple):
    return tuple([concat(a, b) for a, b in zip(A, B)])
  else:
    return concat(A, B)

def batch_generic_func(function, batch_input, batch_size=100, verbose=False):
  a = 0
  b = batch_size
  while a < len(batch_input):
    batch_output = function(batch_input[a:b])
    try: output = batch_add(output, batch_output)
    except(NameError): output = batch_output
    a += batch_size
    b += batch_size
    if verbose: print("Example " + str(min(a,len(batch_input))) + "/" + str(len(batch_input)))
  return output

