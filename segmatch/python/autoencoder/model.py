import numpy as np
import tensorflow as tf

class ModelParams:
  def __init__(self):
    self.INPUT_SHAPE = [24,24,24,1]
    self.CONVOLUTION_LAYERS = [{'type': 'conv3d', 'filter': [5, 5, 5,   1,  64], 'downsampling': {'type': 'max_pool3d', 'k': 2}},
                               {'type': 'conv3d', 'filter': [3, 3, 3,  64, 128], 'downsampling': {'type': 'max_pool3d', 'k': 2}},
                               {'type': 'conv3d', 'filter': [3, 3, 3, 128, 256], 'downsampling': {'type': 'max_pool3d', 'k': 2}}]
    self.HIDDEN_LAYERS = [{'shape': [1000]}, {'shape': [600]}, {'shape': [400]}]
    self.LATENT_SHAPE = [100]
    self.COERCED_LATENT_DIMS = 10
    self.LEARNING_RATE = 0.00001
    self.DROPOUT = 0.8 # Keep-prob
    self.FLOAT_TYPE = tf.float32
    self.DISABLE_SUMMARY = False
    self.ADVERSARIAL = True
    self.MUTUAL_INFO = True
    self.INFO_REG_COEFF = 0.5
  def __str__(self):
    return str(self.__dict__)
  def __eq__(self, other): 
    return self.__dict__ == other.__dict__
  def __ne__(self, other):
    return not self.__eq__(other)

TINY = 1e-8

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

def gaussian_log_likelihood(sample, mean, log_sigma_squared):
    stddev = tf.sqrt(tf.exp(log_sigma_squared))
    epsilon = (sample - mean) / (stddev + TINY)
    return tf.reduce_sum(- 0.5 * np.log(2 * np.pi) - tf.log(stddev + TINY) - 0.5 * tf.square(epsilon), reduction_indices=1)

class Autoencoder(object):
  def __init__(self, model_params):
    self.MP = model_params

    tf.reset_default_graph()
    preset_batch_size = None
    self.variables = []
    self.G_variables = []
    self.Q_only_variables = []
    self.Q_and_D_variables = []
    self.D_only_variables = []
    self.zero = tf.constant(0)
    # Graph input
    with tf.name_scope('Placeholders') as scope:
      self.input_placeholder = tf.placeholder(self.MP.FLOAT_TYPE,
                                              shape=[preset_batch_size] + self.MP.INPUT_SHAPE,
                                              name="input")
      if self.MP.DROPOUT is not None:
        default_dropout = tf.constant(1, dtype=self.MP.FLOAT_TYPE)
        self.dropout_placeholder = tf.placeholder_with_default(default_dropout, (), name="dropout_prob")
      self.stop_gradient_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool),
                                                                    (), name="stop_gradient_at_z")
    # Encoder
    previous_layer = self.input_placeholder
    previous_layer_shape = self.MP.INPUT_SHAPE # Excludes batch dim (which should be at pos 0)
    # Convolutional Layers
    self.conv_layers_input_shapes = []
    for i, LAYER in enumerate(self.MP.CONVOLUTION_LAYERS):
      self.conv_layers_input_shapes.append(previous_layer_shape)
      previous_layer, previous_layer_shape = self.build_conv_layer(LAYER, previous_layer, previous_layer_shape,
                                                                   'ConvLayer'+str(i), 'ConvLayer'+str(i)+'Weights', '_conv_'+str(i),
                                                                   put_variables_in_list=self.Q_and_D_variables)
    # Flatten output
    self.shape_before_flattening = previous_layer_shape
    previous_layer_shape = [np.prod(previous_layer_shape)]
    previous_layer = tf.reshape(previous_layer, shape=[-1]+previous_layer_shape, name="flatten")
    # Fully connected Layers
    for i, LAYER in enumerate(self.MP.HIDDEN_LAYERS):
      previous_layer, previous_layer_shape = self.build_FC_layer(LAYER, previous_layer, previous_layer_shape,
                                                                 'EncoderLayer'+str(i), 'EncoderLayer'+str(i)+'Weights', '_encoder_'+str(i),
                                                                 put_variables_in_list=self.Q_only_variables)
    # Latent space
    self.z_mean, _ = self.build_FC_layer({'shape': self.MP.LATENT_SHAPE}, previous_layer, previous_layer_shape,
                                         'ZMeanLayer', 'LatentLayerWeights', '_z_mean', activation=lambda x: x,
                                         put_variables_in_list=self.Q_only_variables)
    self.z_log_sigma_squared, _ = self.build_FC_layer({'shape': self.MP.LATENT_SHAPE}, previous_layer, previous_layer_shape,
                                                      'ZLogSigmaSquaredLayer', 'LatentLayerWeights', '_z_log_sig2', activation=lambda x: x,
                                                      put_variables_in_list=self.Q_only_variables)
    # Sample Z values from Latent-space Estimate
    with tf.name_scope('SampleZValues') as scope:
        # sample = mean + sigma*epsilon
        epsilon = tf.random_normal(tf.shape(self.z_mean), 0, 1,
                                   dtype=self.MP.FLOAT_TYPE, name='randomnormal')
        self.z_sample = tf.add(self.z_mean,
                               tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_squared)), epsilon),
                               name='z_sample')
    # Stop gradients if desired
    previous_layer = tf.cond(self.stop_gradient_placeholder,
                             lambda: tf.stop_gradient(self.z_sample),
                             lambda: self.z_sample)
    # Decoder (a.k.a Generator)
    # Fully connected layers
    previous_layer = previous_layer
    previous_layer_shape = self.MP.LATENT_SHAPE
    for i, LAYER in enumerate(self.MP.HIDDEN_LAYERS[::-1]):
      previous_layer, previous_layer_shape = self.build_FC_layer(LAYER, previous_layer, previous_layer_shape,
                                                                 'DecoderLayer'+str(i), 'DecoderLayer'+str(i)+'Weights', '_decoder_'+str(i),
                                                                 put_variables_in_list=self.G_variables)
    # Post fully-connected layer
    previous_layer, previous_layer_shape = self.build_FC_layer({'shape': [np.prod(self.shape_before_flattening)]}, 
                                                               previous_layer, previous_layer_shape,
                                                               'Reconstruction', 'ReconstructionLayerWeights', '_reconstruction',
                                                               activation=tf.nn.sigmoid, put_variables_in_list=self.G_variables)
    # Unflatten output
    previous_layer = tf.reshape(previous_layer, shape=[-1]+self.shape_before_flattening, name="unflatten")
    previous_layer_shape = self.shape_before_flattening
    # Deconvolutional Layers
    deconv_layers_output_shapes = self.conv_layers_input_shapes[::-1]
    dynamic_batch_size = tf.shape(previous_layer)[0]
    for i, (LAYER, target_output_shape) in enumerate(zip(self.MP.CONVOLUTION_LAYERS[::-1],
                                                         deconv_layers_output_shapes)):
      previous_layer, previous_layer_shape = self.build_deconv_layer(LAYER, previous_layer, previous_layer_shape,
                                                                     target_output_shape, dynamic_batch_size,
                                                                     'DeConvLayer'+str(i), 'DeConvLayer'+str(i)+'Weights', '_deconv_'+str(i),
                                                                     put_variables_in_list=self.G_variables)
    # Output (as probability of output being 1)
    self.output = tf.minimum(previous_layer, 1)
    # Loss
    with tf.name_scope('Losses') as scope:
      with tf.name_scope('ReconstructionLoss') as sub_scope:
        # Cross entropy loss of output probabilities vs. input certainties.
        self.reconstruction_loss = \
            -tf.reduce_mean(self.input_placeholder * tf.log(TINY + self.output, name="log1")
                           + (1-self.input_placeholder) * tf.log(TINY + (1 - self.output), name="log2"))
      with tf.name_scope('LatentLoss') as sub_scope:
        # Kullback Leibler divergence between latent normal distribution and ideal.
        self.latent_loss = -0.5 * 0.001 * tf.reduce_mean(1 + self.z_log_sigma_squared
                                                         - tf.square(self.z_mean)
                                                         - tf.exp(self.z_log_sigma_squared))
      # Average sum of costs over batch.
      self.cost = self.reconstruction_loss + self.latent_loss
      self.cost_no_MI = self.cost * 1.0
      if not self.MP.DISABLE_SUMMARY:
          tf.summary.scalar('self.reconstruction_loss', self.reconstruction_loss)
          tf.summary.scalar('self.latent_loss', self.latent_loss)
          tf.summary.scalar('autoencoder_loss', self.cost_no_MI)
    # Extra graph
    if self.MP.ADVERSARIAL: self.build_adversarial_graph()
    if self.MP.MUTUAL_INFO: self.build_mutual_info_graph()
    # Optimizers (ADAM)
    with tf.name_scope('Optimizer') as scope:
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.MP.LEARNING_RATE).minimize(self.cost)
    if self.MP.ADVERSARIAL:
      with tf.name_scope('Adversarial_Optimizers_With_MI') as scope:
        with tf.name_scope('Generator_Optimizer') as sub_scope:
          self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.MP.LEARNING_RATE).minimize(self.generator_loss,
                  var_list=self.G_variables+self.Q_only_variables+self.Q_and_D_variables)
        with tf.name_scope('Discriminator_Optimizer') as sub_scope:
          self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.MP.LEARNING_RATE).minimize(self.discriminator_loss,
                  var_list=self.D_only_variables+self.Q_and_D_variables)
    # Initialize session
    self.catch_nans = tf.add_check_numerics_ops()
    self.sess = tf.Session()
    self.merged = tf.summary.merge_all() if not self.MP.DISABLE_SUMMARY else self.zero
    tf.initialize_all_variables().run(session=self.sess)
    # Saver
    self.saver = tf.train.Saver(self.variables)
    tf.get_default_graph().finalize()


  def build_adversarial_graph(self):
    print("Building adversarial graph.")
    with tf.name_scope('Adversarial') as meta_scope:
      ## Apply Discriminator to real x
      previous_layer = self.input_placeholder
      previous_layer_shape = self.MP.INPUT_SHAPE
      # Convolutional Layers
      for i, LAYER in enumerate(self.MP.CONVOLUTION_LAYERS):
        previous_layer, previous_layer_shape = self.build_conv_layer(LAYER, previous_layer, previous_layer_shape,
                                                                     'Discriminator_ConvLayer'+str(i), 'ConvLayer'+str(i)+'Weights', '_conv_'+str(i),
                                                                     reuse=True)
      # Flatten output
      previous_layer_shape = [np.prod(previous_layer_shape)]
      previous_layer = tf.reshape(previous_layer, shape=[-1]+previous_layer_shape, name="flatten")
      # Fully connected Layers
      for i, LAYER in enumerate(self.MP.HIDDEN_LAYERS):
        previous_layer, previous_layer_shape = self.build_FC_layer(LAYER, previous_layer, previous_layer_shape,
                                                                   'Discriminator_EncoderLayer'+str(i), 'DiscriminatorFCLayer'+str(i)+'Weights', '_discriminator_fc_'+str(i),
                                                                   reuse=False, put_variables_in_list=self.D_only_variables)
      # Latent space
      self.discriminator_output_real, _ = self.build_FC_layer({'shape': [1]}, previous_layer, previous_layer_shape,
                                                              'Discriminator_output', 'DiscriminatorOutputWeights',
                                                              reuse=False, activation=tf.nn.sigmoid,
                                                              put_variables_in_list=self.D_only_variables)
      ## Apply Discriminator to generator output
      previous_layer = self.output
      previous_layer_shape = self.MP.INPUT_SHAPE
      # Convolutional Layers
      for i, LAYER in enumerate(self.MP.CONVOLUTION_LAYERS):
        previous_layer, previous_layer_shape = self.build_conv_layer(LAYER, previous_layer, previous_layer_shape,
                                                                     'Discriminator_ConvLayer'+str(i), 'ConvLayer'+str(i)+'Weights', '_conv_'+str(i),
                                                                     reuse=True)
      # Flatten output
      previous_layer_shape = [np.prod(previous_layer_shape)]
      previous_layer = tf.reshape(previous_layer, shape=[-1]+previous_layer_shape, name="flatten")
      # Fully connected Layers
      for i, LAYER in enumerate(self.MP.HIDDEN_LAYERS):
        previous_layer, previous_layer_shape = self.build_FC_layer(LAYER, previous_layer, previous_layer_shape,
                                                                   'Discriminator_EncoderLayer'+str(i), 'DiscriminatorFCLayer'+str(i)+'Weights', '_discriminator_fc_'+str(i),
                                                                   reuse=True)
      # Latent space
      self.discriminator_output_fake, _ = self.build_FC_layer({'shape': [1]}, previous_layer, previous_layer_shape,
                                                              'Discriminator_output', 'DiscriminatorOutputWeights',
                                                              reuse=True, activation=tf.nn.sigmoid)
      # Loss
      with tf.name_scope('Adversarial_Loss') as scope:
        self.discriminator_loss = tf.reduce_mean(0.5 * -tf.log(self.discriminator_output_real + TINY) +
                                                 0.5 * -tf.log(tf.maximum((1.0 - self.discriminator_output_fake), 0.) + TINY, name="logDfake"))
        self.generator_loss = tf.reduce_mean(-tf.log(self.discriminator_output_fake + TINY))
        self.discriminator_loss_no_MI = self.discriminator_loss * 1.
        self.generator_loss_no_MI = self.generator_loss * 1.
    with tf.name_scope('Losses') as meta_scope:
        if not self.MP.DISABLE_SUMMARY:
            tf.summary.scalar('generator_loss', self.generator_loss_no_MI)
            tf.summary.scalar('discriminator_loss', self.discriminator_loss_no_MI)

  def build_mutual_info_graph(self):
    print("Building mutual information graph.")
    with tf.name_scope('Mutual_Info') as meta_scope:
      # Encoder
      previous_layer = self.output
      previous_layer_shape = self.MP.INPUT_SHAPE # Excludes batch dim (which should be at pos 0)
      # Convolutional Layers
      for i, LAYER in enumerate(self.MP.CONVOLUTION_LAYERS):
        previous_layer, previous_layer_shape = self.build_conv_layer(LAYER, previous_layer, previous_layer_shape,
                                                                     'Q_ConvLayer'+str(i), 'ConvLayer'+str(i)+'Weights', '_conv_'+str(i),
                                                                     reuse=True)
      # Flatten output
      previous_layer_shape = [np.prod(previous_layer_shape)]
      previous_layer = tf.reshape(previous_layer, shape=[-1]+previous_layer_shape, name="flatten")
      # Fully connected Layers
      for i, LAYER in enumerate(self.MP.HIDDEN_LAYERS):
        previous_layer, previous_layer_shape = self.build_FC_layer(LAYER, previous_layer, previous_layer_shape,
                                                                   'Q_EncoderLayer'+str(i), 'EncoderLayer'+str(i)+'Weights', '_encoder_'+str(i),
                                                                   reuse=True)
      # Latent space
      self.q_z_mean, _ = self.build_FC_layer({'shape': self.MP.LATENT_SHAPE}, previous_layer, previous_layer_shape,
                                             'Q_ZMeanLayer', 'LatentLayerWeights', '_z_mean',
                                             activation=lambda x: x, reuse=True)
      self.q_z_log_sigma_squared, _ = self.build_FC_layer({'shape': self.MP.LATENT_SHAPE}, previous_layer, previous_layer_shape,
                                                          'Q_ZLogSigmaSquaredLayer', 'LatentLayerWeights', '_z_log_sig2',
                                                          activation=lambda x: x, reuse=True)
      self.q_z_mean = self.q_z_mean[:,:self.MP.COERCED_LATENT_DIMS]
      self.q_z_log_sigma_squared = self.q_z_log_sigma_squared[:,:self.MP.COERCED_LATENT_DIMS]
      # Loss
      with tf.name_scope('MutualInfo_Loss') as scope:
        c_sample = self.z_sample[:,:self.MP.COERCED_LATENT_DIMS]
        c_mean_prior = self.z_mean[:,:self.MP.COERCED_LATENT_DIMS]
        c_log_sigma_squared_prior = self.z_log_sigma_squared[:,:self.MP.COERCED_LATENT_DIMS]
        log_li_q_c_given_x = gaussian_log_likelihood(c_sample, self.q_z_mean, self.q_z_log_sigma_squared)
        log_li_q_c = gaussian_log_likelihood(c_sample, c_mean_prior, c_log_sigma_squared_prior)
        self.mutual_information_est = tf.reduce_mean(-log_li_q_c) - tf.reduce_mean(-log_li_q_c_given_x)
        self.discriminator_loss -= self.MP.INFO_REG_COEFF * self.mutual_information_est
        self.generator_loss -= self.MP.INFO_REG_COEFF * self.mutual_information_est
        self.cost -= self.MP.INFO_REG_COEFF * self.mutual_information_est
    with tf.name_scope('Losses') as meta_scope:
        if not self.MP.DISABLE_SUMMARY:
            tf.summary.scalar('generator_loss_with_MI', self.generator_loss)
            tf.summary.scalar('discriminator_loss_with_MI', self.discriminator_loss)
            tf.summary.scalar('autoencoder_loss_with_MI', self.cost)
            tf.summary.scalar('MI', self.mutual_information_est)

  def variable_summaries(self, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    if not self.MP.DISABLE_SUMMARY:
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

  def build_conv_layer(self, LAYER, previous_layer, previous_layer_shape, scope_name, varscope_name,
                       var_suffix='', reuse=False, put_variables_in_list=None):
    with tf.name_scope(scope_name) as scope:
      filter_shape = LAYER['filter']
      stride = LAYER['stride'] if 'stride' in LAYER else 1
      strides = [1, stride, stride, stride, 1]
      padding = LAYER['padding'] if 'padding' in LAYER else "SAME"
      with tf.variable_scope(varscope_name, reuse=reuse) as varscope:
        weights = tf.get_variable("weights"+var_suffix, dtype=self.MP.FLOAT_TYPE,
                                  shape=filter_shape,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases  = tf.get_variable("biases_conv"+var_suffix, dtype=self.MP.FLOAT_TYPE,
                                  shape=[filter_shape[-1]],
                                  initializer=tf.constant_initializer(0))
      if not reuse:
        self.variables.append(weights)
        self.variables.append(biases)
        if put_variables_in_list is not None: 
            put_variables_in_list.append(weights)
            put_variables_in_list.append(biases)
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
    return layer_output, layer_shape

  def build_FC_layer(self, LAYER, previous_layer, previous_layer_shape, scope_name, varscope_name, var_suffix='',
                     reuse=False, activation=tf.nn.softplus, put_variables_in_list=None):
    with tf.name_scope(scope_name) as scope:
      layer_shape = LAYER['shape']
      with tf.variable_scope(varscope_name, reuse=reuse) as varscope:
        weights = tf.get_variable("weights"+var_suffix, dtype=self.MP.FLOAT_TYPE,
                                  shape=previous_layer_shape + layer_shape,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases  = tf.get_variable("biases"+var_suffix, dtype=self.MP.FLOAT_TYPE,
                                  shape=layer_shape,
                                  initializer=tf.constant_initializer(0))
      if not reuse:
        self.variables.append(weights)
        self.variables.append(biases)
        if put_variables_in_list is not None: 
            put_variables_in_list.append(weights)
            put_variables_in_list.append(biases)
        self.variable_summaries(weights)
      layer_output = tf.add(n_dimensional_weightmul(previous_layer,
                                                    weights,
                                                    previous_layer_shape,
                                                    layer_shape), biases)
      layer_output = activation(layer_output)
      if self.MP.DROPOUT is not None:
        layer_output = tf.nn.dropout(layer_output, self.dropout_placeholder)
    return layer_output, layer_shape

  def build_deconv_layer(self, LAYER, previous_layer, previous_layer_shape, target_output_shape, dynamic_batch_size,
                         scope_name, varscope_name, var_suffix='', reuse=False, put_variables_in_list=None):
    with tf.name_scope(scope_name) as scope:
      filter_shape = LAYER['filter'][:]
      stride = LAYER['downsampling']['k'] if 'downsampling' in LAYER else 1
      strides = [1, stride, stride, stride, 1]
      padding = LAYER['padding'] if 'padding' in LAYER else "SAME"
      with tf.variable_scope(varscope_name, reuse=reuse) as varscope:
        weights = tf.get_variable("weights"+var_suffix, dtype=self.MP.FLOAT_TYPE,
                                  shape=filter_shape,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases  = tf.get_variable("biases"+var_suffix, dtype=self.MP.FLOAT_TYPE,
                                  shape=[filter_shape[-2]],
                                  initializer=tf.constant_initializer(0))
      if not reuse:
        self.variables.append(weights)
        self.variables.append(biases)
        if put_variables_in_list is not None: 
            put_variables_in_list.append(weights)
            put_variables_in_list.append(biases)
      output_shape=[dynamic_batch_size]+target_output_shape
      layer_output = tf.nn.conv3d_transpose(previous_layer, weights, output_shape=output_shape, strides=strides, padding=padding)
      layer_output = tf.nn.bias_add(layer_output, biases)
      layer_output = tf.nn.relu(layer_output)
      layer_shape = output_shape[1:]
    return layer_output, layer_shape

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
  def train_on_single_batch(self, batch_input, train_target=None, cost_only=False, dropout=None, summary_writer=None):
    # feed placeholders
    dict_ = {self.input_placeholder: batch_input}
    if self.MP.DROPOUT is not None:
      dict_[self.dropout_placeholder] = self.MP.DROPOUT if dropout is None else dropout
    else:
      if dropout is not None: raise ValueError('This model does not implement dropout yet a value was specified')
    # Graph nodes to target
    cost = [self.cost]
    if self.MP.ADVERSARIAL: cost = cost + [self.generator_loss_no_MI, self.discriminator_loss_no_MI, self.mutual_information_est]
    opt = train_target if train_target is not None else self.optimizer
    if self.MP.ADVERSARIAL:
      if opt is self.discriminator_optimizer or opt is self.generator_optimizer: dict_[self.stop_gradient_placeholder] = True
    # compute
    cost, _, _, summary = self.sess.run((cost, opt, self.catch_nans, self.merged), feed_dict=dict_)
    if summary_writer is not None: summary_writer.add_summary(summary)
    return np.array(cost)
  def cost_on_single_batch(self, batch_input, summary_writer=None):
    return self.train_on_single_batch(batch_input, train_target=self.zero,
                                      dropout=1.0, summary_writer=summary_writer)

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

