import numpy as np
import tensorflow as tf

class ModelParams:
  def __init__(self):
    self.INPUT_SHAPE = [24,24,24,1]
    self.CONVOLUTION_LAYERS = [{'type': 'conv3d', 'filter': [5, 5, 5,  1,  10], 'downsampling': {'type': 'max_pool3d', 'k': 2}},
                               {'type': 'conv3d', 'filter': [5, 5, 5, 10, 100], 'downsampling': {'type': 'max_pool3d', 'k': 2}}]
    self.HIDDEN_LAYERS = [{'shape': [400]}, {'shape': [400]}]
    self.LATENT_SHAPE = [15]
    self.COERCED_LATENT_DIMS = [{'name': 'big_scale'}, {'name': 'med_scale'}, {'name': 'sml_scale'}]
    self.LEARNING_RATE = 0.0001
    self.CLIP_GRADIENTS = 0
    self.DROPOUT = 0.8 # Keep-prob
    self.FLOAT_TYPE = tf.float32
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
      self.latent_placeholders = [tf.placeholder(self.MP.FLOAT_TYPE, 
                                                 shape=[preset_batch_size],
                                                 name=COERCED_DIM['name']+"_latent_placeholder")
                                  for COERCED_DIM in self.MP.COERCED_LATENT_DIMS]
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
        if self.MP.DROPOUT is not None:
          layer_output = tf.nn.dropout(layer_output, self.dropout_placeholder)
        # set up next loop
        previous_layer = layer_output
        previous_layer_shape = layer_shape
    # Post fully-connected layer
    with tf.name_scope('Reconstruction') as scope:
      layer_shape = [np.prod(self.shape_before_flattening)]
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
      with tf.name_scope('CoercionLoss') as sub_scope:
        # Coerced Latent Values
        if len(self.MP.LATENT_SHAPE) != 1:
          print("Latent space of shape !=1 is not currently supported.")
          print("Here, it would cause the coerced_z_samples to have more than 1 value per batch example")
          raise NotImplementedError
        with tf.name_scope('ZValuesToCoerce') as scope:
          self.coerced_z_samples = [self.z_sample[:,i] for i, _ in enumerate(self.MP.COERCED_LATENT_DIMS)]
        # Latent space coercion (TODO: how to calculate loss on sigma uncertainty?)
        coercion_loss = tf.zeros(tf.shape(latent_loss), dtype=self.MP.FLOAT_TYPE)
        for target, z_sample, DIM in zip(self.latent_placeholders,
                                         self.coerced_z_samples,
                                         self.MP.COERCED_LATENT_DIMS):
          with tf.name_scope(DIM['name']+'Loss') as sub_sub_scope:
            coercion_loss = (coercion_loss +
                             tf.square(target - z_sample))
      # Average sum of costs over batch.
      self.cost = tf.reduce_mean(reconstruction_loss + latent_loss + coercion_loss, name="cost")
    # Optimizer (ADAM)
    with tf.name_scope('Optimizer') as scope:
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.MP.LEARNING_RATE).minimize(self.cost)
      if self.MP.CLIP_GRADIENTS > 0:
        adam = tf.train.AdamOptimizer(learning_rate=self.MP.LEARNING_RATE)
        gvs = adam.compute_gradients(self.cost)
        capped_gvs = [(tf.clip_by_norm(grad, self.MP.CLIP_GRADIENTS), var) for grad, var in gvs]
        self.optimizer = adam.apply_gradients(capped_gvs)
    # Initialize session
    self.catch_nans = tf.add_check_numerics_ops()
    self.sess = tf.Session()
    tf.initialize_all_variables().run(session=self.sess)
    # Saver
    variable_names = {}
    for var in self.variables:
      variable_names[var.name] = var
    self.saver = tf.train.Saver(variable_names)

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
  def train_on_single_batch(self, batch_input, batch_latent_targets=[], cost_only=False, dropout=None):
    # feed placeholders
    dict_ = {self.input_placeholder: batch_input}
    if self.MP.DROPOUT is not None:
      dict_[self.dropout_placeholder] = self.MP.DROPOUT if dropout is None else dropout
    else:
      if dropout is not None:
        raise ValueError('This model does not implement dropout yet a value was specified')
    if len(self.MP.COERCED_LATENT_DIMS) != len(batch_latent_targets):
      print(self.MP.COERCED_LATENT_DIMS)
      print(batch_latent_targets)
      raise ValueError('latent_dim_targets are missing, but required')
    for placeholder, target in zip(self.latent_placeholders, batch_latent_targets):
      dict_[placeholder] = target
    # compute
    if cost_only:
      cost = self.sess.run(self.cost,
                           feed_dict=dict_)
    else:
      cost, _, _ = self.sess.run((self.cost, self.optimizer, self.catch_nans),
                                 feed_dict=dict_)
    return cost
  def cost_on_single_batch(self, batch_input, batch_latent_targets=[]):
    return self.train_on_single_batch(batch_input, batch_latent_targets, cost_only=True, dropout=1.0)

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
