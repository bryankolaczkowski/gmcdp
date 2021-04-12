import tensorflow as tf
from tensorflow.python.keras import backend as K


class NoisyBinaryOneHotEncoding(tf.keras.layers.Layer):
  """
  encodes a [0,1] integer as 'noisy' one-hot
  """
  def __init__(self, **kwargs):
    super(NoisyBinaryOneHotEncoding, self).__init__(**kwargs)
    return

  def build(self, input_shape):
    return

  def call(self, inputs):
    onehot  = tf.one_hot(tf.cast(inputs, tf.int32), 2)
    noise   = tf.random.uniform(shape=tf.shape(onehot), minval=0.0, maxval=0.2)
    normd,n = tf.linalg.normalize(onehot+noise, axis=-1, ord=1)
    return normd

  def get_config(self):
    return super(NoisyBinaryOneHotEncoding, self).get_config()


class BinaryOneHotEncoding(tf.keras.layers.Layer):
  """
  encodes a [0,1] integer as one-hot
  """
  def __init__(self, **kwargs):
    super(BinaryOneHotEncoding, self).__init__(**kwargs)
    return

  def build(self, input_shape):
    return

  def call(self, inputs):
    return tf.one_hot(tf.cast(inputs, tf.int32), 2)

  def get_config(self):
    return super(BinaryOneHotEncoding, self).get_config()


class StandardGaussianNoise(tf.keras.layers.Layer):
  """
  adds standard gaussian noise to a tensor
  """
  def __init__(self, **kwargs):
    super(StandardGaussianNoise, self).__init__(**kwargs)
    return

  def build(self, input_shape):
    # no weights in this layer
    return

  def call(self, inputs):
    noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=1.0)
    return inputs + noise

  def get_config(self):
    return super(StandardGaussianNoise, self).get_config()


class AdaptiveGaussianNoise(tf.keras.layers.Layer):
  """
  adds scaled gaussian noise to a tensor
  """
  def __init__(self, **kwargs):
    super(AdaptiveGaussianNoise, self).__init__(**kwargs)
    return

  def build(self, input_shape):
    # no weights in this layer
    return

  def call(self, inputs):
    data = inputs[0]
    nmlt = inputs[1]
    noise = tf.random.normal(shape=tf.shape(data), mean=0.0, stddev=1.0)
    return data + noise * nmlt

  def get_config(self):
    return super(AdaptiveGaussianNoise, self).get_config()


class NoisyInput(tf.keras.layers.Layer):
  """
  input layer with controlled gaussian noise
  """
  def __init__(self, **kwargs):
    super(NoisyInput, self).__init__(**kwargs)
    return

  def build(self, data_shape):
    return

  def call(self, inputs):
    means = inputs[0]
    stdvs = inputs[1]
    return tf.random.normal(shape=tf.shape(means), mean=means, stddev=stdvs)

  def get_config(self):
    config = super(NoisyInput, self).get_config()
    return config


class LinearInput(tf.keras.layers.Layer):
  """
  linear input layer
  """
  def __init__(self,
               data_dim,
               filters,
               use_bias,
               kernel_initializer=None,
               **kwargs):
    super(LinearInput, self).__init__(**kwargs)
    self.linear  = tf.keras.layers.Dense(units=data_dim * filters,
                                         use_bias=use_bias,
                                         kernel_initializer=kernel_initializer,
                                         name='linear')
    self.reshape = tf.keras.layers.Reshape(target_shape=(data_dim,filters),
                                           name='reshp')
    return

  def build(self, input_shape):
    return

  def call(self, inputs):
    batch_size = tf.shape(inputs)[0]
    x = tf.ones((batch_size,1), name='const')
    x = self.linear(x)  # linear transform constant 1.0 to data length
    x = self.reshape(x) # reshape to get channel for 1D convolutions
    return x

  def get_config(self):
    config = super(LinearInput, self).get_config()
    config.update({
      'linear' : self.linear,
      'reshape': self.reshape,
    })
    return config


#===============================================================================
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
@tf.keras.utils.register_keras_serializable(package="Addons")
class SpectralNormalization(tf.keras.layers.Wrapper):
  """Performs spectral normalization on weights.

  This wrapper controls the Lipschitz constant of the layer by
  constraining its spectral norm, which can stabilize the training of GANs.

  See [Spectral Normalization for Generative Adversarial Networks]
  (https://arxiv.org/abs/1802.05957).

  Wrap `tf.keras.layers.Conv2D`:

  >>> x = np.random.rand(1, 10, 10, 1)
  >>> conv2d = SpectralNormalization(tf.keras.layers.Conv2D(2, 2))
  >>> y = conv2d(x)
  >>> y.shape
  TensorShape([1, 9, 9, 2])

  Wrap `tf.keras.layers.Dense`:

  >>> x = np.random.rand(1, 10, 10, 1)
  >>> dense = SpectralNormalization(tf.keras.layers.Dense(10))
  >>> y = dense(x)
  >>> y.shape
  TensorShape([1, 10, 10, 10])

  Args:
    layer: A `tf.keras.layers.Layer` instance that
      has either `kernel` or `embeddings` attribute.
    power_iterations: `int`, the number of iterations during normalization.
  Raises:
    AssertionError: If not initialized with a `Layer` instance.
    ValueError: If initialized with negative `power_iterations`.
    AttributeError: If `layer` does not has `kernel` or `embeddings`
    attribute.
  """

  def __init__(self, layer: tf.keras.layers, power_iterations=1, **kwargs):
    super().__init__(layer, **kwargs)
    if power_iterations <= 0:
      raise ValueError(
        "`power_iterations` should be greater than zero, got "
        "`power_iterations={}`".format(power_iterations)
      )
    self.power_iterations = power_iterations
    self._initialized = False

  def build(self, input_shape):
    """Build `Layer`"""
    super().build(input_shape)
    input_shape = tf.TensorShape(input_shape)
    self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

    if hasattr(self.layer, "kernel"):
      self.w = self.layer.kernel
    elif hasattr(self.layer, "embeddings"):
      self.w = self.layer.embeddings
    else:
      raise AttributeError(
        "{} object has no attribute 'kernel' nor "
        "'embeddings'".format(type(self.layer).__name__)
      )

    self.w_shape = self.w.shape.as_list()

    self.u = self.add_weight(
      shape=(1, self.w_shape[-1]),
      initializer=tf.initializers.TruncatedNormal(stddev=0.02),
      trainable=False,
      name="sn_u",
      dtype=self.w.dtype,
    )

  def call(self, inputs, training=None):
    """Call `Layer`"""
    if training is None:
      training = tf.keras.backend.learning_phase()

    if training:
      self.normalize_weights()

    output = self.layer(inputs)
    return output

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(self.layer.compute_output_shape(input_shape)\
                          .as_list())

  @tf.function
  def normalize_weights(self):
    """Generate spectral normalized weights.

    This method will update the value of `self.w` with the
    spectral normalized value, so that the layer is ready for `call()`.
    """

    w = tf.reshape(self.w, [-1, self.w_shape[-1]])
    u = self.u

    with tf.name_scope("spectral_normalize"):
      for _ in range(self.power_iterations):
        v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
        u = tf.math.l2_normalize(tf.matmul(v, w))

        sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)

        self.w.assign(self.w / sigma)
        self.u.assign(u)

  def get_config(self):
    config = {"power_iterations": self.power_iterations}
    base_config = super().get_config()
    return {**base_config, **config}


#===============================================================================
# adaptive instance normalization
#    https://arxiv.org/pdf/1703.06868.pdf
#    https://arxiv.org/pdf/1812.04948.pdf
# adapted from:
#    https://github.com/manicman1999/StyleGAN-Keras
# Input b and g should be 1xC for 1D data
class AdaInstanceNormalization(tf.keras.layers.Layer):
  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               **kwargs):
    super(AdaInstanceNormalization, self).__init__(**kwargs)
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    return

  def build(self, input_shape):
    dim = input_shape[0][self.axis]
    if dim is None:
      raise ValueError('Axis ' + str(self.axis) + ' of '
                       'input tensor should have a defined dimension '
                       'but the layer received an input with shape ' +
                       str(input_shape[0]) + '.')
    super(AdaInstanceNormalization, self).build(input_shape)
    return

  def call(self, inputs, training=None):
    input_shape    = K.int_shape(inputs[0])
    reduction_axes = list(range(0, len(input_shape)))

    beta  = inputs[1]
    gamma = inputs[2]

    if self.axis is not None:
      del reduction_axes[self.axis]

    del reduction_axes[0]
    mean   = 0.0
    stddev = 1.0
    if self.center:
      mean   = K.mean(inputs[0], reduction_axes, keepdims=True)
    if self.scale:
      stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
    normed = (inputs[0] - mean) / stddev
    return normed * gamma + beta

  def get_config(self):
    config = super(AdaInstanceNormalization, self).get_config()
    config.update({
      'axis': self.axis,
      'momentum': self.momentum,
      'epsilon': self.epsilon,
      'center': self.center,
      'scale': self.scale
    })
    return config

  def compute_output_shape(self, input_shape):
    return input_shape[0]
