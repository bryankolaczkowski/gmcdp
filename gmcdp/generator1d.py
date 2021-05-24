from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.keras import initializers, regularizers, constraints, Model
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils
import tensorflow.keras.backend as K
import tensorflow as tf

from encoding1d import BinaryOneHotEncoding, NoisyBinaryOneHotEncoding


class AdaModConv1D(Layer):
  """
  adaptively modulated convolutions in 1D
  modulations adapted from https://arxiv.org/pdf/1912.04958.pdf
  code adapted from https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0
  """

  def __init__(self,
               filters,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(AdaModConv1D, self).__init__(**kwargs)
    # config copy
    self.filters            = filters
    self.kernel_size        = conv_utils.normalize_tuple(kernel_size, 1,
                                                         'kernel_size')
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer   = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer   = regularizers.get(bias_regularizer)
    self.kernel_constraint  = constraints.get(kernel_constraint)
    self.bias_constraint    = constraints.get(bias_constraint)
    # construct
    self.weight_scale = tf.keras.layers.Dense(self.filters,
                                    activation=tf.keras.activations.softplus,
                                    use_bias=True,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    return

  def build(self, input_shape):
    channel_axis = -1
    input_dim    = input_shape[0][channel_axis]
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    self.kernel = self.add_weight(name='kernel',
                                  shape=kernel_shape,
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True,
                                  constraint=self.kernel_constraint)
    return

  def call(self, inputs):
    """
    inputs should be [data, latent_space]
    output is new    [data]
    """
    data = inputs[0]
    ltnt = inputs[1]

    ### weight modulation
    # convert latent_space to weight scaling factors for each feature map (N,C)
    s = self.weight_scale(ltnt) + 1

    # self.kernel has shape (kernel_size, C, self.filters)
    # add minibatch dimension to wo -> (1, kernel_size, C, self.filters)
    wo = K.expand_dims(self.kernel, axis=0)

    # w is weight scales for each feature map (N,C)
    # w's shape must be compatible with wo (1, kernel_size, C, self.filters)
    # so we want w's shape: (N,1,C,1)
    w = K.expand_dims(K.expand_dims(s, axis=1), axis=-1)

    # modulate initial weights wo by scales w
    # weights should have shape (N, kernel_size, C, self.filters)
    weights = wo * w

    ### weight demodulation
    # demodulate weights (no reshaping)
    d = tf.math.rsqrt(tf.math.reduce_sum(tf.math.square(weights),
                                         axis=[1,2],
                                         keepdims=True) + 1.0e-8)
    weights *= d

    ### reshape input
    # data should come in as channels-last (N,W,C)
    # convert to channels-first (N,C,W)
    x = tf.transpose(data, [0,2,1])
    # reshape minibatch to convolution groups
    # x has shape (N,C,W) - reshape to (1,C*N,W)
    x = tf.reshape(x, [ 1, -1, x.shape[2] ])
    # reshape weights.
    # weights has shape (N, kernel_size, C, self.filters)
    #    - transpose to (kernel_size, C, N, self.filters)
    #    - reshape to (kernel_size,C,self.filters*N)
    w = tf.reshape(tf.transpose(weights, [1, 2, 0, 3]),
                   [weights.shape[1], weights.shape[2], -1])

    # 1d grouped convolution
    x = tf.nn.conv1d(x, # (1,C*N,W)
                     w, # (kernel_size, C, self.filters*N)
                     stride=1,
                     padding="SAME",
                     data_format="NCW")

    ## Reshape output.
    # reshape convolution back to minibatch: x's new shape (N,self.filters,W)
    x = tf.reshape(x, [ -1, self.filters, x.shape[2] ])
    # tanspose back to channels-last: x's new shape (N,W,self.filters)
    x = tf.transpose(x, [0, 2, 1])
    return x

  def get_config(self):
    config = super(AdaModConv1D, self).get_config()
    config.update({
      'filters'            : self.filters,
      'kernel_size'        : self.kernel_size,
      'kernel_initializer' : initializers.serialize(self.kernel_initializer),
      'bias_initializer'   : initializers.serialize(self.bias_initializer),
      'kernel_regularizer' : regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer'   : regularizers.serialize(self.bias_regularizer),
      'kernel_constraint'  : constraints.serialize(self.kernel_constraint),
      'bias_constraint'    : constraints.serialize(self.bias_constraint),
    })
    return config


class AdaNoise(Layer):
  """
  adaptive noise generation
  """
  def __init__(self,
               filters,
               kernel_initializer='glorot_normal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(AdaNoise, self).__init__(**kwargs)
    # config copy
    self.filters            = filters
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer   = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer   = regularizers.get(bias_regularizer)
    self.kernel_constraint  = constraints.get(kernel_constraint)
    self.bias_constraint    = constraints.get(bias_constraint)
    # construct
    self.bias_vals   = tf.keras.layers.Dense(self.filters,
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    self.noise_scale = tf.keras.layers.Dense(self.filters,
                                    activation=tf.keras.activations.softplus,
                                    use_bias=True,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    return

  def build(self, input_shape):
    """
    check input_shape
    input_shape is [data, latent_space, noise]
      data         is [Nd, Wd, Cd]
      latent_space is [Nl, Cl]
      noise        is [Nn, Wn]
    """
    data = input_shape[0]
    ltnt = input_shape[1]
    nois = input_shape[2]
    # check self.filters == data channels (Cd)
    Cd = data[-1]
    assert self.filters == Cd, \
      'AdaNoise filters {} != data channels {}'.format(self.filters, Cd)
    # check data width (Wd) <= noise width (Wn)
    Wd = data[1]
    Wn = nois[-1]
    assert Wd <= Wn, \
      'AdaNoise data width {} > noise width {}'.format(Wd, Wn)
    # check batch dimensions are the same
    assert data[0] == ltnt[0] == nois[0], \
      'AdaNoise batch dimensions are different'
    return

  def call(self, inputs):
    """
    inputs should be [data, latent_space, noise]
    output is new    [data]
    """
    data = inputs[0]
    ltnt = inputs[1]
    nois = inputs[2]
    ### crop noise (nois)
    # nois has shape (N,W0); data has shape (N,W,C)
    # crop nois to shape (N,W)
    nois = tf.slice(nois,[0,0], [-1,tf.shape(data)[1]])
    # add channel dimension to nois
    nois = K.expand_dims(nois, axis=-1)
    # nois now has shape (N,W,1)
    ### noise scaling (s)
    # calculate noise scaling factors for each channel
    s = self.noise_scale(ltnt)
    # s has shape (N,C); data has shape (N,W,C)
    # add W dimension to s
    s = K.expand_dims(s, axis=1)
    # s now has shape (N,1,C)
    ### calculate noise (n)
    # nois has shape (N,W,1), s has shape (N,1,C)
    # multiply to get noise n with shape (N,W,C)
    n = nois * (s + 1e-8)
    # n now has shape (N,W,C)
    ### bias (b)
    # calculate bias for each channel
    b = self.bias_vals(ltnt)
    # b has shape (N,C); data has shape (N,W,C)
    # add W dimension to b
    b = K.expand_dims(b, axis=1)
    # b now has shape (N,W,C)
    # add bias and noise to data
    return data + b + n

  def get_config(self):
    config = super(AdaNoise, self).get_config()
    config.update({
      'filters'            : self.filters,
      'kernel_initializer' : initializers.serialize(self.kernel_initializer),
      'bias_initializer'   : initializers.serialize(self.bias_initializer),
      'kernel_regularizer' : regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer'   : regularizers.serialize(self.bias_regularizer),
      'kernel_constraint'  : constraints.serialize(self.kernel_constraint),
      'bias_constraint'    : constraints.serialize(self.bias_constraint),
    })
    return config


class GenBlock(Layer):
  """
  encapsulates a single block of layers in a generator
  """
  def __init__(self,
               filters=256,
               kernel_size=3,
               use_bias=False,
               kernel_initializer='glorot_normal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               relu_alpha=0.2,
               **kwargs):
    super(GenBlock, self).__init__(**kwargs)
    # config copy
    self.filters            = filters
    self.kernel_size        = conv_utils.normalize_tuple(kernel_size, 1,
                                                         'kernel_size')
    self.use_bias           = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer   = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer   = regularizers.get(bias_regularizer)
    self.kernel_constraint  = constraints.get(kernel_constraint)
    self.bias_constraint    = constraints.get(bias_constraint)
    self.relu_alpha         = relu_alpha
    # construct
    ### data stream
    self.upsampl  = tf.keras.layers.Conv1DTranspose(self.filters,
                                 kernel_size=self.kernel_size,
                                 strides=2,
                                 padding='same',
                                 use_bias=self.use_bias,
                                 kernel_initializer=self.kernel_initializer,
                                 bias_initializer=self.bias_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 bias_regularizer=self.bias_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 bias_constraint=self.bias_constraint)
    self.convmod1 = AdaModConv1D(self.filters,
                                 kernel_size=self.kernel_size,
                                 kernel_initializer=self.kernel_initializer,
                                 bias_initializer=self.bias_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 bias_regularizer=self.bias_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 bias_constraint=self.bias_constraint)
    self.convmod2 = AdaModConv1D(self.filters,
                                 kernel_size=self.kernel_size,
                                 kernel_initializer=self.kernel_initializer,
                                 bias_initializer=self.bias_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 bias_regularizer=self.bias_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 bias_constraint=self.bias_constraint)
    self.noise    = AdaNoise(self.filters,
                             kernel_initializer=self.kernel_initializer,
                             bias_initializer=self.bias_initializer,
                             kernel_regularizer=self.kernel_regularizer,
                             bias_regularizer=self.bias_regularizer,
                             kernel_constraint=self.kernel_constraint,
                             bias_constraint=self.bias_constraint)
    ### image stream
    self.imgup    = tf.keras.layers.Conv1DTranspose(1,
                                kernel_size=self.kernel_size,
                                strides=2,
                                padding='same',
                                use_bias=self.use_bias,
                                kernel_initializer=self.kernel_initializer,
                                bias_initializer=self.bias_initializer,
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer,
                                kernel_constraint=self.kernel_constraint,
                                bias_constraint=self.bias_constraint)
    self.image    = tf.keras.layers.Conv1D(1,
                                kernel_size=self.kernel_size,
                                strides=1,
                                padding='same',
                                use_bias=self.use_bias,
                                kernel_initializer=self.kernel_initializer,
                                bias_initializer=self.bias_initializer,
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer,
                                kernel_constraint=self.kernel_constraint,
                                bias_constraint=self.bias_constraint)
    self.one_div_root_two = 1.0 / math.sqrt(2.0)
    return

  def _call_worker(self, inputs):
    """
    helper function to do forward-pass
    called by call(self,inputs) method
    inputs should be [data, latent_space, noise, image]
    output is new    [data, latent_space, noise, image]
    latent_space and noise are passed through unaltered
    """
    data = inputs[0]
    ltnt = inputs[1]
    nois = inputs[2]
    imge = inputs[3]
    ### data stream
    # upsample data
    x = self.upsampl(data)
    # modulated convolution 1
    x = self.convmod1([x,ltnt])
    # add bias and noise 1
    x = self.noise([x,ltnt,nois])
    # activation 1
    x = tf.nn.leaky_relu(x, alpha=self.relu_alpha)
    # modulated convolution 2
    x = self.convmod2([x,ltnt])
    # add bias and noise 2
    x = self.noise([x,ltnt,nois])
    # activation 2
    x = tf.nn.leaky_relu(x, alpha=self.relu_alpha)
    ### image stream
    # upsample image
    i = self.imgup(imge)
    # convert data to image
    j = self.image(x)
    # sum original and new images - reduce variance doubling by * 1/sqrt(2)
    k = (i+j) * self.one_div_root_two
    ### collect output
    return (x,ltnt,nois,k)

  def call(self, inputs):
    """
    inputs should be [data, latent_space, noise, image]
    output is new    [data, latent_space, noise, image]
    latent_space and noise are passed through unaltered
    """
    # unpack original inputs
    orig_data = inputs[0]
    orig_ltnt = inputs[1]
    orig_nois = inputs[2]
    orig_imag = inputs[3]
    # set up gradient tape for watching latent space
    with tf.GradientTape() as plrtape:
      plrtape.watch(orig_data)
      plrtape.watch(orig_ltnt)
      # calculate 'real' outputs using original inputs
      real_outputs = self._call_worker(inputs)
      ### start calculating 'local' path-length regularization loss
      # unpack forward-pass results image
      real_imag = real_outputs[3]
      # perturb results image; divide noise by sqrt(image_width)
      img_noise = tf.random.normal(shape=tf.shape(real_imag)) \
                  / tf.math.sqrt(tf.cast(tf.shape(real_imag)[1], tf.float32))
      # reduce noisy image
      ys = tf.math.reduce_sum(real_imag * img_noise)
    # calculate gradients wrt latent space
    grds = plrtape.gradient(ys, [orig_data, orig_ltnt])
    dta_grds = grds[0]
    lbl_grds = grds[1]
    # calculate the 2-norm of path-length gradients
    dta_ssq = tf.math.reduce_sum(tf.square(dta_grds), axis=(1,2))
    lbl_ssq = tf.math.reduce_sum(tf.square(lbl_grds), axis=1)
    plreg   = tf.math.sqrt(dta_ssq + lbl_ssq)
    # store path length regularization loss; return results
    self.add_loss(plreg)
    return real_outputs

  def get_config(self):
    config = super(GenBlock, self).get_config()
    config.update({
      'filters'            : self.filters,
      'kernel_size'        : self.kernel_size,
      'use_bias'           : self.use_bias,
      'kernel_initializer' : initializers.serialize(self.kernel_initializer),
      'bias_initializer'   : initializers.serialize(self.bias_initializer),
      'kernel_regularizer' : regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer'   : regularizers.serialize(self.bias_regularizer),
      'kernel_constraint'  : constraints.serialize(self.kernel_constraint),
      'bias_constraint'    : constraints.serialize(self.bias_constraint),
      'relu_alpha'         : self.relu_alpha,
    })
    return config


class GenStart(Layer):
  """
  starting block for data generator
  """
  def __init__(self,
               full_data_width,
               use_bias=False,
               kernel_initializer='glorot_normal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(GenStart, self).__init__(**kwargs)
    # config copy
    self.use_bias           = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer   = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer   = regularizers.get(bias_regularizer)
    self.kernel_constraint  = constraints.get(kernel_constraint)
    self.bias_constraint    = constraints.get(bias_constraint)
    self.full_data_width    = full_data_width
    # construct
    self.image = tf.keras.layers.Conv1D(1,
                                    kernel_size=1,
                                    strides=1,
                                    padding='same',
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    """
    inputs should be [labl_space, data_space]
    output is new [data, labl_space, noise, image]
    """
    # extract label space
    lbl = inputs[0]
    # extract data space
    dta = inputs[1]
    # convert data to image
    img = self.image(dta)
    # generate noise sample across final output image width
    bs = tf.shape(img)[0]
    noi = tf.random.normal(shape=(bs, self.full_data_width))
    return (dta, lbl, noi, img)

  def get_config(self):
    config = super(GenStart, self).get_config()
    config.update({
      'use_bias'           : self.use_bias,
      'kernel_initializer' : initializers.serialize(self.kernel_initializer),
      'bias_initializer'   : initializers.serialize(self.bias_initializer),
      'kernel_regularizer' : regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer'   : regularizers.serialize(self.bias_regularizer),
      'kernel_constraint'  : constraints.serialize(self.kernel_constraint),
      'bias_constraint'    : constraints.serialize(self.bias_constraint),
      'full_data_width'    : self.full_data_width,
    })
    return config


class LblBlock(Layer):
  """
  encapsulates a layer in the label->latent_space projection
  """
  def __init__(self,
               filters=256,
               use_bias=True,
               kernel_initializer='glorot_normal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               relu_alpha=0.2,
               **kwargs):
    super(LblBlock, self).__init__(**kwargs)
    # config copy
    self.filters            = filters
    self.use_bias           = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer   = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer   = regularizers.get(bias_regularizer)
    self.kernel_constraint  = constraints.get(kernel_constraint)
    self.bias_constraint    = constraints.get(bias_constraint)
    self.relu_alpha         = relu_alpha
    # construct
    self.dense = tf.keras.layers.Dense(self.filters,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    """
    converts flattened input labels to latent space
    """
    x = self.dense(inputs)
    x = tf.nn.leaky_relu(x, alpha=self.relu_alpha)
    return x

  def get_config(self):
    config = super(LblBlock, self).get_config()
    config.update({
      'filters'            : self.filters,
      'use_bias'           : self.use_bias,
      'kernel_initializer' : initializers.serialize(self.kernel_initializer),
      'bias_initializer'   : initializers.serialize(self.bias_initializer),
      'kernel_regularizer' : regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer'   : regularizers.serialize(self.bias_regularizer),
      'kernel_constraint'  : constraints.serialize(self.kernel_constraint),
      'bias_constraint'    : constraints.serialize(self.bias_constraint),
      'relu_alpha'         : self.relu_alpha,
    })
    return config


class DtaBlock(Layer):
  """
  encapsulates a layer in the data->latent_space projection
  """
  def __init__(self,
               filters=256,
               use_bias=True,
               kernel_initializer='glorot_normal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               relu_alpha=0.2,**kwargs):
    super(DtaBlock, self).__init__(**kwargs)
    # config copy
    self.filters            = filters
    self.use_bias           = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer   = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer   = regularizers.get(bias_regularizer)
    self.kernel_constraint  = constraints.get(kernel_constraint)
    self.bias_constraint    = constraints.get(bias_constraint)
    self.relu_alpha         = relu_alpha
    # construct
    self.dnsampl = tf.keras.layers.Conv1D(self.filters,
                                   kernel_size=2,
                                   strides=2,
                                   padding='valid',
                                   use_bias=self.use_bias,
                                   kernel_initializer=self.kernel_initializer,
                                   bias_initializer=self.bias_initializer,
                                   kernel_regularizer=self.kernel_regularizer,
                                   bias_regularizer=self.bias_regularizer,
                                   kernel_constraint=self.kernel_constraint,
                                   bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    """
    converts inputs (N,W,C) to latent space (N,W,C)
    """
    x = self.dnsampl(inputs)
    x = tf.nn.leaky_relu(x, alpha=self.relu_alpha)
    return x

  def get_config(self):
    config = super(DtaBlock, self).get_config()
    config.update({
      'filters'            : self.filters,
      'use_bias'           : self.use_bias,
      'kernel_initializer' : initializers.serialize(self.kernel_initializer),
      'bias_initializer'   : initializers.serialize(self.bias_initializer),
      'kernel_regularizer' : regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer'   : regularizers.serialize(self.bias_regularizer),
      'kernel_constraint'  : constraints.serialize(self.kernel_constraint),
      'bias_constraint'    : constraints.serialize(self.bias_constraint),
      'relu_alpha'         : self.relu_alpha,
    })
    return config


class ImgGenOut(Layer):
  """
  converts [data,latent_space,noise,img,labels] to [img,labels]
  """
  def __init__(self, **kwargs):
    super(ImgGenOut, self).__init__(**kwargs)
    return

  def call(self, inputs, labels):
    """
    inputs should be [data, latent_space, noise, img]
    output is new [img, labels]
    """
    return (inputs[-1], labels)

  def get_config(self):
    return super(ImgGenOut, self).get_config()


def MCGen(labl_input_shape=(1,),
          labl_encoding=NoisyBinaryOneHotEncoding(),
          labl_filters=[256,256,256,256],
          labl_use_bias=True,
          labl_kernel_initializer='glorot_normal',
          labl_bias_initializer='zeros',
          labl_kernel_regularizer=None,
          labl_bias_regularizer=None,
          labl_kernel_constraint=None,
          labl_bias_constraint=None,
          labl_relu_alpha=0.2,

          data_input_shape=(1024,1),
          data_encoding=None,
          data_filters=[32,64,128,256],
          data_use_bias=False,
          data_kernel_initializer='glorot_normal',
          data_bias_initializer='zeros',
          data_kernel_regularizer=None,
          data_bias_regularizer=None,
          data_kernel_constraint=None,
          data_bias_constraint=None,
          data_relu_alpha=0.2,

          genr_filters=[256,128,64,32],
          genr_kernel_size=3,
          genr_use_bias=False,
          genr_kernel_initializer='glorot_normal',
          genr_bias_initializer='zeros',
          genr_kernel_regularizer=None,
          genr_bias_regularizer=None,
          genr_kernel_constraint=None,
          genr_bias_constraint=None,
          genr_relu_alpha=0.2,
          **kwargs):
  """
  returns a generator Model using the functional API
  """
  labl_kernel_initializer = initializers.get(labl_kernel_initializer)
  labl_bias_initializer   = initializers.get(labl_bias_initializer)
  labl_kernel_regularizer = regularizers.get(labl_kernel_regularizer)
  labl_bias_regularizer   = regularizers.get(labl_bias_regularizer)
  labl_kernel_constraint  = constraints.get(labl_kernel_constraint)
  labl_bias_constraint    = constraints.get(labl_bias_constraint)

  data_kernel_initializer = initializers.get(data_kernel_initializer)
  data_bias_initializer   = initializers.get(data_bias_initializer)
  data_kernel_regularizer = regularizers.get(data_kernel_regularizer)
  data_bias_regularizer   = regularizers.get(data_bias_regularizer)
  data_kernel_constraint  = constraints.get(data_kernel_constraint)
  data_bias_constraint    = constraints.get(data_bias_constraint)

  genr_kernel_initializer = initializers.get(genr_kernel_initializer)
  genr_bias_initializer   = initializers.get(genr_bias_initializer)
  genr_kernel_regularizer = regularizers.get(genr_kernel_regularizer)
  genr_bias_regularizer   = regularizers.get(genr_bias_regularizer)
  genr_kernel_constraint  = constraints.get(genr_kernel_constraint)
  genr_bias_constraint    = constraints.get(genr_bias_constraint)

  ### build label to latent-space projection
  # label input and encoding
  labl_input  = tf.keras.Input(shape=labl_input_shape)
  if labl_encoding:
    labl_output = labl_encoding(labl_input)
  else:
    labl_output = labl_input
  # flatten labels
  labl_output = tf.keras.layers.Flatten()(labl_output)
  # build label blocks
  nbase = 'lbl_'
  for i in range(len(labl_filters)):
    name = nbase + str(i)
    labl_output = LblBlock(filters=labl_filters[i],
                           use_bias=labl_use_bias,
                           kernel_initializer=labl_kernel_initializer,
                           bias_initializer=labl_bias_initializer,
                           kernel_regularizer=labl_kernel_regularizer,
                           bias_regularizer=labl_bias_regularizer,
                           kernel_constraint=labl_kernel_constraint,
                           bias_constraint=labl_bias_constraint,
                           relu_alpha=labl_relu_alpha,
                           name=name)(labl_output)
  # label input now in labl_input
  # label latent-space now in labl_output

  ### build data to latent-space projection
  # data input and encoding
  data_input = tf.keras.Input(shape=data_input_shape)
  if data_encoding:
    data_output = data_encoding(data_input)
  else:
    data_output = data_input
  # build data blocks
  nbase = 'dta_'
  for i in range(len(data_filters)):
    name = nbase + str(i)
    data_output = DtaBlock(filters=data_filters[i],
                           use_bias=data_use_bias,
                           kernel_initializer=data_kernel_initializer,
                           bias_initializer=data_bias_initializer,
                           kernel_regularizer=data_kernel_regularizer,
                           bias_regularizer=data_bias_regularizer,
                           kernel_constraint=data_kernel_constraint,
                           bias_constraint=data_bias_constraint,
                           relu_alpha=data_relu_alpha,
                           name=name)(data_output)
  # data input now in data_input
  # data latent-space now in data_output

  ### build generator
  # generator start (labl_output, data_output) ->
  # (data, labl_output, data_output, image)
  output = GenStart(full_data_width=data_input_shape[0],
              use_bias=genr_use_bias,
              kernel_initializer=genr_kernel_initializer,
              bias_initializer=genr_bias_initializer,
              kernel_regularizer=genr_kernel_regularizer,
              bias_regularizer=genr_bias_regularizer,
              kernel_constraint=genr_kernel_constraint,
              bias_constraint=genr_bias_constraint)((labl_output, data_output))
  # build generator blocks
  nbase = 'gen_'
  for i in range(len(genr_filters)):
    name = nbase + str(i)
    output = GenBlock(filters=genr_filters[i],
                      kernel_size=genr_kernel_size,
                      use_bias=genr_use_bias,
                      kernel_initializer=genr_kernel_initializer,
                      bias_initializer=genr_bias_initializer,
                      kernel_regularizer=genr_kernel_regularizer,
                      bias_regularizer=genr_bias_regularizer,
                      kernel_constraint=genr_kernel_constraint,
                      bias_constraint=genr_bias_constraint,
                      relu_alpha=genr_relu_alpha,
                      name=name)(output)
  # build generator output
  output = ImgGenOut()(output, labl_input)
  return Model(inputs=(data_input, labl_input),
               outputs=output,
               name='mc_gen', **kwargs)


if __name__ == '__main__':
  """
  module test (well, example, anyway)
  """
  nbatches = 2
  labelshp = (nbatches,1)
  datashp  = (nbatches,1024,1)
  # create labels
  labels = tf.math.rint(tf.random.uniform(shape=labelshp))
  print('LABELS', labels)
  # create data
  data = tf.random.normal(shape=datashp)
  print('DATA', data)

  # test generator forward pass
  generator = MCGen()
  generator.compile()
  generator.summary(line_length=200, positions=[0.35,0.7,0.8,1.0])
  out = generator((data,labels))
  print('OUTPUT', out)
  print('LOSSES', generator.losses)

  # test model serialization
  generator.save('TMP.model')
  m = tf.keras.models.load_model('TMP.model')

  m.summary(line_length=200, positions=[0.35,0.7,0.8,1.0])
  out = m((data,labels))
  print('OUTPUT', out)
  print('LOSSES', m.losses)
