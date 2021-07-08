from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import initializers, regularizers, constraints, Model
from tensorflow.keras.layers import Layer
import tensorflow as tf
import math

## BASE CLASSES ################################################################

class ConfigLayer(Layer):
  """
  base class for layers having sub-layers requiring configuration
  """
  def __init__(self,
               use_bias=True,
               kernel_initializer='glorot_normal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(ConfigLayer, self).__init__(**kwargs)
    # config copy
    self.use_bias           = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer   = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer   = regularizers.get(bias_regularizer)
    self.kernel_constraint  = constraints.get(kernel_constraint)
    self.bias_constraint    = constraints.get(bias_constraint)
    return

  def get_config(self):
    config = super(ConfigLayer, self).get_config()
    config.update({
      'use_bias'           : self.use_bias,
      'kernel_initializer' : initializers.serialize(self.kernel_initializer),
      'bias_initializer'   : initializers.serialize(self.bias_initializer),
      'kernel_regularizer' : regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer'   : regularizers.serialize(self.bias_regularizer),
      'kernel_constraint'  : constraints.serialize(self.kernel_constraint),
      'bias_constraint'    : constraints.serialize(self.bias_constraint),
    })
    return config


class WidthLayer(ConfigLayer):
  """
  base class for configurable layer with data width
  """
  def __init__(self, width, *args, **kwargs):
    super(WidthLayer, self).__init__(*args, **kwargs)
    # config copy
    self.width = width
    return

  def get_config(self):
    config = super(WidthLayer, self).get_config()
    config.update({
      'width' : self.width,
    })
    return config


class ReluLayer(WidthLayer):
  """
  base class for layers with leaky-ReLU activations
  """
  def __init__(self, relu_alpha=0.2, *args, **kwargs):
    super(ReluLayer, self).__init__(*args, **kwargs)
    # config copy
    self.relu_alpha = relu_alpha
    return

  def get_config(self):
    config = super(ReluLayer, self).get_config()
    config.update({
      'relu_alpha' : self.relu_alpha,
    })
    return config


class EncodeLayer(WidthLayer):
  """
  base class for position and linear projection encoding layers
  """
  def __init__(self, *args, **kwargs):
    super(EncodeLayer, self).__init__(*args, **kwargs)
    # construct
    self.pos = tf.expand_dims(tf.linspace(+1.0, -1.0, self.width), axis=0)
    self.flt = tf.keras.layers.Flatten()
    self.lpr = tf.keras.layers.Dense(units=self.width,
                                  use_bias=self.use_bias,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  bias_regularizer=self.bias_regularizer,
                                  kernel_constraint=self.kernel_constraint,
                                  bias_constraint=self.bias_constraint)
    return

## ENCODER CLASSES #############################################################

class EncodeGen(EncodeLayer):
  """
  encodes labels for generator
  """
  def __init__(self, *args, **kwargs):
    super(EncodeGen, self).__init__(*args, **kwargs)
    # construct
    self.lp2 = tf.keras.layers.Dense(units=self.width,
                                  use_bias=self.use_bias,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  bias_regularizer=self.bias_regularizer,
                                  kernel_constraint=self.kernel_constraint,
                                  bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    bs = tf.shape(inputs)[0]
    ps = tf.tile(self.pos, multiples=(bs,1))    # sequence position encoding
    ip = self.flt(inputs)                       # flatten inputs
    lp = self.lpr(ip)                           # linear project labels
    dt = self.lp2(ip)                           # linear project data
    return tf.stack((ps, lp, dt), axis=-1)


class DecodeGen(ConfigLayer):
  """
  decodes generator output to data
  """
  def __init__(self, *args, **kwargs):
    super(DecodeGen, self).__init__(*args, **kwargs)
    # construct
    self.lpr = tf.keras.layers.LocallyConnected1D(filters=1,
                                    kernel_size=1,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    self.flt = tf.keras.layers.Flatten()
    return

  def call(self, inputs):
    return self.flt(self.lpr(inputs))


class PosMaskedMHABlock(ReluLayer):
  """
  multi-head attention + ffwd block with position vector mask
  """
  def __init__(self, dim, heads, *args, **kwargs):
    super(PosMaskedMHABlock, self).__init__(*args, **kwargs)
    # config copy
    self.dim   = dim
    self.heads = heads
    # construct
    # multi-head attention layer
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.heads,
                                    key_dim=self.dim,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    # feed-forward layers
    self.ff1 = tf.keras.layers.Dense(units=self.dim * 2,
                                     use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     bias_initializer=self.bias_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer,
                                     kernel_constraint=self.kernel_constraint,
                                     bias_constraint=self.bias_constraint)
    self.ff2 = tf.keras.layers.Dense(units=self.dim * 2,
                                     use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     bias_initializer=self.bias_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer,
                                     kernel_constraint=self.kernel_constraint,
                                     bias_constraint=self.bias_constraint)
    # linear back-projection layer
    self.lpr = tf.keras.layers.Dense(units=self.dim,
                                     use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     bias_initializer=self.bias_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer,
                                     kernel_constraint=self.kernel_constraint,
                                     bias_constraint=self.bias_constraint)
    # layer normalizations
    self.ln1 = tf.keras.layers.LayerNormalization(axis=(-2,-1))
    self.ln2 = tf.keras.layers.LayerNormalization(axis=(-2,-1))
    # position masking
    msk_list = [tf.zeros(shape=(1,self.width))]
    for i in range(1,self.dim):
      msk_list.append(tf.ones(shape=(1,self.width)))
    self.msk = tf.stack(msk_list, axis=-1)
    return

  def call(self, inputs):
    # sub-block 1 - multi-head attention with residual connection
    a = self.ln1(self.mha(inputs,inputs))
    a = inputs + (a * self.msk)
    # sub-block 2 - feed-forward with residual connection
    b = tf.nn.leaky_relu(self.ff1(a), alpha=self.relu_alpha)
    b = tf.nn.leaky_relu(self.ff2(b), alpha=self.relu_alpha)
    b = self.ln2(self.lpr(b))
    b = a + (b * self.msk)
    return b

  def get_config(self):
    config = super(PosMaskedMHABlock, self).get_config()
    config.update({
      'dim'   : self.dim,
      'heads' : self.heads,
    })
    return config


class AveUpsample(Layer):
  """
  average upsampling
  """
  def __init__(self, *args, **kwargs):
    super(AveUpsample, self).__init__(*args, **kwargs)
    # construct
    self.upspl = tf.keras.layers.UpSampling1D(size=2)
    self.avepl = tf.keras.layers.AveragePooling1D(pool_size=3,
                                                  strides=1,
                                                  padding='same')
    return

  def call(self, inputs):
    return self.avepl(self.upspl(inputs))


class DataNoise(WidthLayer):
  """
  adaptive noise injection into data
  """
  def __init__(self, *args, **kwargs):
    super(DataNoise, self).__init__(*args, **kwargs)
    # construct
    self.mean = tf.keras.layers.LocallyConnected1D(filters=1,
                                kernel_size=1,
                                use_bias=self.use_bias,
                                kernel_initializer=tf.keras.initializers.zeros,
                                bias_initializer=tf.keras.initializers.zeros,
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer,
                                kernel_constraint=self.kernel_constraint,
                                bias_constraint=self.bias_constraint)
    self.stdv = tf.keras.layers.LocallyConnected1D(filters=1,
                                kernel_size=1,
                                activation=tf.keras.activations.relu,
                                use_bias=self.use_bias,
                                kernel_initializer=tf.keras.initializers.zeros,
                                bias_initializer=tf.keras.initializers.zeros,
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer,
                                kernel_constraint=self.kernel_constraint,
                                bias_constraint=self.bias_constraint)
    self.mask = tf.stack((tf.zeros(shape=(1,self.width)),
                          tf.zeros(shape=(1,self.width)),
                           tf.ones(shape=(1,self.width))), axis=-1)
    return

  def call(self, inputs):
    bs = tf.shape(inputs)[0]
    mn = self.mean(inputs)          # project noise means
    sd = self.stdv(inputs) + 1.0e-5 # project noise stdvs
    # generate masked random vector affecting only data dimension
    rv = tf.random.normal(mean=mn,
                          stddev=sd,
                          shape=(bs,self.width,1)) * self.mask
    return inputs + rv


class UpsamplBlock(ReluLayer):
  """
  incremental generator from initial to final data width
  """
  def __init__(self, init_width, *args, attn_dim=3, attn_hds=4, **kwargs):
    super(UpsamplBlock, self).__init__(*args, **kwargs)
    # config copy
    self.init_width = init_width
    self.attn_dim   = attn_dim
    self.attn_hds   = attn_hds
    # construct
    n_upsampl_blks = int(math.log2(self.width) - math.log2(self.init_width))
    curr_width = self.init_width
    self.mhablocks = []
    for i in range(n_upsampl_blks):
      self.mhablocks.append(PosMaskedMHABlock(width=curr_width,
                                    dim=self.attn_dim,
                                    heads=self.attn_hds,
                                    relu_alpha=self.relu_alpha,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint))
      curr_width *= 2
    self.upsampler = AveUpsample()
    return

  def call(self, inputs):
    x = inputs
    for mhablock in self.mhablocks:
      x = mhablock(x)
      x = self.upsampler(x)
    return x

  def get_config(self):
    config = super(UpsamplBlock, self).get_config()
    config.update({
      'init_width' : self.init_width,
      'attn_dim'   : self.attn_dim,
      'attn_hds'   : self.attn_hds,
    })
    return config

## CONDITIONAL GENERATOR BUILD FUNCTION ########################################

def CondGen1D(input_shape, width, attn_hds=4, nattnblocks=4):
  """
  construct generator using functional API
  """
  ATTNDIM=3  # dimension of internal data representation (pos,data,proj)
  ## input encoding
  start_width = 32
  inputs = tf.keras.Input(shape=input_shape, name='lbin')
  output = EncodeGen(width=start_width, name='encd')(inputs)
  ## upsampling subnet
  output = UpsamplBlock(init_width=start_width,
                        width=width,
                        attn_dim=ATTNDIM,
                        attn_hds=attn_hds,
                        name='upsl')(output)
  ## self-attention subnet
  for i in range(nattnblocks):
    output = PosMaskedMHABlock(width=width,
                               dim=ATTNDIM,
                               heads=attn_hds,
                               name='mha{}'.format(i))(output)
    if i % 2 == 1 and i != nattnblocks-1:
      output = DataNoise(width=width, name='nse{}'.format(i))(output)
  ## data decoding
  output = DecodeGen(name='decd')(output)
  return Model(inputs=inputs, outputs=(output,inputs))


if __name__ == '__main__':
  """
  module test (well, example, anyway)
  """
  import sys
  sys.path.append("../tests")
  import test_data_generator

  ndata = 4

  # generate simulated data and labels
  data,lbls = test_data_generator.gen_dataset(ndata, plot=False)
  print(data,lbls)

  # create a little 'generator model' that just maps the label vector
  # to data space using a linear map
  input_shape  = tf.shape(lbls)
  output_shape = tf.shape(data)
  mdl = CondGen1D((input_shape[1],), output_shape[1])
  mdl.summary(positions=[0.4, 0.7, 0.8, 1.0])
  out = mdl(lbls)
  print(out)
