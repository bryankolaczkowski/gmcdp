from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras        import Model
from tensorflow.keras.layers import Layer
import tensorflow as tf
import math

from .activ import gnact
from .layrs import WidthLayer, ReluLayer
from .wrapr import SpecNorm


## BASE CLASSES ################################################################

class EncodeLayer(WidthLayer):
  """
  base class for position and linear projection encoding layers
  """
  def __init__(self, dim, *args, **kwargs):
    super(EncodeLayer, self).__init__(*args, **kwargs)
    # config copy
    self.dim = dim-1
    # construct
    self.pos = tf.linspace(+2.0, -2.0, self.width)
    self.pos = tf.expand_dims(tf.expand_dims(self.pos, axis=-1), axis=0)
    self.flt = tf.keras.layers.Flatten()
    self.lpr = tf.keras.layers.Dense(units=self.width * self.dim,
                                  use_bias=self.use_bias,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  bias_regularizer=self.bias_regularizer,
                                  kernel_constraint=self.kernel_constraint,
                                  bias_constraint=self.bias_constraint)
    return

  def get_config(self):
    config = super(EncodeLayer, self).get_config()
    config.update({
      'dim' : self.dim,
    })
    return config

## GENERATOR CLASSES ###########################################################

class EncodeGen(EncodeLayer):
  """
  encodes labels for generator
  """
  def __init__(self, *args, **kwargs):
    super(EncodeGen, self).__init__(*args, **kwargs)
    return

  def call(self, inputs):
    bs = tf.shape(inputs)[0]
    ps = tf.tile(self.pos, multiples=(bs,1,1))  # sequence position encoding
    ip = self.flt(inputs)                       # flatten inputs
    lp = self.lpr(ip)                           # linear projections
    lp = tf.reshape(lp, shape=(bs,self.width,self.dim))
    return tf.concat((ps, lp), axis=-1)


class DecodeGen(ReluLayer):
  """
  decodes generator output to data
  """
  def __init__(self, *args, dropout=0.0, **kwargs):
    super(DecodeGen, self).__init__(*args, **kwargs)
    # config copy
    self.dropout = dropout
    # construct
    self.cn1 = SpecNorm(tf.keras.layers.Conv1D(filters=self.width,
                                    kernel_size=3,
                                    padding='same',
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint))
    self.cn2 = SpecNorm(tf.keras.layers.Conv1D(filters=self.width,
                                    kernel_size=3,
                                    padding='same',
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint))
    self.out = tf.keras.layers.Dense(units=1,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    self.dot = tf.keras.layers.Dropout(rate=self.dropout)
    self.flt = tf.keras.layers.Flatten()
    return

  def _finalize(self, inputs):
    return self.flt(self.out(inputs))

  def call(self, inputs):
    x = self.dot(gnact(self.cn1(inputs), alpha=self.relu_alpha))
    x = self.dot(gnact(self.cn2(x),      alpha=self.relu_alpha))
    return self._finalize(x)

  def get_config(self):
    config = super(DecodeGen, self).get_config()
    config.update({
      'dropout' : self.dropout,
    })
    return config


class PosMaskedMHABlock(ReluLayer):
  """
  multi-head attention + ffwd block with position vector mask
  """
  def __init__(self, dim, heads, *args, dropout=0.0, **kwargs):
    super(PosMaskedMHABlock, self).__init__(*args, **kwargs)
    # config copy
    self.dim     = dim
    self.heads   = heads
    self.dropout = dropout
    # construct
    # multi-head attention layer
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.heads,
                                    key_dim=self.dim,
                                    dropout=self.dropout,
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
    self.ff2 = tf.keras.layers.Dense(units=self.dim,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    # layer normalization
    self.ln1 = tf.keras.layers.LayerNormalization(axis=(-2,-1))
    self.ln2 = tf.keras.layers.LayerNormalization(axis=(-2,-1))
    # position masking
    msk_list = [tf.zeros(shape=(1,self.width))]
    for i in range(1,self.dim):
      msk_list.append(tf.ones(shape=(1,self.width)))
    self.msk = tf.stack(msk_list, axis=-1)
    return

  def call(self, inputs):
    # sub-block 1 - pre-lyrnorm, multi-head attn, residual
    a = self.ln1(inputs)
    a = self.mha(a,a)
    a = inputs + (a * self.msk)
    # sub-block 2 - pre-lyrnorm, feed-forward, residual
    b = self.ln2(a)
    b = gnact(self.ff1(b), alpha=self.relu_alpha)
    b = gnact(self.ff2(b), alpha=self.relu_alpha)
    b = a + (b * self.msk)
    return b

  def get_config(self):
    config = super(PosMaskedMHABlock, self).get_config()
    config.update({
      'dim'     : self.dim,
      'heads'   : self.heads,
      'dropout' : self.dropout,
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


class DataNoise(ReluLayer):
  """
  adaptive noise injection into data
  """
  def __init__(self, dim, *args, **kwargs):
    super(DataNoise, self).__init__(*args, **kwargs)
    # config copy
    self.dim = dim
    # construct
    self.stdv = tf.keras.layers.Conv1D(filters=self.dim,
                                kernel_size=3,
                                padding='same',
                                use_bias=self.use_bias,
                                kernel_initializer=tf.keras.initializers.zeros,
                                bias_initializer=tf.keras.initializers.zeros,
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer,
                                kernel_constraint=self.kernel_constraint,
                                bias_constraint=self.bias_constraint)
    msks = [tf.zeros(shape=(1,self.width))] # no noise on position channel
    for i in range(dim-1):
      msks.append(tf.ones(shape=(1,self.width)))
    self.mask = tf.stack(msks, axis=-1)
    return

  def call(self, inputs):
    bs = tf.shape(inputs)[0]
    sd = gnact(self.stdv(inputs), alpha=self.relu_alpha) # project noise stdvs
    # generate masked random vector affecting only last data dimension
    rv = tf.random.normal(mean=0.0,
                          stddev=sd,
                          shape=(bs,self.width,self.dim)) * self.mask
    return inputs + rv

  def get_config(self):
    config = super(DataNoise, self).get_config()
    config.update({
      'dim' : self.dim,
    })
    return config


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
    self.datanoise = DataNoise(width=curr_width,
                               dim=self.attn_dim,
                               use_bias=self.use_bias,
                               kernel_initializer=self.kernel_initializer,
                               bias_initializer=self.bias_initializer,
                               kernel_regularizer=self.kernel_regularizer,
                               bias_regularizer=self.bias_regularizer,
                               kernel_constraint=self.kernel_constraint,
                               bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    x = inputs
    for mhablock in self.mhablocks:
      x = mhablock(x)
      x = self.upsampler(x)
    return self.datanoise(x)

  def get_config(self):
    config = super(UpsamplBlock, self).get_config()
    config.update({
      'init_width' : self.init_width,
      'attn_dim'   : self.attn_dim,
      'attn_hds'   : self.attn_hds,
    })
    return config

## CONDITIONAL GENERATOR BUILD FUNCTION ########################################

def CondGen1D(input_shape,
              width,
              attn_hds=4,
              nattnblocks=8,
              datadim=8,
              dropout=0.0):
  """
  construct generator using functional API
  """
  ## input encoding
  datadim = datadim + 1
  start_width = 64
  inputs = tf.keras.Input(shape=input_shape, name='lbin')
  output = EncodeGen(width=start_width, dim=datadim, name='encd')(inputs)
  ## upsampling subnet
  output = UpsamplBlock(init_width=start_width,
                        width=width,
                        attn_dim=datadim,
                        attn_hds=attn_hds,
                        name='upsl')(output)
  ## self-attention subnet
  for i in range(nattnblocks):
    output = PosMaskedMHABlock(width=width,
                               dim=datadim,
                               heads=attn_hds,
                               dropout=dropout,
                               name='mha{}'.format(i))(output)
    if i % 2 == 1:
      output = DataNoise(width=width,
                         dim=datadim,
                         name='nse{}'.format(i))(output)
  ## data decoding
  output = DecodeGen(width=width, dropout=dropout, name='decd')(output)
  return Model(inputs=inputs, outputs=(output,inputs))
