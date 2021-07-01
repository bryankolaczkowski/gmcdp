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
    self.width = width
    return

  def get_config(self):
    config = super(WidthLayer, self).get_config()
    config.update({
      'width' : self.width,
    })
    return


class ReluLayer(WidthLayer):
  """
  base class for layers with leaky-ReLU activations
  """
  def __init__(self, relu_alpha=0.2, *args, **kwargs):
    super(ReluLayer, self).__init__(*args, **kwargs)
    self.relu_alpha = relu_alpha
    return

  def get_config(self):
    config = super(ReluLayer, self).get_config()
    config.update({
      'relu_alpha' : self.relu_alpha,
    })
    return


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



















class LinMap(ConfigLayer):
  """
  linear projection from one space to another
  """
  def __init__(self, width, dim, *args, **kwargs):
    super(LinMap, self).__init__(*args, **kwargs)
    # config copy
    self.width = width
    self.dim   = dim
    # construct
    self.flt = tf.keras.layers.Flatten()
    self.map = tf.keras.layers.Dense(units=self.width * self.dim,
                                     use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     bias_initializer=self.bias_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer,
                                     kernel_constraint=self.kernel_constraint,
                                     bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    bs = tf.shape(inputs)[0]                            # get batch size
    x  = self.flt(inputs)                               # flatten inputs
    x  = self.map(x)                                    # linear map
    x  = tf.reshape(x, shape=(bs,self.width,self.dim))  # reshape output space
    return x

  def get_config(self):
    config = super(LinMap, self).get_config()
    config.update({
      'width' : self.width,
      'dim'   : self.dim,
    })
    return config


class PointwiseLinMap(ConfigLayer):
  """
  point-wise linear projection from one space to another
  """
  def __init__(self, out_dim, *args, **kwargs):
    super(PointwiseLinMap, self).__init__(*args, **kwargs)
    # config copy
    self.out_dim = out_dim
    # construct
    self.map = tf.keras.layers.Dense(units=self.out_dim,
                                     use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     bias_initializer=self.bias_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer,
                                     kernel_constraint=self.kernel_constraint,
                                     bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    return self.map(inputs)

  def get_config(self):
    config = super(PointwiseLinMap, self).get_config()
    config.update({
      'out_dim' : self.out_dim,
    })
    return config



class LayerNormLinMap(LinMap):
  """
  layer-normalized linear projection from one space to another
  """
  def __init__(self, *args, **kwargs):
    super(LayerNormLinMap, self).__init__(*args, **kwargs)
    self.lnm = tf.keras.layers.LayerNormalization(axis=(-2,-1))
    return

  def call(self, inputs):
    x = super(LayerNormLinMap, self).call(inputs)
    return self.lnm(x)


class StochasticLinMap(LayerNormLinMap):
  """
  linear projection with gaussian random noise
  """
  def __init__(self, *args, **kwargs):
    super(StochasticLinMap, self).__init__(*args, **kwargs)
    """
    self.nmap = tf.keras.layers.Dense(units=self.width * self.dim,
                                     use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     bias_initializer=self.bias_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer,
                                     kernel_constraint=self.kernel_constraint,
                                     bias_constraint=self.bias_constraint)
    """
    return

  def call(self, inputs):
    x = super(StochasticLinMap, self).call(inputs) # project labels to data
    bs = tf.shape(inputs)[0]                       # batch size
    n  = tf.random.normal(shape=(bs,self.width,self.dim)) # random noise
    n  = tf.sort(n, axis=-2, direction='DESCENDING')      # sort
    #n  = self.nmap(n)                              # project noise to data
    #n  = tf.reshape(n, shape=(bs,self.width,self.dim))  # reshape
    out = tf.concat([x,n], -1)                          # concat labels, noise
    return out


class NormalizedResidualAttention(ConfigLayer):
  def __init__(self,
               attn_hds,
               key_dim,
               *args,
               **kwargs):
    super(NormalizedResidualAttention, self).__init__(*args, **kwargs)
    # config copy
    self.attn_hds = attn_hds
    self.key_dim  = key_dim
    # construct
    self.lnm = tf.keras.layers.LayerNormalization(axis=(-2,-1))
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.attn_hds,
                                    key_dim=self.key_dim,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    x = self.lnm(inputs)
    x = self.mha(x,x)
    return inputs + x


class NormalizedResidualFeedForward(ConfigLayer):
  def __init__(self,
               latent_dim,
               *args,
               **kwargs):
    super(NormalizedResidualFeedForward, self).__init__(*args, **kwargs)
    # config copy
    self.latent_dim = latent_dim
    # construct
    self.lnm = tf.keras.layers.LayerNormalization(axis=(-2,-1))
    self.ff1 = tf.keras.layers.Dense(units=self.latent_dim,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    self.ff2 = tf.keras.layers.Dense(units=self.latent_dim,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    x = self.lnm(inputs)
    x = self.ff1(x)
    x = tf.nn.leaky_relu(x)
    x = self.ff2(x)
    return inputs + x

class TransBlock(ConfigLayer):
  """
  self-attention tranformer block with residual connections
  """
  def __init__(self,
               latent_dim,
               attn_hds,
               key_dim,
               *args,
               **kwargs):
    super(TransBlock, self).__init__(*args, **kwargs)
    # config copy
    self.latent_dim = latent_dim
    self.attn_hds   = attn_hds
    self.key_dim    = key_dim
    # construct
    self.attn = NormalizedResidualAttention(attn_hds=self.attn_hds,
                                    key_dim=self.key_dim,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    self.ffwd = NormalizedResidualFeedForward(latent_dim=self.latent_dim,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    x = self.attn(inputs)
    x = self.ffwd(x)
    return x

  def get_config(self):
    config = super(TransBlock, self).get_config()
    config.update({
      'latent_dim' : self.latent_dim,
      'attn_hds'   : self.attn_hds,
      'key_dim'    : self.key_dim,
    })
    return config


class TransUpsamplBlock(TransBlock):
  """
  self-attention transformer block with average upsampling
  """
  def __init__(self,
               *args,
               **kwargs):
    super(TransUpsamplBlock, self).__init__(*args, **kwargs)
    self.upspl = tf.keras.layers.UpSampling1D(size=2)
    self.avepl = tf.keras.layers.AveragePooling1D(pool_size=3,
                                                  strides=1,
                                                  padding='same')
    return

  def call(self, inputs):
    x = super(TransUpsamplBlock, self).call(inputs)
    # upsample
    x = self.upspl(x)
    x = self.avepl(x)
    return x


class Noisify(Layer):
  """
  adds adatptive gaussian noise to a tensor
  """
  def __init__(self,
               *args,
               **kwargs):
    super(Noisify, self).__init__(*args, **kwargs)
    return

  def build(self, input_shape):
    dwdth = input_shape[1]  # data width
    ltntd = input_shape[2]  # latent dimension
    self.n = self.add_weight(shape=(dwdth,ltntd),
                             initializer=tf.keras.initializers.Zeros(),
                             trainable=True,
                             constraint=tf.keras.constraints.NonNeg())
    return

  def call(self, inputs):
    a = tf.random.normal(shape=tf.shape(inputs)) * (self.n + 1.0e-5)
    return inputs + a


class PointwiseLinNoisify(ConfigLayer):
  """
  adaptive gaussian noise scaling based on input
  """
  def __init__(self, *args, **kwargs):
    super(PointwiseLinNoisify, self).__init__(*args, **kwargs)
    self.flt = tf.keras.layers.Flatten()
    return

  def build(self, input_shape):
    self.map = tf.keras.layers.Dense(units=input_shape[-2] * input_shape[-1],
                                     activation=tf.keras.activations.relu,
                                     use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     bias_initializer=self.bias_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer,
                                     kernel_constraint=self.kernel_constraint,
                                     bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    s  = self.map(self.flt(inputs)) + 0.01
    s  = tf.reshape(s, shape=tf.shape(inputs))
    n  = tf.random.normal(shape=tf.shape(inputs)) * s
    return inputs + n










class LinGausSamp(ConfigLayer):
  """
  independent linear projection of labels to mean, stdev; gaussian sampling
  """
  def __init__(self, width, *args, **kwargs):
    super(LinGausSamp, self).__init__(*args, **kwargs)
    # config copy
    self.width = width
    # construct
    self.flat = tf.keras.layers.Flatten()
    self.mean = tf.keras.layers.Dense(units=width,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    self.stdv = tf.keras.layers.Dense(units=width,
                                    activation=tf.keras.activations.sigmoid,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    bs  = tf.shape(inputs)[0]
    x   = self.flat(inputs)
    m   = self.mean(x)
    s   = self.stdv(x)
    d = tf.random.normal(shape=(bs,self.width), mean=m, stddev=s)
    d = tf.reshape(d, shape=(bs,self.width,1))
    return d

  def get_config(self):
    config = super(LinGausSamp, self).get_config()
    config.update({
      'width' : self.width,
    })
    return config


class GenStart(ConfigLayer):
  def __init__(self, width, *args, **kwargs):
    super(GenStart, self).__init__(*args, **kwargs)
    # config copy
    self.width = width
    # constructor
    self.query = LinGausSamp(width=width,
                        use_bias=self.use_bias,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        bias_constraint=self.bias_constraint)
    self.key   = LinMap(width=width,
                        dim=1,
                        use_bias=self.use_bias,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        bias_constraint=self.bias_constraint)
    self.value = LinMap(width=width,
                        dim=1,
                        use_bias=self.use_bias,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    q = self.query(inputs)
    k = self.key(inputs)
    v = self.value(inputs)
    return (q,k,v)

  def get_config(self):
    config = super(GenStart, self).get_config()
    config.update({
      'width' : width,
    })
    return config


class CrossMultHdAttn(ConfigLayer):
  def __init__(self, width, heads, *args, **kwargs):
    super(CrossMultHdAttn, self).__init__(*args, **kwargs)
    # config copy
    self.width = width
    self.heads = heads
    # construct
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=2)
    self.msk = tf.stack((tf.zeros(shape=(1,self.width)),
                          tf.ones(shape=(1,self.width))), axis=-1)
    return

  def call(self, inputs):
    q = inputs[0]
    k = inputs[1]
    v = inputs[2]
    r = self.mha(q,v,k) * self.msk # mask out 'position' from r
    return (q+r, k, v)

  def get_config(self):
    config = super(CrossMultHdAttn, self).get_config()
    config.update({
      'width' : width,
      'heads' : heads,
    })
    return config


class Const(Layer):
  def __init__(self, width):
    super(Const, self).__init__()
    self.width = width
    self.wts = self.add_weight(shape=(1,self.width,),
                               initializer='glorot_normal',
                               trainable=True)

  def call(self, inputs):
    bs = tf.shape(inputs)[0]
    x = tf.convert_to_tensor(self.wts)
    x = tf.tile(x, multiples=(bs,1))
    return x








## ENCODER CLASSES #############################################################

class EncodeGen(EncodeLayer):
  """
  encodes labels for generator
  """
  def __init__(self, *args, **kwargs):
    super(EncodeGen, self).__init__(*args, **kwargs)
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
    self.lpr = tf.keras.layers.Dense(units=1,
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
    self.mean = tf.keras.layers.Dense(units=1,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    self.stdv = tf.keras.layers.Dense(units=1,
                                    use_bias=self.use_bias,
                                    activation=tf.keras.activations.softplus,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
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
    mn = self.mean(inputs)  # project noise means
    sd = self.stdv(inputs)  # project noise stdvs
    # generate masked random vector affecting only data dimension
    rv = tf.random.normal(mean=mn,
                          stddev=sd,
                          shape=(bs,self.width,1)) * self.mask
    return inputs + rv

## CONDITIONAL GENERATOR BUILD FUNCTION ########################################

def CondGen1D(input_shape, width, attn_hds=4, nattnblocks=2):
  """
  construct generator using functional API
  """
  ## input encoding
  start_width = 32
  inputs = tf.keras.Input(shape=input_shape, name='lbin')
  output = EncodeGen(width=start_width, name='encd')(inputs)
  ## upsampling subnet
  n_upsampl_blks = int(math.log2(width) - math.log2(start_width))
  curr_width = start_width
  for i in range(n_upsampl_blks):
    output = PosMaskedMHABlock(width=curr_width,
                               dim=3,
                               heads=attn_hds,
                               name='mau{}'.format(i))(output)
    output = AveUpsample(name='ups{}'.format(i))(output)
    curr_width *= 2
    output = DataNoise(width=curr_width, name='noi{}'.format(i))(output)
  ## self-attention subnet
  for i in range(nattnblocks):
    output = PosMaskedMHABlock(width=width,
                              dim=3,
                              heads=attn_hds,
                              name='mha{}'.format(i))(output)
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
