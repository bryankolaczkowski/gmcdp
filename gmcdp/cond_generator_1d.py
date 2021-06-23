from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import initializers, regularizers, constraints, Model
from tensorflow.keras.layers import Layer
import tensorflow as tf


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


## BEG LINEAR MAPS #############################################################

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

## END LINEAR MAPS #############################################################



### BEG TRANSFORMER CLASSES ####################################################

class DataSelfAttn(ConfigLayer):
  """
  data multi-head self-attention with residual connection
  """
  def __init__(self,
               attn_hds,
               key_dim,
               *args,
               **kwargs):
    super(DataSelfAttn, self).__init__(*args, **kwargs)
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
    x = inputs[0]           # data self-attention
    x = self.lnm(x)         # layer pre-normalization
    x = self.mha(x,x)       # multi-head self-attention
    return (inputs[0]+x, inputs[1]) # residual connection

  def get_config(self):
    config = super(DataSelfAttn, self).get_config()
    config.update({
      'attn_hds' : self.attn_hds,
      'key_dim'  : self.key_dim,
    })
    return config


class LabelSelfAttn(DataSelfAttn):
  """
  label multi-head self-attention with residual connection
  """
  def __init__(self, *args, **kwargs):
    super(LabelSelfAttn, self).__init__(*args, **kwargs)
    return

  def call(self, inputs):
    x = inputs[1]           # label self-attention
    x = self.lnm(x)         # layer pre-normalization
    x = self.mha(x,x)       # multi-head self-attention
    return (inputs[0], inputs[1]+x) # residual connection


class DataCrossAttn(DataSelfAttn):
  """
  data multi-head cross-attention with residual connection
  """
  def __init__(self, *args, **kwargs):
    super(DataCrossAttn, self).__init__(*args, **kwargs)
    return

  def call(self, inputs):
    x = inputs[0]             # data is query
    x = self.lnm(x)           # layer pre-normalization of data
    x = self.mha(x,inputs[1]) # multi-head cross-attention
    return (inputs[0]+x, inputs[1]) # residual connection


class DataFeedForward(ConfigLayer):
  """
  data layer-normalized pointwise feed-forward with residual connection
  """
  def __init__(self, latent_dim, *args, **kwargs):
    super(DataFeedForward, self).__init__(*args, **kwargs)
    # config copy
    self.latent_dim = latent_dim
    # construct
    self.lnm = tf.keras.layers.LayerNormalization(axis=(-2,-1))
    self.ff1 = tf.keras.layers.Dense(units=self.latent_dim*2,
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
    x = inputs[0]         # data
    x = self.lnm(x)       # layer normalization
    x = self.ff1(x)       # linear pointwise feed-forward 1
    x = tf.nn.gelu(x)     # nonlinear activation
    x = self.ff2(x)       # linear pointwise feed-forward 2
    return (inputs[0]+x, inputs[1]) # residual connection

  def get_config(self):
    config = super(DataFeedForward, self).get_config()
    config.update({
      'latent_dim' : self.latent_dim,
    })
    return config


class LabelFeedForward(DataFeedForward):
  """
  label layer-normalized pointwise feed-forward with residual connection
  """
  def __init__(self, *args, **kwargs):
    super(LabelFeedForward, self).__init__(*args, **kwargs)

  def call(self, inputs):
    x = inputs[1]         # data
    x = self.lnm(x)       # layer normalization
    x = self.ff1(x)       # linear pointwise feed-forward 1
    x = tf.nn.gelu(x)     # nonlinear activation
    x = self.ff2(x)       # linear pointwise feed-forward 2
    return (inputs[0], inputs[1]+x) # residual connection


class EncoderBlock(LabelSelfAttn):
  """
  encoder uses label self-attention; data is passed through
  """
  def __init__(self, latent_dim, *args, **kwargs):
    super(EncoderBlock, self).__init__(*args, **kwargs)
    # config copy
    self.latent_dim = latent_dim
    # construct
    self.ffdblock = LabelFeedForward(latent_dim=self.latent_dim,
                                     use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     bias_initializer=self.bias_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer,
                                     kernel_constraint=self.kernel_constraint,
                                     bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    out = super(EncoderBlock, self).call(inputs)
    out = self.ffdblock(out)
    return out

  def get_config(self):
    config = super(EncoderBlock, self).get_config()
    config.update({
      'latent_dim' : self.latent_dim,
    })
    return config


class DecoderBlock(DataSelfAttn):
  """
  decoder uses data self- and cross-attention; labels are passed through
  """
  def __init__(self, latent_dim, *args, **kwargs):
    super(DecoderBlock, self).__init__(*args, **kwargs)
    # config copy
    self.latent_dim = latent_dim
    # construct
    self.crsblock = DataCrossAttn(attn_hds=self.attn_hds,
                                  key_dim=self.key_dim,
                                  use_bias=self.use_bias,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  bias_regularizer=self.bias_regularizer,
                                  kernel_constraint=self.kernel_constraint,
                                  bias_constraint=self.bias_constraint)
    self.ffdblock = DataFeedForward(latent_dim=self.latent_dim,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    out = super(DecoderBlock, self).call(inputs)
    out = self.crsblock(out)
    out = self.ffdblock(out)
    return out

  def get_config(self):
    config = super(DecoderBlock, self).get_config()
    config.update({
      'latent_dim' : self.latent_dim,
    })
    return config

### END TRANSFORMER CLASSES ####################################################












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
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=1)
    return

  def call(self, inputs):
    q = inputs[0]
    k = inputs[1]
    v = inputs[2]
    v = self.mha(q,v,k)
    return (q+v, k, v)

  def get_config(self):
    config = super(CrossMultHdAttn, self).get_config()
    config.update({
      'width' : width,
      'heads' : heads,
    })
    return config


## CONDITIONAL GENERATOR BUILD FUNCTION ########################################

def CondGen1D(input_shape, width, attn_hds=8):
  """
  construct generator using functional API
  """
  inputs = tf.keras.Input(shape=input_shape, name='lblin')
  #output = GenStart(width=width)(output)
  #output = CrossMultHdAttn(width=width, heads=attn_hds)(output)

  # nonlinear label embedding
  #output = tf.keras.layers.Flatten()(inputs)
  #for i in range(2):
  #  output = tf.keras.layers.Dense(units=64,
  #                                 activation=tf.keras.activations.tanh)(output)
  # simple linear map
  output = LinMap(width=width, dim=1)(inputs)
  output = tf.keras.layers.Flatten(name='fltn')(output)

  """
  output = GenStart(width=width, dim=latent_dim, name='genst')(inputs)
  # encoder blocks
  for i in range(nblocks):
    output = EncoderBlock(latent_dim=latent_dim,
                          attn_hds=attn_hds,
                          key_dim=key_dim,
                          name='enc{}'.format(i))(output)
  # decoder blocks
  for i in range(nblocks):
    output = DecoderBlock(latent_dim=latent_dim,
                          attn_hds=attn_hds,
                          key_dim=key_dim,
                          name='dec{}'.format(i))(output)

  # map input to latent space

  output = StochasticLinMap(width=start_width,
                            dim=latent_dim,
                            name='linmp')(inputs)
  latent_dim *= 2
  ## transformer blocks
  nblocks = (int(width).bit_length()) - (int(start_width).bit_length())
  nblocks = 2
  for i in range(nblocks):
    output = TransBlock(latent_dim=latent_dim,
                        attn_hds=attn_hds,
                        key_dim=latent_dim,
                        name='utb_{}'.format(i))(output)
  # introduce noise
  output = PointwiseLinNoisify(name='noi_{}'.format(i))(output)
  # map latent space to data space
  output = LinMap(width, 1, name='plnmp')(output)
  """

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
  mdl.summary(positions=[0.3, 0.75, 0.85, 1.0])
  out = mdl(lbls)
  print(out)
