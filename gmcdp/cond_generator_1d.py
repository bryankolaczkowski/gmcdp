from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import initializers, regularizers, constraints, Model
from tensorflow.keras.layers import Layer
import tensorflow as tf

from wrappers import SpecNorm


class ConfigLayer(Layer):
  """
  base class for layers having sub-layers
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


class LinMap(ConfigLayer):
  """
  linear projection from one space to another
  """
  def __init__(self,
               width,
               dim,
               *args,
               **kwargs):
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
    """
    learned linear transform to (bs,self.width,self.dim)
    """
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
    self.nmap = tf.keras.layers.Dense(units=self.width * self.dim,
                                     use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     bias_initializer=self.bias_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer,
                                     kernel_constraint=self.kernel_constraint,
                                     bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    x = super(StochasticLinMap, self).call(inputs) # project labels to data
    bs = tf.shape(inputs)[0]                       # batch size
    n  = tf.random.normal(shape=(bs,self.width))   # random noise
    n  = self.nmap(n)                              # project noise to data
    n  = tf.reshape(n, shape=(bs,self.width,self.dim))  # reshape
    out = tf.concat([x,n], -1)                          # concat labels, noise
    return out


class PointwiseLinMap(ConfigLayer):
  """
  implements point-wise linear projection from one space to another
  """
  def __init__(self,
               out_dim,
               *args,
               **kwargs):
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
    """
    learned linear transform to (bs,input_width,self.out_dim)
    """
    return self.map(inputs) # linear map

  def get_config(self):
    config = super(PointwiseLinMap, self).get_config()
    config.update({
      'out_dim' : self.out_dim,
    })
    return config


class NormalizedResidualAttention(ConfigLayer):
  """
  layer-normalized multi-head attention with residual connection
  """
  def __init__(self,
               latent_dim,
               attn_hds,
               key_dim,
               *args,
               **kwargs):
    super(NormalizedResidualAttention, self).__init__(*args, **kwargs)
    # config copy
    self.latent_dim = latent_dim
    self.attn_hds   = attn_hds
    self.key_dim    = key_dim
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
    x = self.lnm(inputs)  # layer normalization
    x = self.mha(x,x)     # multi-head attention
    return inputs + x     # residual connection

  def get_config(self):
    config = super(NormalizedResidualAttention, self).get_config()
    config.update({
      'latent_dim' : self.latent_dim,
      'attn_hds'   : self.attn_hds,
      'key_dim'    : self.key_dim,
    })
    return config


class NormalizedResidualCrossAttention(NormalizedResidualAttention):
  """
  layer-normalized multi-head cross-attention with residual connection
  """
  def __init__(self, *args, **kwargs):
    super(NormalizedResidualCrossAttention, self).__init__(*args, **kwargs)
    return

  def call(self, inputs):
    qry = inputs[0]
    val = inputs[1]
    nqy = self.lnm(qry)       # layer normalization on query
    x   = self.mha(nqy, val)  # cross-attention
    return qry + x


class NormalizedResidualFeedForward(ConfigLayer):
  """
  layer-normalized pointwise feed-forward with residual connection
  """
  def __init__(self, latent_dim, *args, **kwargs):
    super(NormalizedResidualFeedForward, self).__init__(*args, **kwargs)
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
    x = self.lnm(inputs)  # layer normalization
    x = self.ff1(x)       # linear pointwise feed-forward 1
    x = tf.nn.gelu(x)     # nonlinear activation
    x = self.ff2(x)       # linear pointwise feed-forward 2
    return inputs + x     # residual connection

  def get_config(self):
    config = super(NormalizedResidualFeedForward, self).get_config()
    config.update({
      'latent_dim' : self.latent_dim,
    })
    return config


class TransUpsamplBlock(ConfigLayer):
  """
  dual self-attention transformer blocks with average upsampling
  """
  def __init__(self,
               latent_dim,
               attn_hds,
               key_dim,
               *args,
               **kwargs):
    super(TransUpsamplBlock, self).__init__(*args, **kwargs)
    # config copy
    self.latent_dim = latent_dim
    self.attn_hds   = attn_hds
    self.key_dim    = key_dim
    # construct
    self.attn1 = NormalizedResidualAttention(latent_dim=self.latent_dim,
                                             attn_hds=self.attn_hds,
                                             key_dim=self.key_dim)
    self.ffwd1 = NormalizedResidualFeedForward(latent_dim=self.latent_dim)
    self.attn2 = NormalizedResidualAttention(latent_dim=self.latent_dim,
                                             attn_hds=self.attn_hds,
                                             key_dim=self.key_dim)
    self.ffwd2 = NormalizedResidualFeedForward(latent_dim=self.latent_dim)
    self.upspl = tf.keras.layers.UpSampling1D(size=2)
    self.avepl = tf.keras.layers.AveragePooling1D(pool_size=3,
                                                  strides=1,
                                                  padding='same')
    return

  def call(self, inputs):
    # transformer 1
    x = self.attn1(inputs)
    x = self.ffwd1(x)
    # transformer 2
    x = self.attn2(x)
    x = self.ffwd2(x)
    # upsample
    x = self.upspl(x)
    x = self.avepl(x)
    # noise

    return x

  def get_config(self):
    config = super(TransUpsamplBlock, self).get_config()
    config.update({
      'latent_dim' : self.latent_dim,
      'attn_hds'   : self.attn_hds,
      'key_dim'    : self.key_dim,
    })
    return config












class CrossAttnTransBlock(ConfigLayer):
  """
  (data,label) transformer with cross-attention
  """
  def __init__(self,
               latent_dim,
               attn_hds,
               key_dim,
               *args,
               **kwargs):
    super(CrossAttnTransBlock, self).__init__(*args, **kwargs)
    # config copy
    self.latent_dim = latent_dim
    self.attn_hds   = attn_hds
    self.key_dim    = key_dim
    # construct
    # label path
    self.lbl_self_attn = NormalizedResidualAttention(latent_dim=self.latent_dim,
                                                  attn_hds=self.attn_hds,
                                                  key_dim=self.key_dim)
    self.lbl_feed_frwd = NormalizedResidualFeedForward(\
                                                  latent_dim=self.latent_dim)
    # data path
    self.dta_self_attn = NormalizedResidualAttention(latent_dim=self.latent_dim,
                                                  attn_hds=self.attn_hds,
                                                  key_dim=self.key_dim)
    self.dta_crss_attn = NormalizedResidualCrossAttention(\
                                                  latent_dim=self.latent_dim,
                                                  attn_hds=self.attn_hds,
                                                  key_dim=self.key_dim)
    self.dta_feed_frwd = NormalizedResidualFeedForward(\
                                                  latent_dim=self.latent_dim)
    return

  def call(self, inputs):
    dta = inputs[0]
    lbl = inputs[1]
    # label path
    lbl = self.lbl_self_attn(lbl)
    lbl = self.lbl_feed_frwd(lbl)
    # data path
    dta = self.dta_self_attn(dta)
    dta = self.dta_crss_attn((dta, lbl))
    dta = self.dta_feed_frwd(dta)
    return (dta, lbl)

  def get_config(self):
    config = super(CrossAttnTransBlock, self).get_config()
    config.update({
      'latent_dim' : self.latent_dim,
      'attn_hds'   : self.attn_hds,
      'key_dim'    : self.key_dim,
    })
    return config


class AveUpsamplCrossAttnTransBlock(CrossAttnTransBlock):
  """
  transformer->upsample block using average upsampling
  """
  def __init__(self, *args, **kwargs):
    super(AveUpsamplCrossAttnTransBlock, self).__init__(*args, **kwargs)
    self.upspl = tf.keras.layers.UpSampling1D(size=2)
    self.avepl = tf.keras.layers.AveragePooling1D(pool_size=3,
                                                  strides=1,
                                                  padding='same')
    return

  def call(self, inputs):
    x = super(AveUpsamplCrossAttnTransBlock, self).call(inputs)
    dta = x[0]
    lbl = x[1]
    # upsample data
    dta = self.upspl(dta)   # duplicative upsampling
    dta = self.avepl(dta)   # average to interpolate values
    # upsample labels
    lbl = self.upspl(lbl)   # duplicative upsampling
    lbl = self.avepl(lbl)   # average to interpolate values
    return (dta,lbl)


class TransBlock(ConfigLayer):
  """
  implements a transformer + feed-forward block
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
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.attn_hds,
                                    key_dim=self.key_dim,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
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
    self.ln1 = tf.keras.layers.LayerNormalization(axis=(-2,-1))
    self.ln2 = tf.keras.layers.LayerNormalization(axis=(-2,-1))
    return

  def call(self, inputs):
    """
    inputs is (bs,width,latent_dim)
    """
    ## sub-block 1 - attention
    x = self.ln1(inputs)  # layer normalization
    x = self.mha(x,x)     # multi-head attention
    x = inputs + x        # residual connection
    ## sub-block 2 - feed-forward
    y = self.ln2(x)       # layer normalization
    y = self.ff1(y)       # linear pointwise feed-forward 1
    y = tf.nn.gelu(y)     # nonlinear activation
    y = self.ff2(y)       # linear pointwise feed-forward 2
    x = x + y             # residual connection
    return x

  def get_config(self):
    config = super(TransBlock, self).get_config()
    config.update({
      'latent_dim' : self.latent_dim,
      'attn_hds'   : self.attn_hds,
      'key_dim'    : self.key_dim,
    })
    return config


class SpecNormTransBlock(TransBlock):
  """
  implements a spectral-normalized transformer + feed-forward block
  """
  def __init__(self, *args, **kwargs):
    super(SpecNormTransBlock, self).__init__(*args, **kwargs)
    self.ff1 = SpecNorm(self.ff1)
    self.ff2 = SpecNorm(self.ff2)
    return


class UpsamplTransBlock(TransBlock):
  """
  base class for transformer->upsample->noise blocks
  """
  def __init__(self, *args, **kwargs):
    super(UpsamplTransBlock, self).__init__(*args, **kwargs)
    # construct
    self.nscale = tf.keras.layers.Dense(units=1,
                                     use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     bias_initializer=self.bias_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer,
                                     kernel_constraint=self.kernel_constraint,
                                     bias_constraint=self.bias_constraint)
    return

  def _upsample(self, inputs):
    """
    override private method to upsample inputs
    """
    raise NotImplementedError

  def call(self, inputs):
    x = super(UpsamplTransBlock, self).call(inputs)   # transformer block
    x = self._upsample(x)                             # upsampling
    n = tf.random.normal(shape=tf.shape(x))           # gaussian noise
    x = x + n * self.nscale(x)                        # scale noise
    return x


class AveUpsamplTransBlock(UpsamplTransBlock):
  """
  transformer->upsample->noise block using average upsampling
  """
  def __init__(self, *args, **kwargs):
    super(AveUpsamplTransBlock, self).__init__(*args, **kwargs)
    self.upspl = tf.keras.layers.UpSampling1D(size=2)
    self.avepl = tf.keras.layers.AveragePooling1D(pool_size=3,
                                                  strides=1,
                                                  padding='same')
    return

  def _upsample(self, inputs):
    """
    average-based upsampling
    """
    x = self.upspl(inputs)    # duplicative upsampling
    x = self.avepl(x)         # average to interpolate values
    return x









def CondGen1D(input_shape, width, latent_dim=16, attn_hds=8, start_width=64):
  """
  construct generator using functional API
  """
  # map input to latent space
  inputs = tf.keras.Input(shape=input_shape, name='lblin')
  output = StochasticLinMap(width=start_width,
                            dim=latent_dim,
                            name='linmp')(inputs)
  latent_dim *= 2
  ## transformer->transformer->upsample blocks
  nblocks = (int(width).bit_length()) - (int(start_width).bit_length())
  for i in range(nblocks):
    output = TransUpsamplBlock(latent_dim=latent_dim,
                               attn_hds=attn_hds,
                               key_dim=latent_dim,
                               name='utb_{}'.format(i))(output)
  # map latent space to data space
  output = PointwiseLinMap(out_dim=1, name='plnmp')(output)
  output = tf.keras.layers.Flatten(name='dtout')(output)
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
  mdl.summary(line_length=150)
  out = mdl(lbls)
  print(out)
