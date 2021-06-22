from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import initializers, regularizers, constraints, Model
from tensorflow.keras.layers import Layer
import tensorflow as tf

from cond_generator_1d import ConfigLayer, LinMap, PointwiseLinMap, \
                              EncoderBlock, DecoderBlock, LayerNormLinMap, \
                              TransBlock, NormalizedResidualAttention


class PackedInputMap(Layer):
  """
  independent linear map for packed input
  """
  def __init__(self,
               width,
               use_bias=True,
               kernel_initializer='glorot_normal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(PackedInputMap, self).__init__(**kwargs)
    # config copy
    self.width              = width
    self.use_bias           = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer   = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer   = regularizers.get(bias_regularizer)
    self.kernel_constraint  = constraints.get(kernel_constraint)
    self.bias_constraint    = constraints.get(bias_constraint)
    # construct
    self.mapl = tf.keras.layers.Dense(units=self.width,
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
    maps data to latent space embedding (bs,width,latent_dim)
    """
    # transpose (bs,width,channels) to (bs,channels,width)
    x = tf.transpose(inputs, perm=(0,2,1))
    # independently map each channel's data using linear map
    x = self.mapl(x)
    # un-transpose (bs,channels,width) to (bs,width,channels)
    x = tf.transpose(x, perm=(0,2,1))
    return x

  def get_config(self):
    config = super(LabelDataMap, self).get_config()
    config.update({
      'width'              : self.width,
      'use_bias'           : self.use_bias,
      'kernel_initializer' : initializers.serialize(self.kernel_initializer),
      'bias_initializer'   : initializers.serialize(self.bias_initializer),
      'kernel_regularizer' : regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer'   : regularizers.serialize(self.bias_regularizer),
      'kernel_constraint'  : constraints.serialize(self.kernel_constraint),
      'bias_constraint'    : constraints.serialize(self.bias_constraint),
    })
    return config





class DisStart(ConfigLayer):
  """
  discriminator starting layer
  """
  def __init__(self, width, dim, *args, **kwargs):
    super(DisStart, self).__init__(*args, **kwargs)
    # config copy
    self.width = width
    self.dim   = dim
    # constructor
    self.dtamap = PointwiseLinMap(self.dim,
                         use_bias=self.use_bias,
                         kernel_initializer=self.kernel_initializer,
                         bias_initializer=self.bias_initializer,
                         kernel_regularizer=self.kernel_regularizer,
                         bias_regularizer=self.bias_regularizer,
                         kernel_constraint=self.kernel_constraint,
                         bias_constraint=self.bias_constraint)
    self.lblmap = LinMap(self.width,
                         self.dim,
                         use_bias=self.use_bias,
                         kernel_initializer=self.kernel_initializer,
                         bias_initializer=self.bias_initializer,
                         kernel_regularizer=self.kernel_regularizer,
                         bias_regularizer=self.bias_regularizer,
                         kernel_constraint=self.kernel_constraint,
                         bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    dta = self.dtamap(inputs[0])
    lbl = self.lblmap(inputs[1])
    return (dta,lbl)

  def get_config(self):
    config = super(GenStart, self).get_config()
    config.update({
      'width' : self.width,
      'dim'   : self.dim,
    })
    return config


def CondDis1D(data_width, label_width, pack_dim=4, latent_dim=8, attn_hds=4):
  """
  construct a discriminator using functional API
  """
  nblocks = 2
  key_dim = latent_dim // 2

  # calculate real data and label shapes from width * pack_dim
  dta_shap = (data_width,pack_dim,)
  lbl_shap = (label_width,pack_dim,)
  # data and label inputs
  dinput  = tf.keras.Input(shape=dta_shap, name='dta_in')
  linput  = tf.keras.Input(shape=lbl_shap, name='lbl_in')

  """
  output = DisStart(data_width,
                    latent_dim*pack_dim,
                    name='disst')((dinput, linput))
  # encoder blocks
  for i in range(nblocks):
    output = EncoderBlock(latent_dim=latent_dim*pack_dim,
                          attn_hds=attn_hds,
                          key_dim=key_dim,
                          name='enc{}'.format(i))(output)
  # decoder blocks
  for i in range(nblocks):
    output = DecoderBlock(latent_dim=latent_dim*pack_dim,
                          attn_hds=attn_hds,
                          key_dim=key_dim,
                          name='dec{}'.format(i))(output)
  """

  ## construct model
  # data input map

  doutput = LayerNormLinMap(data_width, latent_dim, name='dtamap')(dinput)
  #doutput = PointwiseLinMap(latent_dim, name='dtamap')(dinput)
  #doutput = tf.keras.layers.LayerNormalization(axis=(-2,-1),
  #                                             name='dtanrm')(doutput)
  # label input map

  loutput = LayerNormLinMap(data_width, latent_dim, name='lblmap')(linput)
  #loutput = tf.keras.layers.LayerNormalization(axis=(-2,-1),
  #                                             name='lblnrm1')(loutput)
  #loutput = PointwiseLinMap(latent_dim, name='lblprj')(loutput)
  #loutput = tf.keras.layers.LayerNormalization(axis=(-2,-1),
  #                                             name='lblnrm2')(loutput)
  # combine data and label maps
  output = tf.keras.layers.Concatenate(name='dtalbl')((doutput,loutput))
  latent_dim *= 2
  # transformer blocks
  nblocks = 2
  for i in range(nblocks):
    output = TransBlock(latent_dim=latent_dim,
                        attn_hds=attn_hds,
                        key_dim=latent_dim,
                        name='trblk{}'.format(i))(output)
  # decision layers

  #output = tf.keras.layers.Concatenate(name='conct')(output)
  output = tf.keras.layers.Flatten(name='flt')(output)
  output = tf.keras.layers.Dense(units=1, name='output')(output)
  return Model(inputs=(dinput,linput), outputs=output)


if __name__ == '__main__':
  """
  module example
  """
  import sys
  sys.path.append("../tests")
  import test_data_generator
  from cond_generator_1d import CondGen1D

  ndata = 16

  # generate simulated data and labels
  data,lbls = test_data_generator.gen_dataset(ndata, plot=False)
  print(data,lbls)

  # create a little 'generator model' that just maps the label vector
  # to data space using a linear map
  input_shape  = tf.shape(lbls)
  output_shape = tf.shape(data)
  gen = CondGen1D((input_shape[1],), output_shape[1])
  gen.summary(positions=[0.3, 0.75, 0.85, 1.0])
  out = gen(lbls)
  print(out)

  # convert generator output to 'packed' discriminator
  pack_dim = 4
  ## pack data
  dta = out[0] # data  shape is (bs, width)
  # get new packed batch size
  bs  = tf.shape(dta)[0] // pack_dim
  dwd = tf.shape(dta)[1]
  dta = tf.reshape(dta, shape=(bs,dwd,-1))
  ## pack labels
  lbl = out[1] # label shape is (bs, labels)
  lwd = tf.shape(lbl)[1]
  lbl = tf.reshape(lbl, shape=(bs,lwd,-1))

  # create a little 'discriminator model'
  dis = CondDis1D(output_shape[1], input_shape[1], pack_dim=pack_dim)
  dis.summary(positions=[0.3, 0.75, 0.85, 1.0])
  out = dis((dta,lbl))
  print(out)
