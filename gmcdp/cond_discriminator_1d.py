from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import initializers, regularizers, constraints, Model
from tensorflow.keras.layers import Layer
import tensorflow as tf

from wrappers import SpecNorm
from cond_generator_1d import SpecNormTransBlock


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
    self.mapl = SpecNorm(tf.keras.layers.Dense(units=self.width,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint))
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
    return tf.transpose(x, perm=(0,2,1))

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


class SummaryStats(Layer):
  """
  packed input -> summary statistics
  """
  def __init__(self,
               **kwargs):
    super(SummaryStats, self).__init__(**kwargs)
    return

  def call(self, inputs):
    """
    converts a packed input into summary statistics
    """
    mean = tf.math.reduce_mean(inputs, axis=-1, keepdims=True)
    sdev = tf.math.reduce_std( inputs, axis=-1, keepdims=True)
    resi = mean - inputs
    return tf.concat([mean,sdev,resi], axis=-1)

  def get_config(self):
    config = super(SummaryStats, self).get_config()
    return config


def CondDis1D(data_width,
              label_width,
              pack_dim=4,
              attn_hds=8):
  """
  construct a discriminator using functional API
  """
  # calculate real data and label shapes from width * pack_dim
  dta_shap = (data_width,pack_dim,)
  lbl_shap = (label_width,pack_dim,)
  ## construct model
  # data input
  dinput  = tf.keras.Input(shape=dta_shap, name='dta_in')
  # project labels to data
  linput  = tf.keras.Input(shape=lbl_shap, name='lbl_in')
  loutput = PackedInputMap(data_width, name='lblmap')(linput)
  # combine data and projected labels
  output = tf.keras.layers.Concatenate(name='dtalbl')((dinput,loutput))
  # convert data to summary statistics
  #output = SummaryStats()(output)
  output = SpecNorm(tf.keras.layers.Dense(units=pack_dim*2,
                                 kernel_initializer='glorot_normal'),
                                 name='linprj')(output)
  # transformer blocks
  output = SpecNormTransBlock(latent_dim=pack_dim*2,
                              attn_hds=attn_hds,
                              key_dim=pack_dim,
                              name='trns_0')(output)
  # sequence model
  #output = tf.keras.layers.Bidirectional(
  #                            tf.keras.layers.LSTM(units=32,
  #                                kernel_initializer='glorot_normal'),
  #                            name='seqmodl')(output)
  # decision layers
  output = tf.keras.layers.Flatten(name='outflt')(output)
  #output = SpecNorm(tf.keras.layers.Dense(units=pack_dim,),
  #                                        name='desc_0')(output)
  #output = tf.keras.layers.LeakyReLU(alpha=0.2, name='reluac')(output)
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
  gen.summary()
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
  dis.summary()
  out = dis((dta,lbl))
  print(out)
