from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import initializers, regularizers, constraints, Model
from tensorflow.keras.layers import Layer
import tensorflow as tf

from cond_generator_1d import ConfigLayer, LinMap, CrossMultHdAttn


class DisStart(ConfigLayer):
  """
  discriminator start - linear project labels to input data space
  """
  def __init__(self, width, *args, **kwargs):
    super(DisStart, self).__init__(*args, **kwargs)
    # config copy
    self.width = width
    # construct
    self.flt    = tf.keras.layers.Flatten()
    self.lblprj = LinMap(width=width,
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
    bs = tf.shape(inputs[0])[0]
    # query
    q = inputs[0]
    q = tf.reshape(q, shape=(bs,self.width,1))
    # value
    v = inputs[1]
    v = tf.reshape(v, shape=(bs,self.width,1))
    # key
    k = self.lblprj(self.flt(inputs[2]))
    return (q,k,v)

  def get_config(self):
    config = super(DisStart, self).get_config()
    config.update({
      'width' : self.width,
    })
    return config


def CondDis1D(data_width, label_width, attn_hds=8):
  """
  construct a discriminator using functional API
  """
  in1 = tf.keras.Input(shape=(data_width,),  name='in1')
  in2 = tf.keras.Input(shape=(data_width,),  name='in2')
  in3 = tf.keras.Input(shape=(label_width,), name='in3')
  out = DisStart(width=data_width)((in1,in2,in3))
  out = CrossMultHdAttn(width=data_width, heads=attn_hds)(out)

  """
  nblocks = 2
  key_dim = latent_dim // 2

  # calculate real data and label shapes from width * pack_dim
  dta_shap = (data_width,pack_dim,)
  lbl_shap = (label_width,pack_dim,)
  # data and label inputs
  dinput  = tf.keras.Input(shape=dta_shap, name='dta_in')
  linput  = tf.keras.Input(shape=lbl_shap, name='lbl_in')

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

  ## construct model
  # data input map

  #doutput = LayerNormLinMap(data_width, latent_dim, name='dtamap')(dinput)
  doutput = PointwiseLinMap(latent_dim, name='dtamap')(dinput)
  #doutput = tf.keras.layers.LayerNormalization(axis=(-2,-1),
  #                                             name='dtanrm')(doutput)
  # label input map

  loutput = LinMap(data_width, latent_dim, name='lblmap')(linput)
  #loutput = tf.keras.layers.LayerNormalization(axis=(-2,-1),
  #                                             name='lblnrm1')(loutput)
  #loutput = PointwiseLinMap(latent_dim, name='lblprj')(loutput)
  #loutput = tf.keras.layers.LayerNormalization(axis=(-2,-1),
  #                                             name='lblnrm2')(loutput)
  # combine data and label maps
  output = tf.keras.layers.Concatenate(name='dtalbl')((doutput,loutput))
  #latent_dim *= 2
  # transformer blocks
  #nblocks = 2
  #for i in range(nblocks):
  #  output = TransBlock(latent_dim=latent_dim,
  #                      attn_hds=attn_hds,
  #                      key_dim=latent_dim,
  #                      name='trblk{}'.format(i))(output)
  # decision layers
  """

  out = tf.keras.layers.Flatten()(out[0])
  out = tf.keras.layers.Dense(units=256)(out)
  out = tf.keras.layers.LeakyReLU()(out)
  out = tf.keras.layers.Dense(units=256)(out)
  out = tf.keras.layers.LeakyReLU()(out)
  out = tf.keras.layers.Dense(units=1, name='output')(out)
  return Model(inputs=(in1,in2,in3), outputs=out)


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
