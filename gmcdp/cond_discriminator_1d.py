from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import initializers, regularizers, constraints, Model
from tensorflow.keras.layers import Layer
import tensorflow as tf

from cond_generator_1d import EncodeLayer, WidthLayer, PosMaskedMHABlock


class EncodeDis(EncodeLayer):
  """
  encodes data for discriminator
  """
  def __init__(self, *args, **kwargs):
    super(EncodeDis, self).__init__(*args, **kwargs)
    return

  def call(self, inputs):
    dt1 = inputs[0]
    dt2 = inputs[1]
    lbl = inputs[2]
    bs  = tf.shape(lbl)[0]
    pse = tf.tile(self.pos, multiples=(bs,1))   # sequence position encoding
    lpl = self.lpr(self.flt(lbl))               # linear project labels
    return tf.stack((pse, dt1, dt2, lpl), axis=-1)


class DecodeDis(WidthLayer):
  """
  decodes discriminator output to score
  """
  def __init__(self, *args, **kwargs):
    super(DecodeDis, self).__init__(*args, **kwargs)
    # construct
    self.flt = tf.keras.layers.Flatten()
    self.dn1 = tf.keras.layers.Dense(units=self.width,
                                  use_bias=self.use_bias,
                                  activation=tf.keras.activations.tanh,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  bias_regularizer=self.bias_regularizer,
                                  kernel_constraint=self.kernel_constraint,
                                  bias_constraint=self.bias_constraint)
    self.dn2 = tf.keras.layers.Dense(units=self.width,
                                  use_bias=self.use_bias,
                                  activation=tf.keras.activations.tanh,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  bias_regularizer=self.bias_regularizer,
                                  kernel_constraint=self.kernel_constraint,
                                  bias_constraint=self.bias_constraint)
    self.out = tf.keras.layers.Dense(units=1,
                                  use_bias=self.use_bias,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  bias_regularizer=self.bias_regularizer,
                                  kernel_constraint=self.kernel_constraint,
                                  bias_constraint=self.bias_constraint)

    return

  def call(self, inputs):
    x = self.flt(inputs)
    x = self.dn1(x)
    x = self.dn2(x)
    return self.out(x)


def CondDis1D(data_width, label_width, attn_hds=4):
  """
  construct a discriminator using functional API
  """
  in1 = tf.keras.Input(shape=(data_width,),  name='in1')
  in2 = tf.keras.Input(shape=(data_width,),  name='in2')
  in3 = tf.keras.Input(shape=(label_width,), name='in3')
  out = EncodeDis(width=data_width, name='enc')((in1,in2,in3))
  out = PosMaskedMHABlock(width=data_width,
                          dim=4,
                          heads=attn_hds,
                          name='ma1')(out)
  out = DecodeDis(width=data_width, name='dec')(out)
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
  gen.summary(positions=[0.4, 0.7, 0.8, 1.0])
  out = gen(lbls)
  print(out)

  """
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
  """

  # create a little 'discriminator model'
  dis = CondDis1D(output_shape[1], input_shape[1])
  dis.summary(positions=[0.4, 0.7, 0.8, 1.0])
  out = dis((out[0],out[0],out[1]))
  print(out)
