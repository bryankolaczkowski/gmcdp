from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import initializers, regularizers, constraints, Model
from tensorflow.keras.layers import Layer
import tensorflow as tf

from cond_generator_1d import EncodeLayer, ReluLayer, PosMaskedMHABlock


class EncodeDis(EncodeLayer):
  """
  encodes data for discriminator
  """
  def __init__(self, *args, **kwargs):
    super(EncodeDis, self).__init__(*args, **kwargs)
    return

  def call(self, inputs):
    dt1 = inputs[0]   # data - real or fake?
    dt2 = inputs[1]   # data - definitely fake
    lbl = inputs[2]   # labels
    bs  = tf.shape(lbl)[0]
    pse = tf.tile(self.pos, multiples=(bs,1))   # sequence position encoding
    lpl = self.lpr(self.flt(lbl))               # linear project labels
    return tf.stack((pse, lpl, dt2, dt1), axis=-1)


class DecodeDis(ReluLayer):
  """
  decodes discriminator output to score
  """
  def __init__(self, *args, **kwargs):
    super(DecodeDis, self).__init__(*args, **kwargs)
    # construct
    self.flt = tf.keras.layers.Flatten()
    self.dn1 = tf.keras.layers.Dense(units=self.width,
                                  use_bias=self.use_bias,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  bias_regularizer=self.bias_regularizer,
                                  kernel_constraint=self.kernel_constraint,
                                  bias_constraint=self.bias_constraint)
    self.dn2 = tf.keras.layers.Dense(units=self.width,
                                  use_bias=self.use_bias,
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
    x = tf.nn.leaky_relu(self.dn1(x), alpha=self.relu_alpha)
    x = tf.nn.leaky_relu(self.dn2(x), alpha=self.relu_alpha)
    return self.out(x)


def CondDis1D(data_width,
              label_width,
              attn_hds=4,
              nattnblocks=8,
              l1_penalty=0.01):
  """
  construct a discriminator using functional API
  """
  in1 = tf.keras.Input(shape=(data_width,),  name='in1')
  in2 = tf.keras.Input(shape=(data_width,),  name='in2')
  in3 = tf.keras.Input(shape=(label_width,), name='in3')
  out = EncodeDis(width=data_width, name='enc')((in1,in2,in3))
  for i in range(nattnblocks):
    out = PosMaskedMHABlock(width=data_width,
                            dim=4,
                            heads=attn_hds,
                            name='ma{}'.format(i))(out)
  out = DecodeDis(width=data_width,
                  kernel_regularizer=tf.keras.regularizers.L1(l1=l1_penalty),
                  name='dec')(out)
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

  # create a little 'discriminator model'
  dis = CondDis1D(output_shape[1], input_shape[1])
  dis.summary(positions=[0.4, 0.7, 0.8, 1.0])
  out = dis((out[0],out[0],out[1]))
  print(out)
