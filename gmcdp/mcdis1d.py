from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import Model
import tensorflow as tf

from .mcgen1d import EncodeLayer, DecodeGen, PosMaskedMHABlock
from .wrapr   import SpecNorm


class EncodeDis(EncodeLayer):
  """
  encodes data for discriminator
  """
  def __init__(self, *args, **kwargs):
    super(EncodeDis, self).__init__(*args, **kwargs)
    # construct
    self.dp1 = tf.keras.layers.Dense(units=self.dim,
                                  use_bias=self.use_bias,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  bias_regularizer=self.bias_regularizer,
                                  kernel_constraint=self.kernel_constraint,
                                  bias_constraint=self.bias_constraint)
    self.dp2 = tf.keras.layers.Dense(units=self.dim,
                                  use_bias=self.use_bias,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  bias_regularizer=self.bias_regularizer,
                                  kernel_constraint=self.kernel_constraint,
                                  bias_constraint=self.bias_constraint)
    return

  def call(self, inputs):
    dt1 = tf.expand_dims(inputs[0], axis=-1)  # data - real or fake?
    dt2 = tf.expand_dims(inputs[1], axis=-1)  # data - definitely fake
    lbl = inputs[2]                           # labels
    bs  = tf.shape(lbl)[0]                    # batch size
    # sequence position encoding
    pse = tf.tile(self.pos, multiples=(bs,1,1))
    # linear project labels
    lpl = self.lpr(self.flt(lbl))
    lpl = tf.reshape(lpl, shape=(bs,self.width,self.dim))
    # linear project data 2
    dt2 = self.dp2(dt2)
    # linear project data 1
    dt1 = self.dp1(dt1)
    return tf.concat((pse, lpl, dt2, dt1), axis=-1)


class DecodeDis(DecodeGen):
  """
  decodes discriminator output to score
  """
  def __init__(self, *args, **kwargs):
    super(DecodeDis, self).__init__(*args, **kwargs)
    self.out = SpecNorm(self.out)
    return

  def _finalize(self, inputs):
    return self.out(self.flt(inputs))


def CondDis1D(data_width,
              label_width,
              attn_hds=4,
              nattnblocks=8,
              lbldim=4,
              dropout=0.2):
  """
  construct a discriminator using functional API
  """
  datadim = lbldim * 3 + 1
  in1 = tf.keras.Input(shape=(data_width,),  name='in1')
  in2 = tf.keras.Input(shape=(data_width,),  name='in2')
  in3 = tf.keras.Input(shape=(label_width,), name='in3')
  out = EncodeDis(width=data_width, dim=lbldim+1, name='enc')((in1,in2,in3))
  for i in range(nattnblocks):
    out = PosMaskedMHABlock(width=data_width,
                            dim=datadim,
                            heads=attn_hds,
                            dropout=dropout,
                            name='ma{}'.format(i))(out)
  out = DecodeDis(width=data_width, name='dec')(out)
  return Model(inputs=(in1,in2,in3), outputs=out)
