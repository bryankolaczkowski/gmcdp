from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Layer
import tensorflow as tf


class BinaryOneHotEncoding(Layer):
  """
  encodes a [0,1] integer as one-hot
  """
  def __init__(self, **kwargs):
    super(BinaryOneHotEncoding, self).__init__(**kwargs)
    return

  def build(self, input_shape):
    return

  def call(self, inputs):
    return tf.one_hot(tf.cast(inputs, tf.int32), 2)

  def get_config(self):
    return super(BinaryOneHotEncoding, self).get_config()


class NoisyBinaryOneHotEncoding(BinaryOneHotEncoding):
  """
  one-hot encoding with added noise
  """
  def __init__(self, **kwargs):
    super(NoisyBinaryOneHotEncoding, self).__init__(**kwargs)
    return

  def call(self, inputs):
    onehot = super(NoisyBinaryOneHotEncoding, self).call(inputs)
    noise  = tf.random.normal(shape=tf.shape(onehot))
    return onehot * noise
