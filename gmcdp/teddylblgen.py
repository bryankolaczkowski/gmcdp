from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .activ import lrsact

class TeddyInternalLayer(tf.keras.layers.Dense):
  def __init__(self, *args, **kwargs):
    super(TeddyInternalLayer, self).__init__(*args, **kwargs)
    return

    def call(self, inputs):
      return lrsact(super(TeddyInternalLayer, self).call(inputs))


def TeddyLabelGen(one_hot_sets,
                  units=128,
                  layers=8,
                  input_len=64):
  # random gaussian input
  inputs = tf.keras.Input(shape=(input_len,), name='rndin')
  output = inputs
  # internal mapping layers
  for i in range(layers):
    output = TeddyInternalLayer(units=units,
                                name='dns{}'.format(i))(output)
  # split one-hot outputs
  oh_layers = []
  for i in range(len(one_hot_sets)):
    cats = one_hot_sets[i]
    if cats == 1:
      lyr = tf.keras.layers.Dense(units=cats,
                                  activation='sigmoid',
                                  name='out{}'.format(i))(output)
    else:
      lyr = tf.keras.layers.Dense(units=cats,
                                  activation='softmax',
                                  name='out{}'.format(i))(output)
    oh_layers.append(lyr)
  # combine one-hot outputs
  output = tf.keras.layers.Concatenate(name='cnct')(oh_layers)
  output = tf.keras.layers.Flatten(name='fltn')(output)
  # done
  return tf.keras.Model(inputs=inputs, outputs=output)
