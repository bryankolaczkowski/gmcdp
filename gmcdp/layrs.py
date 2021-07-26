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
    # config copy
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


class WidthLayer(ConfigLayer):
  """
  base class for configurable layer with data width
  """
  def __init__(self, width, *args, **kwargs):
    super(WidthLayer, self).__init__(*args, **kwargs)
    # config copy
    self.width = width
    return

  def get_config(self):
    config = super(WidthLayer, self).get_config()
    config.update({
      'width' : self.width,
    })
    return config


class ReluLayer(WidthLayer):
  """
  base class for layers with leaky-ReLU activations
  """
  def __init__(self, relu_alpha=0.4, *args, **kwargs):
    super(ReluLayer, self).__init__(*args, **kwargs)
    # config copy
    self.relu_alpha = relu_alpha
    return

  def get_config(self):
    config = super(ReluLayer, self).get_config()
    config.update({
      'relu_alpha' : self.relu_alpha,
    })
    return config
