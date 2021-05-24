from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import initializers, regularizers, constraints, Model
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils
import tensorflow as tf

from encoding1d import BinaryOneHotEncoding
from wrappers   import SpecNorm


class DisBlock(Layer):
  """
  encapsulates a single block of layers in a discriminator
  """
  def __init__(self,
               filters,
               kernel_size=3,
               use_bias=True,
               kernel_initializer='glorot_normal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               relu_alpha=0.2,
               spec_norm=True,
               **kwargs):
    super(DisBlock, self).__init__(**kwargs)
    # config copy
    self.filters            = filters
    self.kernel_size        = conv_utils.normalize_tuple(kernel_size, 1,
                                                         'kernel_size')
    self.use_bias           = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer   = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer   = regularizers.get(bias_regularizer)
    self.kernel_constraint  = constraints.get(kernel_constraint)
    self.bias_constraint    = constraints.get(bias_constraint)
    self.relu_alpha         = relu_alpha
    self.spec_norm          = spec_norm
    # construct
    self.conv1 = tf.keras.layers.Conv1D(self.filters,
                                    kernel_size=self.kernel_size,
                                    strides=1,
                                    padding='same',
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    self.conv2 = tf.keras.layers.Conv1D(self.filters,
                                    kernel_size=self.kernel_size,
                                    strides=1,
                                    padding='same',
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    if self.spec_norm:
      self.conv1 = SpecNorm(self.conv1)
      self.conv2 = SpecNorm(self.conv2)
    return

  def call(self, inputs):
    """
    inputs should be [N,W,C] 1D image
    """
    # convolution blocks
    x = self.conv1(inputs)
    x = tf.nn.leaky_relu(x, alpha=self.relu_alpha)
    x = self.conv2(inputs)
    x = tf.nn.leaky_relu(x, alpha=self.relu_alpha)
    # concatenate results to inputs
    x = tf.concat([inputs,x], axis=-1)
    # downsample by 1/2
    x = tf.nn.avg_pool1d(x, ksize=2, strides=2, padding='VALID')
    return x

  def get_config(self):
    config = super(DisBlock, self).get_config()
    config.update({
      'filters'            : self.filters,
      'kernel_size'        : self.kernel_size,
      'use_bias'           : self.use_bias,
      'kernel_initializer' : initializers.serialize(self.kernel_initializer),
      'bias_initializer'   : initializers.serialize(self.bias_initializer),
      'kernel_regularizer' : regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer'   : regularizers.serialize(self.bias_regularizer),
      'kernel_constraint'  : constraints.serialize(self.kernel_constraint),
      'bias_constraint'    : constraints.serialize(self.bias_constraint),
      'relu_alpha'         : self.relu_alpha,
      'spec_norm'          : self.spec_norm,
    })
    return config


class ProjectLabelsToData(Layer):
  """
  projects a set of labels as additional channels across data width
  """
  def __init__(self, **kwargs):
    super(ProjectLabelsToData, self).__init__(**kwargs)
    self.proj  = 1
    self.reshp = None
    return

  def build(self, input_shape):
    data_shape = input_shape[0]
    labl_shape = input_shape[1]
    # set label projection multiplier to width of data (N,W,C)
    self.proj  = data_shape[1]
    # set label reshape to (N,1,W*C)
    self.reshp = tf.keras.layers.Reshape(target_shape=(1,
                                         labl_shape[1]*labl_shape[2]))
    return

  def call(self, inputs):
    """
    inputs should be [data, labels], with data an [N,W,C] image
    output is new [data], with labels projected along width as new channels
    """
    data = inputs[0]
    lbls = inputs[1]
    # labels have shape (N,W,C); reshape to (N,1,W*C)
    lbls = self.reshp(lbls)
    # now all labels are in the channel dimension
    # tile labels to get projection
    lbls = tf.tile(lbls, multiples=(1, self.proj, 1))
    # concatenate projected labels to data
    data = tf.concat([data, lbls], axis=-1)
    return data

  def get_config(self):
    return super(ProjectLabelsToData, self).get_config()


def MCDis(data_width0=1024,
          data_channels=1,
          label_shape=(1,),
          desc_layer_units=256,
          filters=[32,32,64,64,128,128,256,256],
          kernel_size=3,
          use_bias=True,
          kernel_initializer='glorot_normal',
          bias_initializer='zeros',
          kernel_regularizer=None,
          bias_regularizer=None,
          kernel_constraint=None,
          bias_constraint=None,
          relu_alpha=0.2,
          spectral_normalization=True,
          **kwargs):
    """
    returns a discriminator Model using the functional API
    """
    ## build input layers
    data_in = tf.keras.Input(shape=(data_width0, data_channels))
    labl_in = tf.keras.Input(shape=label_shape)
    enc_out = BinaryOneHotEncoding()(labl_in)
    output  = ProjectLabelsToData()((data_in, enc_out))
    ## build discriminator blocks
    for fltrs in filters:
      output = DisBlock(filters=fltrs,
                        kernel_size=kernel_size,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint,
                        relu_alpha=relu_alpha,
                        spec_norm=spectral_normalization)(output)
    ## flatten and discriminate
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(units=1,
                                   use_bias=use_bias,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   bias_regularizer=bias_regularizer,
                                   kernel_constraint=kernel_constraint,
                                   bias_constraint=bias_constraint)(output)
    ## build model
    return Model(inputs=(data_in, labl_in), outputs=output,
                 name='mc_dis', **kwargs)


if __name__ == '__main__':
  """
  module example
  """
  import generator1d
  nbatches = 2
  labelshp = (nbatches,1)
  datashp  = (nbatches,1024,1)
  # create labels
  labels = tf.math.rint(tf.random.uniform(shape=labelshp))
  print('LABELS', labels)
  # create data
  data = tf.random.normal(shape=datashp)
  print('DATA', data)
  # create generator
  gen = generator1d.MCGen()
  # generate data
  data = gen((data,labels))
  print('GENDATA', data)

  # test discriminator forward pass
  discriminator = MCDis()
  discriminator.compile()
  discriminator.summary(line_length=200, positions=[0.35,0.7,0.8,1.0])
  out = discriminator(data)
  print('OUTPUT', out)

  # test serialization
  discriminator.save('TMP.model')
  m = tf.keras.models.load_model('TMP.model')
  m.summary(line_length=200, positions=[0.35,0.7,0.8,1.0])
  out = m(data)
  print('OUTPUT', out)
