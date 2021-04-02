import tensorflow as tf
from . import customlayers


def _buildDenseBlock(filters, use_bias, init, relu_alpha, name, input):
  """
  builds a dense block consisting of a dense layer connected to input,
  with relu activation
  returns the output after activation
  """
  dl = tf.keras.layers.Dense(filters,
                             use_bias=use_bias,
                             kernel_initializer=init,
                             name=name+'_dense')(input)
  ra = tf.keras.layers.LeakyReLU(alpha=relu_alpha, name=name+'_lrelu')(dl)
  return ra

def _buildGenBlock(filters,
                   filtersize,
                   use_bias,
                   init,
                   relu_alpha,
                   name,
                   beta,
                   gamma,
                   input):
  """
  builds a generator block
  implements noise injection into layers from
    https://arxiv.org/pdf/2006.05891.pdf
  implements adaptive instance normalization from
    https://arxiv.org/pdf/1703.06868.pdf
  adapted from stylegan paper
    https://arxiv.org/pdf/1812.04948.pdf
  """
  con = tf.keras.layers.Conv1D(filters,
                               filtersize,
                               padding='same',
                               use_bias=use_bias,
                               kernel_initializer=init,
                               name=name+'_conv')(input)
  nor = customlayers.AdaInstanceNormalization(
                     name=name+'_adain')([con,beta,gamma])
  act = tf.keras.layers.LeakyReLU(alpha=relu_alpha, name=name+'_lrelu')(nor)
  sig = tf.keras.layers.MaxPool1D(pool_size=filtersize,
                                  strides=1,
                                  padding='same',
                                  name=name+'_maxpool')(act)
  noi = customlayers.GausNoiseOn(stddev=1.0, name=name+'_noise')(sig)
  out = tf.keras.layers.Add(name=name+'_add')([act,noi])
  return out

def _buildDataSubnet(dim,
                     input_filters,
                     blocks,
                     filters,
                     filtersize,
                     use_bias,
                     relu_alpha,
                     init,
                     betas,
                     gammas):
  """
  builds a subnetwork to generate fake data
  """
  namebase = 'gen_data' # base for layer names
  lname    = namebase + '_linear_in'
  # linear transformation of constant input to data dimensions
  con = tf.keras.layers.Input(shape=(1), name=namebase+'_const')
  lin = tf.keras.layers.Dense(dim * input_filters,
                              use_bias=use_bias,
                              kernel_initializer=init,
                              name=lname)(con)
  out = tf.keras.layers.Reshape((dim,input_filters),
                                name=lname+'_rshp')(lin)
  # data generator blocks
  for i in range(blocks):
    nbase = namebase + '_block{}'.format(i)
    out = _buildGenBlock(filters,
                         filtersize,
                         use_bias,
                         init,
                         relu_alpha,
                         nbase,
                         betas[i],
                         gammas[i],
                         out)
  # final generator output becomes data
  out = tf.keras.layers.Conv1D(1,
                               filtersize,
                               padding='same',
                               use_bias=use_bias,
                               kernel_initializer=init,
                               name='gen_out')(out)
  return (con,out)

def _buildLabelSubnet(dim,
                      blocks,
                      filters,
                      outputs,
                      use_bias,
                      relu_alpha,
                      init,
                      output_filters):
  """
  builds a subnetwork to transform input labels into beta,gamma values for
  controlling adaptive instance normalization
  """
  betas    = []          # will hold beta outputs for instance normalization
  gammas   = []          # will hold gamma outputs for instance normalization
  namebase = 'gen_label' # base for layer names
  # input layers
  lin = tf.keras.layers.Input(shape=dim, name=namebase+'_input')
  out = tf.keras.layers.Flatten(name=namebase+'_input_flatten')(lin)
  # dense blocks with leaky_relu activation
  for i in range(blocks):
    nbase = namebase + '_block{}'.format(i)
    out = _buildDenseBlock(filters, use_bias, init, relu_alpha, nbase, out)
  # output layers
  for i in range(outputs):
    bnm  = namebase + '_outbeta{}'.format(i)
    beta = tf.keras.layers.Dense(output_filters,
                                 use_bias=use_bias,
                                 kernel_initializer=init,
                                 name=bnm)(out)
    beta = tf.keras.layers.Reshape((1,output_filters),
                                   name=bnm+'_rshp')(beta)
    gnm  = namebase + '_outgamma{}'.format(i)
    gama = tf.keras.layers.Dense(output_filters,
                                 use_bias=use_bias,
                                 kernel_initializer=init,
                                 name=gnm)(out)
    gama = tf.keras.layers.Reshape((1,output_filters),
                                   name=gnm+'_rshp')(gama)
    betas.append(beta)
    gammas.append(gama)
  return (lin,betas,gammas)

def buildGenerator(data_dim=None,
                   data_input_filters=32,
                   data_blocks=4,
                   data_filters=128,
                   data_filtersize=3,
                   data_bias=False,
                   data_relu_alpha=0.2,
                   data_initializer=None,

                   label_dim=None,
                   label_blocks=4,
                   label_filters=64,
                   label_bias=False,
                   label_relu_alpha=0.2,
                   label_initializer=None):
  """
  builds a generator model
  """
  # some basic error checking
  if data_dim is None:
    raise ValueError('data dimension None is a poor choice')
  if label_dim is None:
    raise ValueError('label dimension None is a poor choice')

  # setup default initializers
  if data_initializer is None:
    data_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
  if label_initializer is None:
    label_initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.02)

  # build label subnetwork
  input,betas,gammas = _buildLabelSubnet(label_dim,
                                         label_blocks,
                                         label_filters,
                                         data_blocks,
                                         label_bias,
                                         label_relu_alpha,
                                         label_initializer,
                                         data_filters)

  # build data generator subnetwork
  con,output = _buildDataSubnet(data_dim,
                                data_input_filters,
                                data_blocks,
                                data_filters,
                                data_filtersize,
                                data_bias,
                                data_relu_alpha,
                                data_initializer,
                                betas,
                                gammas)

  return tf.keras.Model(inputs=(input,con),
                        outputs=(input,output),
                        name='generator')
