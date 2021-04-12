import tensorflow as tf
from . import customlayers


def _buildDenseBlock(filters, use_bias, init, relu_alpha, name, input):
  """
  input -> dense -> leakyReLU
  returns the output after activation
  """
  dl = tf.keras.layers.Dense(filters,
                             use_bias=use_bias,
                             kernel_initializer=init,
                             name=name+'_dns')(input)
  ra = tf.keras.layers.LeakyReLU(alpha=relu_alpha, name=name+'_lrlu')(dl)
  return ra

def _buildLatentSpaceSngl(filters,
                          activation,
                          use_bias,
                          init,
                          name,
                          latent_space):
  """
  builds linear controls from latent_space
  returns a (?,1,filters) control vector
  """
  out = tf.keras.layers.Dense(filters,
                              activation=activation,
                              use_bias=use_bias,
                              kernel_initializer=init,
                              name=name)(latent_space)
  out = tf.keras.layers.Reshape(target_shape=(1,filters),
                                name=name+'_rshp')(out)
  return out

def _buildLatentSpaceBlock(filters,
                           use_bias,
                           init,
                           name,
                           latent_space):
  """
  builds linear transform from latent_space to adaptive controls
  returns data_scale, data_bias, noise_scale tuple
  """
  # data scaling for AdaIN
  nbase = name + '_ds'
  ds = _buildLatentSpaceSngl(filters,
                             tf.keras.activations.softplus,
                             use_bias,
                             init,
                             nbase,
                             latent_space)
  # data bias for AdaIN
  nbase = name + '_bs'
  bs = _buildLatentSpaceSngl(filters,
                             None,
                             use_bias,
                             init,
                             nbase,
                             latent_space)
  # noise scaling for AdaptiveGaussianNoise
  nbase = name + '_ns'
  ns = _buildLatentSpaceSngl(filters,
                             tf.keras.activations.softplus,
                             use_bias,
                             init,
                             nbase,
                             latent_space)
  # return tuple
  return ds, bs, ns

def _buildGenBlock(filters,
                   filtersize,
                   use_bias,
                   init,
                   relu_alpha,
                   name,
                   latent_space,
                   input,
                   choke):
  """
  builds a generator block
  implements noise injection into layers from
    https://arxiv.org/pdf/2006.05891.pdf
  implements adaptive instance normalization from
    https://arxiv.org/pdf/1703.06868.pdf
  adapted from stylegan and stylegan2
    https://arxiv.org/pdf/1812.04948.pdf
    https://arxiv.org/pdf/1912.04958.pdf
  """
  # latent space block
  data_scale, bias, noise_scale = _buildLatentSpaceBlock(filters,
                                                         use_bias,
                                                         init,
                                                         name+'_ada',
                                                         latent_space)

  in1 = input
  # transform input into filters 'feature maps'
  # up/down samples feature maps, providing choke points for dense net
  if choke:
    in1 = tf.keras.layers.Conv1D(filters,
                                 1,
                                 padding='same',
                                 use_bias=use_bias,
                                 kernel_initializer=init,
                                 name=name+'_chk')(input)
    in1 = tf.keras.layers.LeakyReLU(alpha=relu_alpha,
                                    name=name+'_chkact')(in1)
  # convolution-activation block
  con = tf.keras.layers.Conv1D(filters,
                               filtersize,
                               padding='same',
                               use_bias=use_bias,
                               kernel_initializer=init,
                               name=name+'_con')(in1)
  cact = tf.keras.layers.LeakyReLU(alpha=relu_alpha, name=name+'_conact')(con)
  # adaptive noise injection
  noi = customlayers.AdaptiveGaussianNoise(name=name+'_noi')([cact,
                                                              noise_scale])
  # adaptive instance normalization
  nor = customlayers.AdaInstanceNormalization(center=True, scale=True,
                                        name=name+'_nor')([noi,
                                                           bias,
                                                           data_scale])
  # concatenate input to block output (aka 'dense net')
  out = tf.keras.layers.Concatenate(name=name+'_concat')([input,nor])
  return out

def _buildDataSubnet(dim,
                     blocks,
                     filters,
                     filtersize,
                     use_bias,
                     relu_alpha,
                     init,
                     latent_space):
  """
  builds a subnetwork to generate fake data
  """
  namebase = 'gen_dta' # base for layer names
  lname    = namebase + '_lin'
  # linear transformations of constant input to data dimensions x filters
  #lin = customlayers.LinearInput(dim,
  #                               filters,
  #                               use_bias=use_bias,
  #                               kernel_initializer=init,
  #                               name=lname)(latent_space)
  # apply style-specific noise
  #ns = _buildLatentSpaceSngl(filters,
  #                           use_bias,
  #                           init,
  #                           relu_alpha,
  #                           lname+'_ns',
  #                           latent_space)
  #out = customlayers.AdaptiveGaussianNoise(name=lname+'_noi')([lin,ns])
  mns = _buildLatentSpaceSngl(dim,
                              None,
                              use_bias,
                              init,
                              lname+'_mean',
                              latent_space)
  sdv = _buildLatentSpaceSngl(dim,
                              tf.keras.activations.softplus,
                              use_bias,
                              init,
                              lname+'_stdv',
                              latent_space)
  out = customlayers.NoisyInput(name=lname)([mns,sdv])
  out = tf.keras.layers.Reshape(target_shape=(dim,1), name=lname+'_rshp')(out)
  # data generator blocks
  for i in range(blocks):
    nbase = namebase + '_blk{}'.format(i)
    out = _buildGenBlock(filters,
                         filtersize,
                         use_bias,
                         init,
                         relu_alpha,
                         nbase,
                         latent_space,
                         out,
                         True)
  # bidirectional LSTM over concatenated feature maps
  bdr = tf.keras.layers.Bidirectional(
                              tf.keras.layers.LSTM(filters//2,
                                                   kernel_initializer=init,
                                                   return_sequences=True),
                              name='gen_lstm')(out)
  # final generator output becomes data
  out = tf.keras.layers.Conv1D(1,
                               filtersize,
                               padding='same',
                               use_bias=use_bias,
                               kernel_initializer=init,
                               name='gen_out')(bdr)
  return out

def _buildLabelSubnet(dim,
                      blocks,
                      filters,
                      outputs,
                      use_bias,
                      relu_alpha,
                      init):
  """
  subnetwork learns label->latent_space mapping
  """
  namebase = 'gen_labl' # base for layer names
  # input layers
  lin = tf.keras.layers.Input(shape=dim, name=namebase+'_in')
  out = customlayers.NoisyBinaryOneHotEncoding(name=namebase+'_onehot')(lin)
  # dense blocks
  for i in range(blocks):
    nbase = namebase + '_blk{}'.format(i)
    out = _buildDenseBlock(filters, use_bias, init, relu_alpha, nbase, out)
  return (lin,out)

def buildGenerator(data_dim=None,
                   data_blocks=4,
                   data_filters=128,
                   data_filtersize=3,
                   data_bias=False,
                   data_relu_alpha=0.2,
                   data_initializer=None,

                   label_dim=None,
                   label_blocks=4,
                   label_filters=128,
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
    data_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)
  if label_initializer is None:
    label_initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.1)

  # build label subnetwork
  input,latent = _buildLabelSubnet(label_dim,
                                   label_blocks,
                                   label_filters,
                                   data_blocks,
                                   label_bias,
                                   label_relu_alpha,
                                   label_initializer)

  # build data generator subnetwork
  output = _buildDataSubnet(data_dim,
                            data_blocks,
                            data_filters,
                            data_filtersize,
                            data_bias,
                            data_relu_alpha,
                            data_initializer,
                            latent)

  return tf.keras.Model(inputs=input,
                        outputs=(output,input),
                        name='generator')
