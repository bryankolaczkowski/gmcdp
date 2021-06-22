from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend as K
from tensorflow.keras import Model, optimizers
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer
import tensorflow as tf

from cond_generator_1d     import CondGen1D
from cond_discriminator_1d import CondDis1D


class WassersteinLoss(Loss):
  """
  implements wasserstein loss function

  'earth mover' distance from:
    https://arxiv.org/pdf/1701.07875.pdf
    https://arxiv.org/pdf/1704.00028.pdf
  """
  def __init__(self):
    super(WassersteinLoss, self).__init__(name='wasserstein_loss')
    return

  def call(self, y_true, y_pred):
    return K.mean(y_true * y_pred)


class GanOptimizer(Optimizer):
  """
  implements a generator,discriminator optimizer pair
  """
  def __init__(self,
               gen_optimizer='adam',
               dis_optimizer='adam',
               **kwargs):
    super(GanOptimizer, self).__init__(name='GanOptimizer', **kwargs)
    self.gen_optimizer = optimizers.get(gen_optimizer)
    self.dis_optimizer = optimizers.get(dis_optimizer)
    return

  def apply_gradients(self, grads_and_vars,
                      name=None, experimental_aggregate_gradients=True):
    raise NotImplementedError('GAN optimizer should call '
                              'apply_generator_gradients and '
                              'apply_discriminator_gradients instead')

  def apply_generator_gradients(self, grads_and_vars):
    return self.gen_optimizer.apply_gradients(grads_and_vars)

  def apply_discriminator_gradients(self, grads_and_vars):
    return self.dis_optimizer.apply_gradients(grads_and_vars)

  def get_config(self):
    config = super(GanOptimizer, self).get_config()
    config.update({
      'gen_optimizer' : tf.keras.optimizers.serialize(self.gen_optimizer),
      'dis_optimizer' : tf.keras.optimizers.serialize(self.dis_optimizer),
    })
    return config


class CondGan1D(Model):
  """
  microbial community generative adversarial network
  """
  def __init__(self,
               generator,
               discriminator,
               pack_dim,
               **kwargs):
    super(CondGan1D, self).__init__(**kwargs)
    self.genr     = generator
    self.disr     = discriminator
    self.pack_dim = pack_dim
    return

  def compile(self,
              optimizer=GanOptimizer(),
              loss=WassersteinLoss(),
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              steps_per_execution=None,
              **kwargs):
    super(CondGan1D, self).compile(optimizer=optimizer,
                               loss=loss,
                               metrics=metrics,
                               loss_weights=loss_weights,
                               weighted_metrics=weighted_metrics,
                               run_eagerly=run_eagerly,
                               steps_per_execution=steps_per_execution,
                               **kwargs)
    return

  def call(self, inputs):
    #x = self.pack(self.genr(inputs))
    #x = self.disr(x)
    x = self.genr(inputs[1])
    return x

  def augment(self, inputs):
    """
    data augmenation function
    """
    dta = inputs[0]
    lbl = inputs[1]
    """
    ## noisify random data entries
    mask = tf.cast(tf.random.categorical(tf.math.log([[0.9, 0.1]]),
                                         tf.math.reduce_prod(tf.shape(dta))),
                                         tf.float32)
    mask = tf.reshape(mask, shape=tf.shape(dta))
    nois = tf.random.normal(shape=tf.shape(dta)) * mask
    dta  = dta + nois
    ## obliterate random data entries
    mask = tf.cast(tf.random.categorical(tf.math.log([[0.1, 0.9]]),
                                         tf.math.reduce_prod(tf.shape(dta))),
                                         tf.float32)
    mask = tf.reshape(mask, shape=tf.shape(dta))
    dta  = dta * mask
    """
    return inputs

  def pack(self, inputs):
    """
    convert generator output for 'packed' discriminator
    """
    ## pack data
    dta = inputs[0] # data  shape is (bs, width)
    # get new packed batch size
    dta_shp = tf.shape(dta)
    bs  = dta_shp[0] // self.pack_dim
    dw  = dta_shp[1]
    dta = tf.reshape(dta, shape=(bs,dw,-1))
    ## pack labels
    lbl = inputs[1] # label shape is (bs, labels)
    lw  = tf.shape(lbl)[1]
    lbl = tf.reshape(lbl, shape=(bs,lw,-1))
    return (dta,lbl)

  def train_step(self, inputs):
    """
    single training step
    """
    bs = tf.shape(inputs[0])[0]

    # labels need to take into account discriminator's pack_dim
    #p_bs  = bs // self.pack_dim
    pones =  tf.ones((bs,1))
    nones = -tf.ones((bs,1))

    data = inputs[0]
    lbls = inputs[1]

    # train discriminator using real data
    with tf.GradientTape() as tape:
      #preds   = self.disr(self.pack(self.augment(inputs)))
      preds   = self.disr((data, self.genr(lbls)[0]))
      disr_rl = self.compiled_loss(nones, preds)
    grds = tape.gradient(disr_rl, self.disr.trainable_weights)
    self.optimizer.apply_discriminator_gradients(zip(grds,
                                                 self.disr.trainable_weights))

    # train discriminator and generator using fake data
    with tf.GradientTape() as tape:
      #fake_data = self.pack(self.augment(self.genr(inputs[1])))
      #preds     = self.disr(fake_data)
      preds   = self.disr((self.genr(lbls)[0], self.genr(lbls)[0]))
      disr_fk = self.compiled_loss(pones, preds)
    grds = tape.gradient(disr_fk, self.disr.trainable_weights)
    self.optimizer.apply_discriminator_gradients(zip(grds,
                                                 self.disr.trainable_weights))

    # train generator
    with tf.GradientTape() as tape:
      #fake_data = self.pack(self.augment(self.genr(inputs[1])))
      # calculate discriminator-induced loss
      #preds     = self.disr(fake_data)
      preds     = self.disr((self.genr(lbls)[0], self.genr(lbls)[0]))
      genr_loss = self.compiled_loss(nones, preds)
    grds = tape.gradient(genr_loss, self.genr.trainable_weights)
    self.optimizer.apply_generator_gradients(zip(grds,
                                             self.genr.trainable_weights))

    return {'disr_rl'   : disr_rl,
            'disr_fk'   : disr_fk,
            'genr_loss' : genr_loss,}

  def get_config(self):
    config = super(CondGan1D, self).get_config()
    config.update({
      'generator'     : tf.keras.layers.serialize(self.genr),
      'discriminator' : tf.keras.layers.serialize(self.disr),
      'pack_dim'      : self.pack_dim,
    })
    return config


if __name__ == '__main__':
  """
  module example test
  """
  import sys
  sys.path.append("../tests")
  import test_data_generator

  ndata     = 16344
  #ndata     = 262144
  batchsize = 128
  pack_dim  = 1

  # generate simulated data and labels
  dtas,lbls = test_data_generator.gen_dataset(ndata, plot=False)
  print(dtas,lbls)
  lbl_shp = tf.shape(lbls)
  dta_shp = tf.shape(dtas)

  # create a little 'generator model' that maps the label vector
  # to data space
  gen = CondGen1D((lbl_shp[1],), dta_shp[1])
  gen.summary(positions=[0.3, 0.75, 0.85, 1.0])

  # create a little 'packed discriminator model'
  dis = CondDis1D(dta_shp[1], lbl_shp[1], pack_dim=pack_dim)
  dis.summary(positions=[0.3, 0.75, 0.85, 1.0])

  # package data into dataset
  data = tf.data.Dataset.from_tensor_slices((dtas, lbls))
  data = data.shuffle(ndata).batch(batchsize)

  # create optimizer
  gopt = tf.keras.optimizers.Nadam(learning_rate=1e-5, beta_1=0.5, beta_2=0.9)
  dopt = tf.keras.optimizers.Nadam(learning_rate=1e-6, beta_1=0.5, beta_2=0.9)
  opt  = GanOptimizer(gopt, dopt)

  # create gan
  gan = CondGan1D(generator=gen, discriminator=dis, pack_dim=pack_dim)
  gan.compile(optimizer=opt)

  # generated data image callback
  import io
  import numpy as np
  import matplotlib.pyplot as plt
  class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir='logs'):
      self.writer = tf.summary.create_file_writer(log_dir + '/gen')
      return

    def plot_data(self, data):
      x = np.arange(0,tf.shape(data)[1],1)
      fig = plt.figure()
      ax  = fig.add_subplot(111)
      y   = data.numpy().transpose()
      #y.sort(axis=0)
      #y = np.flip(y, axis=0)
      ax.plot(x, y, 'o', markersize=2, alpha=0.5)
      ax.set_ylim([-5,+5])
      return fig

    def plot_to_image(self, plot):
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      plt.close(plot)
      buf.seek(0)
      image = tf.image.decode_png(buf.getvalue(), channels=4)
      image = tf.expand_dims(image, 0)
      return image

    def on_epoch_end(self, epoch, logs=None):
      # generate 20 example datas
      x = self.model((dtas[0:20],lbls[0:20]))[0]
      fig = self.plot_data(x)
      img = self.plot_to_image(fig)
      with self.writer.as_default():
        tf.summary.image('GenData', img, step=epoch)
      return

  # fit gan
  gan.fit(data,
          epochs=10000,
          callbacks=[tf.keras.callbacks.TensorBoard(),
                     PlotCallback()])
